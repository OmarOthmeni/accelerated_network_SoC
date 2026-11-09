#!/usr/bin/env python3
"""Post-training INT8 quantization + bit-exact integer reference model.

Integer arithmetic (identical in golden_infer() below and in hls/cnn.cpp):
  conv layer l : acc[o] = b_q[o] + sum(w_q[o,i,ky,kx] * a_q[i, y+ky-1, x+kx-1])   (int32, zero padding)
                 a_out  = clamp((acc * M[o] + 2^(s[o]-1)) >> s[o], 0, 255)         (uint8; ReLU = clamp at 0)
                 then 2x2 max pooling on the uint8 values
  global avg   : g[i]   = sum over the 4x4 map of a_q[i]   (the 1/16 is folded into the FC scale)
  fc           : logit[k] = fcb_q[k] + sum(fcw_q[k,i] * g[i])                       (int32)
Weights int8 (symmetric; per output channel for conv, per tensor for fc so that argmax works on
raw int32 logits), activations uint8 with zero point 0, biases int32.

Outputs:
  hls/weights.h            constant arrays for the HLS kernel
  hls/data/test_images.bin uint8 test images, hls/data/test_labels.bin, hls/data/golden_logits.bin
  model/out/int8_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from net import HwCNN, CHANNELS
from train import load_split


def fold_bn(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    gamma, beta = bn.weight.detach().numpy(), bn.bias.detach().numpy()
    mean, var = bn.running_mean.numpy(), bn.running_var.numpy()
    k = gamma / np.sqrt(var + bn.eps)
    return w * k[:, None, None, None], (beta - mean * k).astype(np.float64)


def quant_multiplier(real):
    """real (0 < real < 1) -> (m, s) with real ~= m * 2^-s, m in [2^30, 2^31)."""
    m, e = np.frexp(real)                     # real = m * 2^e, m in [0.5, 1)
    mq = np.round(m * (1 << 31)).astype(np.int64)
    s = (31 - e).astype(np.int64)
    over = mq == (1 << 31)                    # rounding overflow
    mq[over] //= 2
    s[over] -= 1
    assert (s >= 1).all() and (s <= 62).all(), "requantization shift out of range"
    return mq, s


# ---------------------------------------------------------------- float model with folded BN
def float_layers(x01, layers, fc_w, fc_b, collect=None):
    """x01: N x 1 x 64 x 64 in [0,1]. Returns logits; optionally collects post-ReLU activations."""
    t = torch.from_numpy(x01).double()
    for li, (w, b) in enumerate(layers):
        t = torch.nn.functional.conv2d(t, torch.from_numpy(w), torch.from_numpy(b), padding=1)
        t = torch.relu(t)
        if collect is not None:
            collect[li].append(t.numpy().ravel())
        t = torch.nn.functional.max_pool2d(t, 2)
    g = t.mean(dim=(2, 3))
    return (g @ torch.from_numpy(fc_w).T + torch.from_numpy(fc_b)).numpy()


# ---------------------------------------------------------------- integer golden model
def conv_int(a, w_q, b_q):
    """a: N x C x H x W (int64), w_q: O x C x 3 x 3 -> N x O x H x W int64 accumulators."""
    n, c, h, wd = a.shape
    p = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(p, (3, 3), axis=(2, 3))  # N C H W 3 3
    acc = np.einsum("nchwyx,ocyx->nohw", cols, w_q.astype(np.int64), optimize=True)
    return acc + b_q[None, :, None, None]


def golden_infer(img_u8, q):
    a = img_u8.astype(np.int64)[:, None]                      # N x 1 x 64 x 64
    for l in range(len(CHANNELS) - 1):
        acc = conv_int(a, q[f"w{l}"], q[f"b{l}"])
        m = q[f"m{l}"][None, :, None, None]
        s = q[f"s{l}"][None, :, None, None]
        r = (acc * m + (np.int64(1) << (s - 1))) >> s
        a = np.clip(r, 0, 255)
        n, c, h, w = a.shape
        a = a.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))
    g = a.sum(axis=(2, 3))                                     # N x 64
    return q["fcb"][None, :] + g @ q["fcw"].astype(np.int64).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default=Path("model/out"), type=Path)
    ap.add_argument("--hls", default=Path("hls"), type=Path)
    ap.add_argument("--calib", type=int, default=1000, help="training images used for calibration")
    ap.add_argument("--percentile", type=float, default=99.99)
    a = ap.parse_args()

    classes = json.loads((a.model / "classes.json").read_text())
    net = HwCNN(len(classes))
    net.load_state_dict(torch.load(a.model / "hwcnn_fp32.pt"))
    net.eval()

    mods = list(net.features)
    layers = [fold_bn(mods[4 * i], mods[4 * i + 1]) for i in range(len(CHANNELS) - 1)]
    fc_w = net.fc.weight.detach().numpy().astype(np.float64)
    fc_b = net.fc.bias.detach().numpy().astype(np.float64)
    layers = [(w.astype(np.float64), b.astype(np.float64)) for w, b in layers]

    xtr, _ = load_split(a.data / "train", classes)
    xte, yte = load_split(a.data / "test", classes)
    rng = np.random.default_rng(0)
    calib = xtr[rng.choice(len(xtr), min(a.calib, len(xtr)), replace=False)]

    # ---- calibration: per-layer output range of the post-ReLU activations
    acts = [[] for _ in layers]
    for i in range(0, len(calib), 100):
        float_layers(calib[i:i + 100, None] / 255.0, layers, fc_w, fc_b, collect=acts)
    s_act = [1.0 / 255.0]                                     # input: pixel / 255
    for li in range(len(layers)):
        hi = np.percentile(np.concatenate(acts[li]), a.percentile)
        s_act.append(hi / 255.0)

    q = {}
    for l, (w, b) in enumerate(layers):
        s_w = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-12) / 127.0
        q[f"w{l}"] = np.clip(np.round(w / s_w[:, None, None, None]), -127, 127).astype(np.int8)
        q[f"b{l}"] = np.round(b / (s_act[l] * s_w)).astype(np.int64)
        q[f"m{l}"], q[f"s{l}"] = quant_multiplier(s_act[l] * s_w / s_act[l + 1])
    s_g = s_act[-1] / 16.0                                    # avg pool = sum / 16
    s_fc = np.abs(fc_w).max() / 127.0
    q["fcw"] = np.clip(np.round(fc_w / s_fc), -127, 127).astype(np.int8)
    q["fcb"] = np.round(fc_b / (s_g * s_fc)).astype(np.int64)

    # ---- accuracy: float (folded) vs integer
    fl = np.concatenate([float_layers(xte[i:i + 100, None] / 255.0, layers, fc_w, fc_b)
                         for i in range(0, len(xte), 100)])
    gi = np.concatenate([golden_infer(xte[i:i + 100], q) for i in range(0, len(xte), 100)])
    assert np.abs(gi).max() < 2**31, "int32 overflow in logits"

    def scores(pred):
        acc = float((pred == yte).mean())
        bal = float(np.mean([(pred[yte == c] == c).mean() for c in range(len(classes)) if (yte == c).any()]))
        return acc, bal

    f_acc, f_bal = scores(fl.argmax(1))
    i_acc, i_bal = scores(gi.argmax(1))
    agree = float((fl.argmax(1) == gi.argmax(1)).mean())
    print(f"float32 (BN folded): accuracy {f_acc:.4f} | balanced {f_bal:.4f}")
    print(f"int8 golden        : accuracy {i_acc:.4f} | balanced {i_bal:.4f} | top-1 agreement {agree:.4f}")

    # ---- exports
    (a.model / "int8_results.json").write_text(json.dumps({
        "float_test_accuracy": f_acc, "float_test_balanced_accuracy": f_bal,
        "int8_test_accuracy": i_acc, "int8_test_balanced_accuracy": i_bal,
        "float_int8_top1_agreement": agree, "test_images": int(len(yte)),
        "weights_int8": int(sum(q[f"w{l}"].size for l in range(len(layers))) + q["fcw"].size)}, indent=1))

    d = a.hls / "data"
    d.mkdir(parents=True, exist_ok=True)
    xte.astype(np.uint8).tofile(d / "test_images.bin")
    yte.astype(np.int32).tofile(d / "test_labels.bin")
    gi.astype(np.int32).tofile(d / "golden_logits.bin")
    np.savez(a.model / "int8_params.npz", **q)

    def arr(name, ctype, v, dims):
        flat = ", ".join(str(int(x)) for x in np.asarray(v).ravel())
        return f"static const {ctype} {name}{''.join(f'[{d}]' for d in dims)} = {{{flat}}};\n"

    h = ["// GENERATED by model/quantize.py - do not edit.\n#pragma once\n#include <stdint.h>\n\n",
         f"#define N_CLASSES {len(classes)}\n#define N_TEST_IMAGES {len(yte)}\n\n"]
    for l in range(len(layers)):
        co, ci = q[f"w{l}"].shape[:2]
        h += [arr(f"W{l}", "int8_t", q[f"w{l}"], [co, ci, 3, 3]),
              arr(f"B{l}", "int32_t", q[f"b{l}"], [co]),
              arr(f"M{l}", "int32_t", q[f"m{l}"], [co]),
              arr(f"S{l}", "uint8_t", q[f"s{l}"], [co]), "\n"]
    h += [arr("FCW", "int8_t", q["fcw"], list(q["fcw"].shape)), arr("FCB", "int32_t", q["fcb"], [len(classes)])]
    (a.hls / "weights.h").write_text("".join(h))
    print(f"wrote {a.hls/'weights.h'} and test vectors in {d}")


if __name__ == "__main__":
    main()
