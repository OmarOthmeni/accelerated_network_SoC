#!/usr/bin/env python3
"""Train the hardware-friendly CNN on the Curiosity (MSL) image dataset.

Dataset layout expected (64x64 grayscale images, one folder per class):
    <data>/train/<class>/*.jpg   <data>/validation/<class>/*.jpg   <data>/test/<class>/*.jpg

Usage:  python model/train.py --data path/to/data --out model/out
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from net import HwCNN

torch.manual_seed(0)
np.random.seed(0)


def load_split(root: Path, classes):
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).glob("*")):
            img = Image.open(f).convert("L").resize((64, 64))
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(ci)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def to_tensor(x_u8):
    # the accelerator receives raw uint8 pixels; the network normalizes internally
    return torch.from_numpy(x_u8).float().unsqueeze(1)


def evaluate(model, x, y, n_classes):
    model.eval()
    with torch.no_grad():
        pred = model(to_tensor(x)).argmax(1).numpy()
    acc = float((pred == y).mean())
    per_class = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return acc, float(np.mean(per_class))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("model/out"), type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    classes = sorted(p.name for p in (a.data / "train").iterdir() if p.is_dir())
    xtr, ytr = load_split(a.data / "train", classes)
    xva, yva = load_split(a.data / "validation", classes)
    xte, yte = load_split(a.data / "test", classes)
    print(f"{len(classes)} classes | train {len(ytr)} | val {len(yva)} | test {len(yte)}")

    model = HwCNN(len(classes))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # class-balanced loss: the dataset ranges from 15 to 1878 images per class
    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float64)
    weights = torch.tensor((counts.sum() / (len(classes) * counts)) ** 0.5, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=a.epochs,
                                                steps_per_epoch=(len(ytr) + 63) // 64)

    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(len(ytr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = to_tensor(xtr[idx])
            # light augmentation: horizontal flip + brightness jitter (keeps uint8 range)
            flip = torch.rand(len(idx)) < 0.5
            xb[flip] = xb[flip].flip(-1)
            xb = (xb * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1))).clamp(0, 255).round()
            loss = loss_fn(model(xb), torch.from_numpy(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        va_acc, va_bal = evaluate(model, xva, yva, len(classes))
        if va_bal > best:
            best, best_state = va_bal, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep+1:2d} | loss {loss.item():.3f} | val acc {va_acc:.4f} "
              f"| val balanced acc {va_bal:.4f} | {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    te_acc, te_bal = evaluate(model, xte, yte, len(classes))
    print(f"TEST (float32): accuracy {te_acc:.4f} | balanced accuracy {te_bal:.4f}")

    torch.save(model.state_dict(), a.out / "hwcnn_fp32.pt")
    (a.out / "classes.json").write_text(json.dumps(classes, indent=1))
    (a.out / "fp32_results.json").write_text(json.dumps(
        {"test_accuracy": te_acc, "test_balanced_accuracy": te_bal,
         "val_balanced_accuracy": best}, indent=1))


if __name__ == "__main__":
    main()
