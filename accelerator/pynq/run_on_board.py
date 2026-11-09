#!/usr/bin/env python3
"""Run the CNN overlay on a PYNQ-Z1/Z2 and measure accuracy and throughput.

Copy to the board: pynq/ (with overlay/cnn.bit + overlay/cnn.hwh) and hls/data/.
    sudo python3 run_on_board.py --data ../hls/data

Reports: bit-exactness vs the golden model, top-1 accuracy, images/s for batch sizes 1 and full.
Power: the PYNQ-Z1/Z2 have no on-board power sensor; measure board power with a USB power
meter (idle vs running this script with --loop) and subtract.
"""
import argparse
import time

import numpy as np
from pynq import Overlay, allocate

IMG = 64 * 64
N_OUT = 24


def run(ip, img_buf, out_buf, n):
    ip.register_map.images_1 = img_buf.device_address   # if names differ, see
    ip.register_map.logits_1 = out_buf.device_address   # hls/cnn_prj/sol1/impl/misc/drivers/*/xcnn_top_hw.h
    ip.register_map.n_images = n
    ip.register_map.CTRL.AP_START = 1
    while ip.register_map.CTRL.AP_DONE == 0:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../hls/data")
    ap.add_argument("--bit", default="overlay/cnn.bit")
    ap.add_argument("--loop", type=float, default=0, help="keep running for N seconds (power measurement)")
    a = ap.parse_args()

    images = np.fromfile(f"{a.data}/test_images.bin", dtype=np.uint8)
    labels = np.fromfile(f"{a.data}/test_labels.bin", dtype=np.int32)
    golden = np.fromfile(f"{a.data}/golden_logits.bin", dtype=np.int32).reshape(-1, N_OUT)
    n = len(labels)

    ol = Overlay(a.bit)
    ip = ol.cnn_top
    img_buf = allocate(shape=(n * IMG,), dtype=np.uint8)
    out_buf = allocate(shape=(n * N_OUT,), dtype=np.int32)
    img_buf[:] = images
    img_buf.sync_to_device()

    # full test set in one call
    t0 = time.perf_counter()
    run(ip, img_buf, out_buf, n)
    dt = time.perf_counter() - t0
    out_buf.sync_from_device()
    logits = np.array(out_buf).reshape(n, N_OUT)

    exact = np.array_equal(logits, golden)
    acc = float((logits.argmax(1) == labels).mean())
    print(f"test images      : {n}")
    print(f"bit-exact        : {exact}")
    print(f"top-1 accuracy   : {acc:.4f}")
    print(f"batch throughput : {n/dt:.1f} images/s ({dt/n*1e3:.3f} ms/image)")

    # single-image latency (includes driver overhead)
    lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        run(ip, img_buf, out_buf, 1)
        lat.append(time.perf_counter() - t0)
    print(f"single-image latency (median of 50): {np.median(lat)*1e3:.3f} ms")

    if a.loop > 0:
        print(f"running continuously for {a.loop:.0f} s: read the USB power meter now")
        t_end = time.time() + a.loop
        count = 0
        while time.time() < t_end:
            run(ip, img_buf, out_buf, n)
            count += n
        print(f"sustained: {count/a.loop:.1f} images/s")


if __name__ == "__main__":
    main()
