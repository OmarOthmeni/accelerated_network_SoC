#!/usr/bin/env python3
"""Stream images from the laptop to the board and print the predictions.

    pip install numpy pillow                     # (opencv-python only for --webcam)
    python3 stream_client.py --board 192.168.2.99 --fps 30                # test set, shuffled
    python3 stream_client.py --board 192.168.2.99 --folder my_images/     # any JPG/PNG
    python3 stream_client.py --board 192.168.2.99 --webcam                # live camera
    python3 stream_client.py --board localhost                            # CPU backend test

Images are converted exactly like the training data: grayscale, resized to 64x64.
With the test set, the true class is sent too, so the dashboard shows the running accuracy.
Watch the results live at http://<board>:8080 in a browser.
"""
import argparse
import http.client
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def to_u8(img: Image.Image) -> bytes:
    return np.asarray(img.convert("L").resize((64, 64)), dtype=np.uint8).tobytes()


def test_set(classes, root):
    items = []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).glob("*")):
            items.append((f, ci))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="192.168.2.99", help="board IP (PYNQ default 192.168.2.99)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30, help="target rate; 0 = as fast as possible")
    ap.add_argument("--folder", help="classify every image in this folder (no labels)")
    ap.add_argument("--data", default=str(REPO / "data" / "test"), help="labelled test set (one folder per class)")
    ap.add_argument("--classes", default=str(HERE.parent / "model" / "out" / "classes.json"))
    ap.add_argument("--webcam", action="store_true", help="stream the laptop camera (needs opencv-python)")
    ap.add_argument("--count", type=int, default=0, help="stop after N images (0 = all / forever)")
    ap.add_argument("--loop", action="store_true", help="repeat the image list forever")
    a = ap.parse_args()
    classes = json.loads(Path(a.classes).read_text())

    conn = http.client.HTTPConnection(a.board, a.port, timeout=10)
    conn.request("POST", "/reset")
    conn.getresponse().read()

    def send(raw, label=None, name=""):
        headers = {"Content-Type": "application/octet-stream", "X-Name": name}
        if label is not None:
            headers["X-Label"] = str(label)
        conn.request("POST", "/classify", body=raw, headers=headers)
        return json.loads(conn.getresponse().read())

    if a.webcam:
        import cv2
        cap = cv2.VideoCapture(0)
        frames = iter(lambda: cap.read(), None)

        def items():
            for ok, frame in frames:
                if not ok:
                    break
                h, w = frame.shape[:2]
                s = min(h, w)  # centre crop to a square, like the dataset images
                crop = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
                yield Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), None, "webcam"
    else:
        if a.folder:
            files = [(f, None) for f in sorted(Path(a.folder).rglob("*"))
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        else:
            files = test_set(classes, Path(a.data))
            random.seed(0)
            random.shuffle(files)
        print(f"{len(files)} images")

        def items():
            while True:
                for f, label in files:
                    yield Image.open(f), label, f.name
                if not a.loop:
                    break

    period = 1.0 / a.fps if a.fps > 0 else 0.0
    n = correct = labelled = 0
    t_start = t_next = time.perf_counter()
    try:
        for img, label, name in items():
            r = send(to_u8(img), label, name)
            n += 1
            if label is not None:
                labelled += 1
                correct += int(r["correct"])
            mark = "" if label is None else (" ok " if r["correct"] else f" X (true: {classes[label]})")
            rate = n / (time.perf_counter() - t_start)
            acc = f" | acc {100 * correct / labelled:.1f}%" if labelled else ""
            print(f"{n:5d}  {r['pred_name']:<22}{mark:<32} accel {r['hw_ms']:.2f} ms | {rate:.1f} img/s{acc}")
            if a.count and n >= a.count:
                break
            if period:
                t_next += period
                time.sleep(max(0.0, t_next - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    print(f"\n{n} images, {n / (time.perf_counter() - t_start):.1f} img/s end-to-end"
          + (f", accuracy {100 * correct / labelled:.2f}% on {labelled} labelled" if labelled else ""))


if __name__ == "__main__":
    main()
