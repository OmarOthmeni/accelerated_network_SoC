#!/usr/bin/env python3
"""Real-time classification server with a live web dashboard.

Runs on the PYNQ board (accelerator on the FPGA) or on any Linux machine with the same C++ kernel
compiled for the CPU (for testing the demo without the board).

    # on the board (needs overlay/cnn.bit + cnn.hwh built by vivado/build_overlay.tcl)
    sudo python3 board_server.py --backend pynq --bit ../pynq/overlay/cnn.bit
    # on a laptop, no board: bit-exact CPU version of the same kernel
    make -C ../hls lib && python3 board_server.py --backend cpu

Then open http://<board-ip>:8080 in a browser and stream images with stream_client.py.

Protocol (plain HTTP, no extra Python packages needed on the board):
    POST /classify   body = 4096 raw bytes (64x64 uint8 grayscale, row-major)
                     optional headers: X-Label (true class index), X-Name (file name)
                     reply: JSON {pred, pred_name, label, correct, hw_ms, top3}
    GET  /events     server-sent events stream used by the dashboard
    GET  /stats      JSON counters
    GET  /           dashboard
"""
import argparse
import base64
import ctypes
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

IMG_BYTES = 64 * 64
N_OUT = 24
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- backends
class PynqBackend:
    """cnn_top IP on the FPGA, one image per call (buffers allocated once)."""

    def __init__(self, bit):
        from pynq import Overlay, allocate
        self.ol = Overlay(bit)
        self.ip = self.ol.cnn_top
        self.img = allocate(shape=(IMG_BYTES,), dtype=np.uint8)
        self.out = allocate(shape=(N_OUT,), dtype=np.int32)
        rm = self.ip.register_map
        rm.images_1 = self.img.device_address   # names from the HLS driver (xcnn_top_hw.h)
        rm.logits_1 = self.out.device_address
        rm.n_images = 1
        self.name = "FPGA (PYNQ overlay)"

    def classify(self, img_u8):
        self.img[:] = img_u8
        self.img.sync_to_device()
        rm = self.ip.register_map
        rm.CTRL.AP_START = 1
        while rm.CTRL.AP_DONE == 0:
            pass
        self.out.sync_from_device()
        return np.array(self.out, dtype=np.int32)


class CpuBackend:
    """The same HLS C++ kernel compiled as a shared library (bit-exact with the FPGA)."""

    def __init__(self, lib):
        self.lib = ctypes.CDLL(str(lib))
        self.lib.cnn_classify.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.out = np.zeros(N_OUT, dtype=np.int32)
        self.name = "CPU (same C++ kernel, for testing)"

    def classify(self, img_u8):
        img = np.ascontiguousarray(img_u8, dtype=np.uint8)
        self.lib.cnn_classify(img.ctypes.data, self.out.ctypes.data, 1)
        return self.out.copy()


# ----------------------------------------------------------------------------- state
class Hub:
    """Counters and the list of dashboard subscribers (one queue per open browser tab)."""

    def __init__(self, classes, backend_name):
        self.classes = classes
        self.backend_name = backend_name
        self.lock = threading.Lock()
        self.subs = []
        self.reset()

    def reset(self):
        self.n = self.labelled = self.correct = 0
        self.hw_ms = []
        self.t_recent = []

    def stats(self):
        now = time.time()
        recent = [t for t in self.t_recent if now - t < 2.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        return {"images": self.n, "labelled": self.labelled, "correct": self.correct,
                "accuracy": self.correct / self.labelled if self.labelled else None,
                "fps": round(fps, 1),
                "hw_ms_median": round(float(np.median(self.hw_ms[-200:])), 3) if self.hw_ms else None,
                "backend": self.backend_name}

    def publish(self, event):
        msg = "data: " + json.dumps(event) + "\n\n"
        for q in list(self.subs):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # slow browser: drop frames rather than block the classifier


def make_handler(backend, hub, accel_lock):
    page = (HERE / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: one TCP connection per client

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/stats":
                with hub.lock:
                    self._send(200, json.dumps(hub.stats()).encode())
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                q = queue.Queue(maxsize=64)
                hub.subs.append(q)
                try:
                    self.wfile.write(("data: " + json.dumps({"type": "hello", "classes": hub.classes,
                                                             "backend": hub.backend_name}) + "\n\n").encode())
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15)
                        except queue.Empty:
                            msg = ": keep-alive\n\n"
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.subs.remove(q)
                    self.close_connection = True
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path == "/reset":
                with hub.lock:
                    hub.reset()
                self._send(200, b'{"ok":true}')
                return
            if self.path != "/classify":
                self._send(404, b'{"error":"not found"}')
                return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            if n != IMG_BYTES:
                self._send(400, json.dumps({"error": f"expected {IMG_BYTES} bytes, got {n}"}).encode())
                return
            img = np.frombuffer(body, dtype=np.uint8)
            label = self.headers.get("X-Label")
            label = int(label) if label not in (None, "") else None
            with accel_lock:  # one image at a time on the accelerator
                t0 = time.perf_counter()
                logits = backend.classify(img)
                hw_ms = (time.perf_counter() - t0) * 1e3
            pred = int(np.argmax(logits))
            top3 = [int(i) for i in np.argsort(logits)[::-1][:3]]
            res = {"pred": pred, "pred_name": hub.classes[pred], "label": label,
                   "correct": (pred == label) if label is not None else None,
                   "hw_ms": round(hw_ms, 3), "top3": top3,
                   "top3_names": [hub.classes[i] for i in top3]}
            with hub.lock:
                hub.n += 1
                hub.hw_ms.append(hw_ms)
                hub.t_recent.append(time.time())
                hub.t_recent = hub.t_recent[-200:]
                if label is not None:
                    hub.labelled += 1
                    hub.correct += int(pred == label)
                stats = hub.stats()
            hub.publish({"type": "result", **res, "name": self.headers.get("X-Name", ""),
                         "label_name": hub.classes[label] if label is not None else None,
                         "img": base64.b64encode(body).decode(), "stats": stats})
            self._send(200, json.dumps(res).encode())

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["pynq", "cpu"], default="pynq")
    ap.add_argument("--bit", default=str(HERE.parent / "pynq" / "overlay" / "cnn.bit"))
    ap.add_argument("--lib", default=str(HERE.parent / "hls" / "libcnn.so"))
    ap.add_argument("--classes", default=str(HERE.parent / "model" / "out" / "classes.json"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    classes = json.loads(Path(a.classes).read_text())
    assert len(classes) == N_OUT, f"expected {N_OUT} classes, got {len(classes)}"
    backend = PynqBackend(a.bit) if a.backend == "pynq" else CpuBackend(a.lib)
    hub = Hub(classes, backend.name)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(backend, hub, threading.Lock()))
    srv.daemon_threads = True
    print(f"backend: {backend.name}")
    print(f"dashboard: http://<this-machine-ip>:{a.port}/   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
