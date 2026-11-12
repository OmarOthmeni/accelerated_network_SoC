# Live demo: stream images from a laptop, classify on the PYNQ board

```
laptop                                   PYNQ-Z1/Z2 board
stream_client.py  ── HTTP (Ethernet) ──▶ board_server.py ──▶ cnn_top on the FPGA
browser  ◀──────── live dashboard (:8080) ─────────┘
```

The laptop sends 64×64 grayscale images (test set, any folder, or the webcam). The board
classifies each one on the accelerator and serves a dashboard showing the image, the predicted
class, ✓/✗ against the true class, the running accuracy, images/s and accelerator latency.
The board needs nothing beyond the stock PYNQ image (numpy + `pynq`): no extra packages.

## 1. Try it without the board (any Linux laptop)
The same C++ kernel, compiled for the CPU, is bit-exact with the FPGA:
```bash
pip install numpy pillow
make -C ../hls lib                                   # builds hls/libcnn.so
python3 board_server.py --backend cpu                # terminal 1
python3 stream_client.py --board localhost --fps 10  # terminal 2
# open http://localhost:8080
```
Checked on the full test set: 1,028 images, 97.28% accuracy (same as the integer model).

## 2. On the board
1. **Build the overlay** (on the laptop with Vivado, Zynq-7000 devices installed):
   `cd hls && vitis-run --mode hls --tcl run_hls.tcl`, then
   `vivado -mode batch -source vivado/build_overlay.tcl`. Copy the resulting `.bit` and `.hwh`
   (same base name) to `pynq/overlay/cnn.bit` and `pynq/overlay/cnn.hwh`.
2. **Network:** connect the board to the laptop with an Ethernet cable. The PYNQ image's
   default static address is 192.168.2.99. Give the laptop's wired interface a static
   192.168.2.1/24 (or plug both into the same router and use the board's DHCP address).
3. **Copy the code:** `scp -r accelerator xilinx@192.168.2.99:~/` (default login: see the
   PYNQ documentation), or `git clone` this repository on the board.
4. **Start the server on the board:**
   ```bash
   cd ~/accelerator/demo
   sudo python3 board_server.py --backend pynq --bit ../pynq/overlay/cnn.bit
   ```
   If the register names differ (`images_1`, `logits_1`), use the names from
   `hls/cnn_prj/sol1/impl/misc/drivers/*/src/xcnn_top_hw.h`.
5. **On the laptop:** open http://192.168.2.99:8080, then stream:
   ```bash
   python3 stream_client.py --board 192.168.2.99 --fps 30            # test set with labels
   python3 stream_client.py --board 192.168.2.99 --folder photos/    # your own images
   python3 stream_client.py --board 192.168.2.99 --webcam            # needs opencv-python
   ```

## What to expect
- **Low-power build** (default, 50 MHz): ~28 ms of accelerator time per image, so up to
  ~36 images/s. 30 images/s from the laptop is within budget, but Python and HTTP on the ARM
  add a few ms per image: measure the end-to-end rate with `--fps 0`.
- **High-throughput build** (`-DPO1=4 -DPO2=8 -DPO3=8 -DPO4=4`, 100 MHz): ~2.2 ms per image.
  The end-to-end rate is then limited by the network and Python, not the FPGA.
- The dashboard shows the accelerator time measured by the server (call to done), not the
  network round trip; `stream_client.py` prints the end-to-end images/s.
- Power: run the client with `--loop` and read a USB power meter on the board's 5 V input.
  Compare with the idle board to get the accelerator's share.
