# INT8 CNN Accelerator on Zynq-7020 (PYNQ)

An end-to-end hardware/software co-design flow: a CNN designed for an FPGA, trained in PyTorch,
quantized to INT8, implemented as a **Vitis HLS** accelerator for the **Zynq-7020** (PYNQ-Z1/Z2),
and verified **bit-exact** against an integer reference model on the full test set.

Task: classify 64×64 grayscale images from NASA's Curiosity rover (MSL) into 24 classes
(wheel, drill, ChemCam calibration target, horizon, …).

## Results

| | Accuracy | Balanced accuracy* | Status |
|---|---|---|---|
| Float32 model (BatchNorm folded) | 97.67% | 98.04% | measured (PyTorch, 1,028 test images) |
| INT8 integer model | **97.28%** | **98.00%** | measured (Python golden model) |
| HLS kernel, C simulation | **97.28%, 0 / 24,672 logit mismatches** | — | measured (g++ and Vitis HLS 2026.1 csim) |
| On the PYNQ board | — | — | *to be measured* (`pynq/run_on_board.py`) |

\*mean per-class recall; the dataset is very imbalanced (15 to 1,878 training images per class).
Float32 and INT8 agree on the predicted class for 99.51% of test images.

| Model / hardware figure | Value |
|---|---|
| Parameters (all INT8 weights, held in on-chip ROM) | 61,960 |
| Operations per image | 12.4 M multiply-accumulates |
| Parallelism | 4 / 8 / 8 / 4 output channels × 3×3 kernel per layer (36–72 MACs per cycle) |
| Default build (low power) | 1.39 M cycles per image → **≈ 36 images/s at 50 MHz**; accelerator logic ≈ 0.165 W (post-route estimate, excludes ARM/DDR/board) |
| High-throughput build | 223,200 cycles per image → ≈ 448 images/s at 100 MHz |
| Measured throughput and power on the board | *pending Vivado bitstream + PYNQ run* |

### Low-power build: post-route results (default, `PO = 1/1/1/1`, 50 MHz)

Vivado 2026.1 place-and-route on an Artix-7 xc7a100t-1 (same 7-series fabric as the Zynq-7020
programmable logic), out of context. Reports in `reports/lowpower_artix7_50MHz/`.

| | Value |
|---|---|
| Throughput | 1,386,331 cycles per image → **36 images/s at 50 MHz** (target: 30) |
| Timing | met at 20 ns; 11.95 ns achieved post-route (≈ 84 MHz possible) |
| LUT / FF | **7,251 / 5,055** (13.6% / 4.8% of a Zynq-7020) |
| DSP | **54** (25% of a Zynq-7020) |
| BRAM | 50.5 BRAM36 (36% of a Zynq-7020) |
| Power of the accelerator logic (vectorless estimate, 25 °C) | **0.165 W = 0.078 W dynamic + 0.087 W device static** (not the board: see note) |
| Dynamic power by part | BRAM 0.024 · signals 0.019 · logic 0.015 · clocks 0.011 · DSP 0.008 W |
| Energy per image | ≈ 2.2 mJ dynamic (≈ 4.6 mJ including static) |

The power figure is Vivado's vectorless estimate (default switching activity, confidence
"medium") for the accelerator alone, on an Artix-7: it excludes the ARM cores, DDR, AXI
interconnect and the rest of the board. A PYNQ-Z2 draws roughly 2–2.5 W from 5 V when idle
(verify on the board); the meaningful board figure is the measured increase while the
accelerator runs, and energy per image = that increase ÷ images/s.

Going from 3 to 4 cyclic banks for the 3×3 window buffers was the key change for this build:
with power-of-two map sizes, bank = index & 3 and address = index >> 2, instead of /3 and %3
address logic on 9 reads per cycle. In HLS estimates it cut DSP from 165 to 39 and LUT from
48k to 24k at the same latency.

### Synthesis results (Vitis HLS 2026.1, 100 MHz target)

| | v1 (one pipeline per input-channel loop) | **v2 (rebalanced)** | Zynq-7020 budget |
|---|---|---|---|
| Cycles per image | 456,331 | **223,200 (2.04× faster)** | — |
| Cycles per layer (L1 / L2 / L3 / L4) | 4k / 239k / … (layer 2 = 53%) | **16k / 66k / 66k / 66k** | — |
| Estimated clock period | 7.30 ns | 7.30 ns (≈ 137 MHz possible) | 10 ns |
| DSP | 426 | **192** | 220 |
| BRAM18 | 222 | **124** | 280 |
| LUT | 42,262 | **51,434** | 53,200 |
| FF | 19,242 | 46,896 | 106,400 |

v1 was synthesized for the Kria K26 (xck26) because it does not fit the Zynq-7020. v2 resource
figures come from an Artix-7 xc7a100t, which has the same 7-series fabric as the Zynq-7020
programmable logic (the xc7z020 device files were not installed); HLS LUT estimates are usually
pessimistic compared with post-route results, which the Vivado run will confirm.

## Pipeline

```
model/train.py      PyTorch training (class-weighted loss, flip + brightness augmentation)
model/quantize.py   BN folding, INT8 post-training quantization, integer golden model,
                    exports hls/weights.h and test vectors
hls/cnn.cpp         HLS accelerator (also plain C++), hls/tb.cpp bit-exactness testbench
hls/run_hls.tcl     Vitis HLS: csim, synthesis, IP export (xc7z020clg400-1, 100 MHz)
vivado/build_overlay.tcl  Zynq PS + accelerator block design, bitstream, reports
pynq/run_on_board.py      runs on the board: bit-exactness, accuracy, throughput, latency
demo/                     live demo: stream images from a laptop, classify on the board, web dashboard
```

### Network (designed for the hardware)

```
input 1×64×64 uint8
4 × [ conv 3×3 (pad 1) + BatchNorm + ReLU + maxpool 2×2 ]   → 16, 32, 64, 64 channels
global average pool (4×4) → fully connected → 24 logits
```
Only operations that map cleanly onto fixed-point hardware: no strided convolutions,
no softmax on chip (argmax of int32 logits), BatchNorm folded into the weights.

### Integer arithmetic (identical in Python and in the HLS kernel)

- Weights: INT8 symmetric, per output channel (conv) / per tensor (FC). Activations: UINT8, zero point 0.
- Convolution: INT32 accumulation, `acc = bias + Σ w·a`.
- Requantization + ReLU: `a = clamp((acc · M + 2^(s−1)) >> s, 0, 255)` with a 31-bit integer
  multiplier `M` and shift `s` per channel (no floating point on the FPGA).
- 2×2 max pooling is fused with the convolution, so full-resolution feature maps are never stored.
- Average pooling = sum over the 4×4 map; the 1/16 is folded into the FC scale.

### Accelerator architecture

- All weights in on-chip ROMs, reshaped so the 9 taps of each (output, input) channel pair are
  one 72-bit word; feature maps in BRAM (largest: 16×32×32 bytes).
- Each conv layer is one flat II = 1 pipeline over (pooled pixel, 2×2 sub-pixel, input channel),
  so the pipeline never drains between pixels. Each cycle reads a 3×3 window (buffers cyclically
  partitioned by 3 on rows and columns) and updates PO output-channel accumulators.
- PO is chosen per layer (4, 8, 8, 4) so the four layers take about the same time.
- Requantization (a 32×32-bit multiply, ~4 DSP) is time-shared: one unit per layer requantizes
  the held accumulators of the previous sub-pixel, one channel per cycle, while the next
  sub-pixel accumulates. Layer 1 (one input channel) requantizes its 4 channels in parallel.
- Profiling v1 per layer showed layer 2 took 53% of the latency while layer 1 used 268 DSPs:
  the rebalance halves the latency and cuts DSPs from 426 to 192 so the design fits the Zynq-7020.
- AXI4 master ports for images (HP0) and logits (HP1); AXI4-Lite control (start, buffer
  addresses, number of images) so one call can process a whole batch.

## Reproduce

```bash
pip install -r requirements.txt
python model/train.py    --data ../data --out model/out
python model/quantize.py --data ../data
make -C hls csim                                   # bit-exactness on the whole test set (g++)
```
Run these from `accelerator/`. The data directory (`../data`, at the repository root) contains `train/`, `validation/` and `test/` with one sub-folder per class
(64×64 grayscale). It is built from the MSL Curiosity labeled image dataset (Lu & Wagstaff,
NASA JPL; v2.1 on Zenodo) using its official train/validation/test split.

Hardware (needs Vitis HLS + Vivado, and the PYNQ board files):
```bash
cd hls && vitis-run --mode hls --tcl run_hls.tcl && cd ..   # csim, synthesis, IP export (Vitis 2024.2+;
                                                          # older releases: vitis_hls -f run_hls.tcl)
vivado -mode batch -source vivado/build_overlay.tcl   # bitstream + utilization/timing/power reports
# copy pynq/ and hls/data/ to the board, then on the board:
sudo python3 pynq/run_on_board.py --data hls/data
```
The ARM Cortex-A9 software baseline is the same C++ code compiled on the board:
`make -C hls csim CXXFLAGS="-O3 -mcpu=cortex-a9 -mfpu=neon"`.

## Next steps

- Build the bitstream and measure on the PYNQ board: post-route resources, Fmax, images/s, W.
- Overlap the layers with `DATAFLOW` across images (each layer is now ~66k cycles, so a layer
  pipeline would approach 1 image per 66k cycles ≈ 1,500 images/s at 100 MHz).
- Reduce LUTs in the window muxes (line buffers instead of 3×3 cyclic banks).
- Quantization-aware training and INT4 weights.

*Omar Othmeni*
