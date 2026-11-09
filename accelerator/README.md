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
| HLS kernel, C simulation | **97.28%, 0 / 24,672 logit mismatches** | — | measured (g++) |
| On the PYNQ board | — | — | *to be measured* (`pynq/run_on_board.py`) |

\*mean per-class recall; the dataset is very imbalanced (15 to 1,878 training images per class).
Float32 and INT8 agree on the predicted class for 99.51% of test images.

| Model / hardware figure | Value |
|---|---|
| Parameters (all INT8 weights, held in on-chip ROM) | 61,960 |
| Operations per image | 12.4 M multiply-accumulates |
| Parallelism | 8 output channels × 3×3 kernel = 72 MACs per cycle |
| Ideal compute time (II = 1, 100 MHz, no loop overhead) | 172k cycles ≈ 1.7 ms → upper bound ≈ 580 images/s |
| FPGA resources, Fmax, measured throughput and power | *pending Vivado run on hardware* |

## Pipeline

```
model/train.py      PyTorch training (class-weighted loss, flip + brightness augmentation)
model/quantize.py   BN folding, INT8 post-training quantization, integer golden model,
                    exports hls/weights.h and test vectors
hls/cnn.cpp         HLS accelerator (also plain C++), hls/tb.cpp bit-exactness testbench
hls/run_hls.tcl     Vitis HLS: csim, synthesis, IP export (xc7z020clg400-1, 100 MHz)
vivado/build_overlay.tcl  Zynq PS + accelerator block design, bitstream, reports
pynq/run_on_board.py      runs on the board: bit-exactness, accuracy, throughput, latency
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

- All weights in on-chip ROMs; feature maps in BRAM (largest: 16×32×32 bytes).
- Inner loop pipelined at II = 1 over input channels; each cycle reads a 3×3 window (buffers
  cyclically partitioned by 3 on rows and columns) and updates 8 output-channel accumulators.
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
cd hls && vitis_hls -f run_hls.tcl && cd ..        # csim, synthesis report, IP export
vivado -mode batch -source vivado/build_overlay.tcl   # bitstream + utilization/timing/power reports
# copy pynq/ and hls/data/ to the board, then on the board:
sudo python3 pynq/run_on_board.py --data hls/data
```
The ARM Cortex-A9 software baseline is the same C++ code compiled on the board:
`make -C hls csim CXXFLAGS="-O3 -mcpu=cortex-a9 -mfpu=neon"`.

## Next steps

- Run synthesis and board measurements, and fill in resources, Fmax, images/s and W.
- Remove the per-pixel loop overhead in layer 1 (only one input channel) and overlap the
  layers with `DATAFLOW` across images.
- Quantization-aware training and INT4 weights.

*Omar Othmeni*
