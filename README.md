# Accelerated Network SoC — Mars Rover Image Classification on an FPGA

Classify 64×64 grayscale images from NASA's Curiosity rover (MSL) into 24 classes (wheel, drill,
ChemCam calibration target, horizon, …), then run the network on a **Zynq-7020 FPGA** as an
INT8 hardware accelerator.

The repository has two parts:

| Part | Folder | What it is | Test accuracy |
|---|---|---|---|
| **Software baseline** | `scripts/`, `models/`, `results/` | Keras CNN, 2 conv + 2 dense layers, 407,832 parameters (Donia Mabrouk) | 90.47% |
| **Hardware accelerator** | [`accelerator/`](accelerator/) | Hardware-friendly CNN, INT8 quantization, Vitis HLS accelerator for PYNQ-Z1/Z2 (Omar Othmeni) | **97.28% (INT8)**, bit-exact in C simulation |

## Hardware accelerator (summary)

- **Network designed for the FPGA**: 4 × [conv 3×3 + ReLU + maxpool] (16/32/64/64 channels),
  global average pooling, fully connected. 61,960 parameters, 12.4 M MACs per image.
- **INT8 post-training quantization** with integer-only requantization (multiplier + shift per
  channel); float32 97.67% → INT8 97.28% (98.00% balanced accuracy).
- **HLS accelerator**: 72 MACs per cycle, fused conv/ReLU/maxpool, all weights in on-chip ROM,
  AXI4 masters for images and logits, AXI4-Lite control.
- **Verified bit-exact**: the HLS C++ kernel reproduces the Python integer model on all 1,028 test
  images (0 of 24,672 logits differ).
- **Next**: Vitis HLS synthesis, Vivado bitstream and PYNQ board measurements (throughput,
  latency, power, resources). Scripts are in `accelerator/hls`, `accelerator/vivado`, `accelerator/pynq`.

Details, arithmetic and reproduction steps: [`accelerator/README.md`](accelerator/README.md).

## Software baseline (Donia Mabrouk)

- 2 conv + 2 dense layers (407,832 parameters, 1.56 MB), Keras.
- **Test accuracy 90.47%**; best class apxs cal target (100%), worst chemcam cal target (33%).
- `scripts/train_baseline.py`, `scripts/evaluate_baseline.py`, model in `models/baseline/minimal_cnn.h5`,
  evaluation in `results/baseline/`.

```bash
pip install -r requirements.txt
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
```

## Dataset

- 24 classes, 64×64 grayscale; train 4,673 / validation 990 / test 1,028 images.
- Highly imbalanced: 15 to 1,878 training images per class, so the accelerator also reports
  balanced accuracy (mean per-class recall).
- Built from the MSL Curiosity labeled image dataset (Lu & Wagstaff, NASA JPL) with its official
  splits; `data/data_prepare.ipynb` shows the preparation.

## Repository layout

```
data/          train / validation / test images (one folder per class)
scripts/       Keras baseline training and evaluation
models/        Keras baseline model
results/       Keras baseline evaluation
accelerator/   model/ (PyTorch + INT8 quantization), hls/ (accelerator + testbench),
               vivado/ (Zynq overlay), pynq/ (on-board measurement)
```
