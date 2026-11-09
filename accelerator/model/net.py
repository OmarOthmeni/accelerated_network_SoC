"""Hardware-friendly CNN: every layer maps directly onto the HLS kernel.

input 1x64x64 (uint8 pixels)
  4 x [ conv3x3 (pad 1) + BatchNorm + ReLU + maxpool 2x2 ]   16, 32, 64, 64 channels
  global average pool (4x4) -> fully connected -> 24 classes

BatchNorm is folded into the convolution weights before quantization, so the
deployed network is only conv / ReLU / maxpool / average pool / FC.
"""
import torch
import torch.nn as nn

CHANNELS = [1, 16, 32, 64, 64]


class HwCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        blocks = []
        for cin, cout in zip(CHANNELS[:-1], CHANNELS[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.fc = nn.Linear(CHANNELS[-1], n_classes)

    def forward(self, x_pixels):
        x = x_pixels / 255.0                      # uint8 pixels in [0, 255]
        x = self.features(x)                      # -> N x 64 x 4 x 4
        x = x.mean(dim=(2, 3))                    # global average pool
        return self.fc(x)
