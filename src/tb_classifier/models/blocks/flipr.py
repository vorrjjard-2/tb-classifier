import torch
from torch import nn


class FlipRBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_asym = torch.flip(x, dims=[-1])
        asym = self.blur(x - x_asym)
        gate_raw = self.act(self.conv(asym))
        x = x * (1 + gate_raw)

        return x
