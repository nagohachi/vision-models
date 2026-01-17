import torch
import torch.nn as nn


class ResNetStem(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self._out_channels = 64
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self._out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,  # since BatchNorm is applied
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
