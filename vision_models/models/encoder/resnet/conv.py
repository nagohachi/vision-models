import torch
import torch.nn as nn


class PostNormConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int,
    ) -> None:
        super().__init__()
        padding = 1 if kernel_size == (3, 3) else 0
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PreNormConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int,
    ) -> None:
        super().__init__()
        padding = 1 if kernel_size == (3, 3) else 0
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
