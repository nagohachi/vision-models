import torch
import torch.nn as nn

from vision_models.models.encoder.resnet.config import ResNetBlockConfig
from vision_models.models.encoder.resnet.conv import PostNormConv, PreNormConv


class ResNetBlockBase(nn.Module):
    def _build_shortcut(self, config: ResNetBlockConfig) -> nn.Module:
        return (
            nn.Identity()
            if config.in_channels == config.out_channels
            else nn.Conv2d(
                config.in_channels,
                config.out_channels,
                kernel_size=1,
                stride=config.stride,
            )
        )

    def _build_model(
        self,
        config: ResNetBlockConfig,
        norm_conv_class: type[PreNormConv] | type[PostNormConv],
    ) -> nn.Module:
        block_type = "basic" if (1, 1) not in config.kernel_sizes else "bottleneck"

        in_channels = config.in_channels
        out_channels = config.out_channels

        if block_type == "basic":
            mid_channels = config.out_channels
            return nn.Sequential(
                norm_conv_class(
                    in_channels,
                    mid_channels,
                    config.kernel_sizes[0],
                    config.stride,
                ),
                norm_conv_class(mid_channels, out_channels, config.kernel_sizes[1], 1),
            )
        else:
            mid_channels = config.out_channels // 4
            return nn.Sequential(
                norm_conv_class(
                    in_channels,
                    mid_channels,
                    config.kernel_sizes[0],
                    1,
                ),
                norm_conv_class(
                    mid_channels,
                    mid_channels,
                    config.kernel_sizes[1],
                    config.stride,
                ),
                norm_conv_class(mid_channels, out_channels, config.kernel_sizes[2], 1),
            )


class ResNetBlock(ResNetBlockBase):
    def __init__(self, config: ResNetBlockConfig) -> None:
        super().__init__()
        self.shortcut = self._build_shortcut(config)
        self.model = self._build_model(config, norm_conv_class=PostNormConv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shortcut(x) + self.model(x))


class ResNetBlockV2(ResNetBlockBase):
    def __init__(self, config: ResNetBlockConfig) -> None:
        super().__init__()
        self.shortcut = self._build_shortcut(config)
        self.model = self._build_model(config, norm_conv_class=PreNormConv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.model(x)
