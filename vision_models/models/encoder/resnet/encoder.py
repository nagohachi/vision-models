import torch
import torch.nn as nn

from vision_models.models.encoder.resnet.block import ResNetBlock, ResNetBlockV2
from vision_models.models.encoder.resnet.config import ResNetConfig
from vision_models.models.encoder.resnet.stem import ResNetStem


class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.stem = ResNetStem(config.in_channels)
        block_class = ResNetBlock if config.version == "v1" else ResNetBlockV2
        self.stages = nn.Sequential(
            *[
                # each stage
                nn.Sequential(
                    *[
                        # each block
                        block_class(block_config)
                        for block_config in stage_config.block_configs
                    ]
                )
                for stage_config in config.stages
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))
