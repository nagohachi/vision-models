from dataclasses import dataclass
from typing import Literal


@dataclass
class ResNetBlockConfig:
    in_channels: int
    out_channels: int
    kernel_sizes: list[tuple[int, int]]
    stride: int


@dataclass
class ResNetStageConfig:
    block_configs: list[ResNetBlockConfig]


@dataclass
class ResNetConfig:
    version: Literal["v1", "v2"]
    stages: list[ResNetStageConfig]
