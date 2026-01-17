from pathlib import Path

import yaml
from dacite import Config, from_dict

from vision_models.models.encoder.resnet.config import ResNetConfig
from vision_models.models.encoder.resnet.encoder import ResNetEncoder

CONFIGS_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "resnet"


def load_config(config_path: str | Path) -> ResNetConfig:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return from_dict(
        data_class=ResNetConfig,
        data=data,
        config=Config(cast=[tuple]),
    )


def create_resnet_encoder(config_path: str | Path) -> ResNetEncoder:
    config = load_config(config_path)
    return ResNetEncoder(config)
