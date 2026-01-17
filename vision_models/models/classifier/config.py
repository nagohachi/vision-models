from dataclasses import dataclass


@dataclass
class LinearClassifierConfig:
    in_features: int
    num_classes: int
