import torch
import torch.nn as nn

from vision_models.models.classifier.config import LinearClassifierConfig


class LinearClassifier(nn.Module):
    def __init__(self, config: LinearClassifierConfig) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config.in_features, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
