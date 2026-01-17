from vision_models.factory import register
from vision_models.models.classifier.config import LinearClassifierConfig
from vision_models.models.classifier.linear import LinearClassifier
from vision_models.models.encoder.resnet.config import ResNetConfig
from vision_models.models.encoder.resnet.encoder import ResNetEncoder

register("resnet", ResNetEncoder, ResNetConfig)
register("linear_classifier", LinearClassifier, LinearClassifierConfig)
