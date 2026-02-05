import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class AgeMobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.model = mobilenet_v3_large(weights="IMAGENET1K_V1")
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
