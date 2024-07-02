import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super(ImageEncoder, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 最終層を削除して2048次元の特徴量を出力するようにする
        self.resnet50.fc = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.resnet50(X)
