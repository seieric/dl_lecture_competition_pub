import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # 最終層を削除して512次元の特徴量を出力するようにする
        self.resnet.fc = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.resnet(X)
