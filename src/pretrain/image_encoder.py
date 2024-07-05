import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super(ImageEncoder, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.vit.parameters():
            param.requires_grad = False
        # 最終層を削除して768次元の特徴量を出力するようにする
        self.vit.heads = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.vit(X)
