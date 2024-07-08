import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, pretrained_weights) -> None:
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet34()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1854)
        self.resnet.load_state_dict(pretrained_weights)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 最終層を削除して512次元の特徴量を出力するようにする
        self.resnet.fc = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.resnet(X)
