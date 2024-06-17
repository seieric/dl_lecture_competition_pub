import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models


class ResNet34(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 64, kernel_size=3, stride=1, padding=1)
        if pretrained:
            self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.resnet34 = models.resnet34()
        self.resnet34.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, 1854)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1d(X)
        X = X.unsqueeze(2)
        return self.resnet34(X)

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 64, kernel_size=3, stride=1, padding=1)
        self.resnet50 = models.resnet50()
        self.resnet50.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 1854)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1d(X)
        X = X.unsqueeze(2)
        return self.resnet50(X)

class ResNet152(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 64, kernel_size=3, stride=1, padding=1)
        self.resnet152 = models.resnet152()
        self.resnet152.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet152.fc = nn.Linear(self.resnet152.fc.in_features, 1854)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1d(X)
        X = X.unsqueeze(2)        
        return self.resnet152(X)

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)