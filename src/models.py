import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models
import math
import os

from .layers.subject import SubjectLayer
from .layers.resnet_gelu.resnet34 import ResNet34GELU


class MyModel(nn.Module):
    def __init__(self, pretrained=False, num_freezed_params=0) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 271, kernel_size=1, stride=1, padding=1)
        self.subject_layer = SubjectLayer(4, 271)
        self.classifier = ResNet34GELU(in_channels=271, num_classes=1854, dropout=0.3)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.conv1d(X)
        X = self.subject_layer(X, subject_idx)
        X = X.unsqueeze(2)
        return self.classifier(X)


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 64, kernel_size=3, stride=1, padding=1)
        self.resnet50 = models.resnet50()
        self.resnet50.conv1 = nn.Conv2d(
            64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
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
        self.resnet152.conv1 = nn.Conv2d(
            64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet152.fc = nn.Linear(self.resnet152.fc.in_features, 1854)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1d(X)
        X = X.unsqueeze(2)
        return self.resnet152(X)


class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 256, 256, dilation_k=1),
            ConvBlock(256, 256, 256, dilation_k=2),
            ConvBlock(256, 256, 512, dilation_k=3),
            ConvBlock(512, 512, 512, dilation_k=4),
            ConvBlock(512, 512, 512, dilation_k=5),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(512, num_classes),
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
        hid_dim,
        out_dim,
        dilation_k,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(
            in_dim,
            hid_dim,
            kernel_size,
            dilation=2 ** (2 * dilation_k) % 5,
            padding="same",
        )
        self.conv1 = nn.Conv1d(
            hid_dim,
            hid_dim,
            kernel_size,
            dilation=2 ** (2 * dilation_k + 1) % 5,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            hid_dim, out_dim * 2, kernel_size, dilation=2, padding="same"
        )

        self.batchnorm0 = nn.BatchNorm1d(num_features=hid_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hid_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim * 2)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.hid_dim:
            X = F.gelu(self.batchnorm0(self.conv0(X))) + X  # skip connection
        else:
            X = F.gelu(self.batchnorm0(self.conv0(X)))

        X = F.gelu(self.batchnorm1(self.conv1(X))) + X  # skip connection
        X = self.batchnorm2(self.conv2(X))
        X = F.glu(X, dim=-2)

        return self.dropout(X)
