import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models
import math


class ResNet34(nn.Module):
    def __init__(self, pretrained=False, num_freezed_params=0) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 271, kernel_size=1, stride=1, padding=1)
        self.subject_layer = SubjectLayer(4, 271)
        if pretrained:
            self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            for i, param in enumerate(self.resnet34.parameters()):
                if i >= num_freezed_params:
                    param.requires_grad = True
        else:
            self.resnet34 = models.resnet34()
        self.resnet34.conv1 = nn.Conv2d(271, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, 1024)
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1854)
        )

    def forward(self, X_and_subject_idx: (torch.Tensor, torch.Tensor)) -> torch.Tensor:
        X, subject_idx = X_and_subject_idx
        X = self.conv1d(X)
        X = self.subject_layer(X, subject_idx)
        X = X.unsqueeze(2)
        X = self.resnet34(X)
        return self.final(X)
    
class SubjectLayer(nn.Module):
    def __init__(self, num_subjects, num_channels):
        super(SubjectLayer, self).__init__()
        self.num_subjects = num_subjects
        self.num_channels = num_channels

        self.Ms = nn.Parameter(torch.Tensor(num_subjects, num_channels, num_channels))
        nn.init.kaiming_uniform_(self.Ms, a=math.sqrt(5))

    def forward(self, X, subject_idx):        
        # subject_idx: (128)
        idx = subject_idx.view(-1, 1, 1).expand(X.size(0), self.num_channels, self.num_channels)
        # idx: (128, 271, 271)
        Ms_selected = torch.gather(self.Ms, 0, idx)
        # Ms_selected: (128, 271, 271)
        X = torch.matmul(Ms_selected, X)
        return X.squeeze(-1)

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