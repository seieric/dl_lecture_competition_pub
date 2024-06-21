import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models
import math
import os


class ResNet34(nn.Module):
    def __init__(self, pretrained=False, num_freezed_params=0) -> None:
        super().__init__()
        self.spatial_attention = SpatialAttentionLayer(271)
        self.conv1d = nn.Conv1d(271, 271, kernel_size=1, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.subject_layer = SubjectLayer(4, 271)
        if pretrained:
            self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            for i, param in enumerate(self.resnet34.parameters()):
                if i >= num_freezed_params:
                    param.requires_grad = True
        else:
            self.resnet34 = models.resnet34()
        self.resnet34.conv1 = nn.Conv2d(271, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, 1854)

    def forward(self, X_and_subject_idx: (torch.Tensor, torch.Tensor)) -> torch.Tensor:
        X, subject_idx = X_and_subject_idx
        X = self.spatial_attention(X)
        X = self.conv1d(X)
        X = self.dropout1(X)
        X = self.subject_layer(X, subject_idx)
        X = X.unsqueeze(2)
        return self.resnet34(X)
    
class SubjectLayer(nn.Module):
    def __init__(self, num_subjects, num_channels):
        super(SubjectLayer, self).__init__()
        self.num_subjects = num_subjects
        self.num_channels = num_channels

        self.Ms = nn.Parameter(torch.Tensor(num_subjects, num_channels, num_channels))
        nn.init.uniform_(self.Ms, -math.sqrt(3), math.sqrt(3))

    def forward(self, X, subject_idx):        
        # subject_idx: (128)
        idx = subject_idx.view(-1, 1, 1).expand(X.size(0), self.num_channels, self.num_channels)
        # idx: (128, 271, 271)
        Ms_selected = torch.gather(self.Ms, 0, idx)
        # Ms_selected: (128, 271, 271)
        X = torch.matmul(Ms_selected, X)
        return X.squeeze(-1)

class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_channels, layout=None):
        super(SpatialAttentionLayer, self).__init__()
        self.num_channels = num_channels
        if not layout and os.path.exists(os.path.join("./data", "layout.pt")):
            layout = torch.load(os.path.join("./data", "layout.pt"))
        else:
            raise Exception("preprocess/layout.pyを実行してlayout.ptを生成してください")
        
        self.K = 32
        # shape: (271, 32*32, 2)
        self.layout = layout.unsqueeze(1).expand(-1, self.K**2, -1)
        # (271, 32, 32)
        self.weights_real = nn.Parameter(torch.randn(num_channels, self.K, self.K))
        self.weights_imaginary = nn.Parameter(torch.randn(num_channels, self.K, self.K))
        
        self.epsilon = 1e-8
        self.drop_distance = 0.2

        # shape: (32, 32)
        k_indices, l_indices = torch.meshgrid(torch.arange(self.K), torch.arange(self.K), indexing="ij")
        # shape: (32*32)
        k_indices = k_indices.reshape(-1)
        l_indices = l_indices.reshape(-1)

        # shape: (271, 32*32)
        theta = 2 * math.pi * (k_indices * self.layout[:,:, 0] + l_indices * self.layout[:,:, 1])
        # shape: (271, 32*32)
        self.cos_theta = torch.cos(theta)
        self.sin_theta = torch.sin(theta)

    def forward(self, X):
        # shape: (271, 32*32, 1)
        weights_real = self.weights_real.view(self.num_channels, -1).to(X.device).unsqueeze(2)
        weights_imaginary = self.weights_imaginary.view(self.num_channels, -1).to(X.device).unsqueeze(2)
        # shape: (271, 32*32, 1) -> (1, 32*32, 271)
        cos_theta = self.cos_theta.to(X.device).unsqueeze(0).permute(0, 2, 1)
        sin_theta = self.sin_theta.to(X.device).unsqueeze(0).permute(0, 2, 1)
        # shape: (271, 32*32, 271) - sum -> (271, 271)
        a = torch.sum(weights_real * cos_theta
                         + weights_imaginary * sin_theta, dim=1)
        attention_weights = sum_of_exps_times_tensor(a, X) / sum_of_exps(a)

        # Normalize attention weights
        return attention_weights / (torch.sum(attention_weights, dim=2, keepdim=True) + self.epsilon)

@torch.jit.script
def sum_of_exps(a: torch.Tensor) -> torch.Tensor:
    # shape: (1, 271, 1)
    return torch.sum(torch.exp(a), dim=1).view(1, 271, 1)

@torch.jit.script
def sum_of_exps_times_tensor(a: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # shape: (271, 271)
    # j方向を列にするために入れ替える
    exps = torch.exp(a).permute(1, 0)
    # shape: (1, 271, 271, 1)
    exps = exps.unsqueeze(0).unsqueeze(3)
    # X shape: (batch_size, 271, 1, 281)
    multiplied_tensor = exps * X.unsqueeze(2)
    return torch.sum(multiplied_tensor, dim=1)

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