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
            self.layout = torch.load(os.path.join("./data", "layout.pt")).to("cuda")
        else:
            raise Exception("preprocess/layout.pyを実行してlayout.ptを生成してください")
        
        self.K = 32
        # (271, 32, 32)
        self.weights_real = nn.Parameter(torch.randn(num_channels, self.K, self.K))
        self.weights_imaginary = nn.Parameter(torch.randn(num_channels, self.K, self.K))
        
        self.epsilon = 1e-8
        self.drop_distance = 0.2

    def forward(self, X):
        # X: (128, 271, 281)
        # attention_weights: (128, 271, 281)
        attention_weights = torch.zeros(X.size(0), self.num_channels, 281).to("cuda")
        
        for j in range(self.num_channels):
            # (271)
            a_j = torch.zeros(self.num_channels).to("cuda")
            for k in range(self.K):
                for l in range(self.K):
                    # (271)
                    theta = 2 * math.pi * (k * self.layout[:, 0] + l * self.layout[:, 1])
                    # (271)
                    a_j += self.weights_real[j, k, l] * torch.cos(theta) + self.weights_imaginary[j, k, l] * torch.sin(theta)
            # jのattention_weightsを求める
            # attetion_weights: (128, 271)
            # output: (128, 281)
            attention_weights[:, j, :] = sum_of_exps_times_tensor(a_j, X) / sum_of_exps(a_j)

        # spatial dropout/空間的ドロップアウト
        drop_position = torch.rand(X.size(0), 2).to("cuda")
        distances = torch.sqrt(torch.sum((drop_position.unsqueeze(1) - self.layout.unsqueeze(0))**2, dim=2))
        # drop_distanceの範囲にある入力を除去
        mask = distances < self.drop_distance
        attention_weights[:, :, mask] = 0

        # Normalize attention weights
        return attention_weights / (torch.sum(attention_weights, dim=2, keepdim=True) + self.epsilon)

@torch.jit.script
def sum_of_exps(tensor):
    return torch.sum(torch.exp(tensor))

@torch.jit.script
def sum_of_exps_times_tensor(a, T):
    # (271)
    exps = torch.exp(a)
    #  出力チャンネル271, 入力チャンネル271
    # (271) * (128, 271, 281)
    multiplied_tensor = exps * T.view(128, 281, 271)
    result = torch.sum(multiplied_tensor, dim=2)
    return result

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