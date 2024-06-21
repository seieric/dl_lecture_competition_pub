import torch
import torch.nn as nn
import math
import os


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
        # (271, 32*32, 1)
        self.weights_real = nn.Parameter(torch.Tensor(num_channels, self.K**2, 1))
        self.weights_imaginary = nn.Parameter(torch.Tensor(num_channels, self.K**2, 1))
        nn.init.xavier_uniform_(self.weights_real)
        nn.init.xavier_uniform_(self.weights_imaginary)

        self.epsilon = 1e-8

        # shape: (32, 32)
        k_indices, l_indices = torch.meshgrid(
            torch.arange(self.K), torch.arange(self.K), indexing="ij"
        )
        # shape: (32*32)
        k_indices = k_indices.reshape(-1)
        l_indices = l_indices.reshape(-1)

        # shape: (271, 32*32)
        theta = (
            2
            * math.pi
            * (k_indices * self.layout[:, :, 0] + l_indices * self.layout[:, :, 1])
        )
        # shape: (271, 32*32) -> (271, 32*32, 1) -> (1, 32*32, 271)
        self.cos_theta = nn.Parameter(
            torch.cos(theta).unsqueeze(0).permute(0, 2, 1), requires_grad=False
        )
        self.sin_theta = nn.Parameter(
            torch.sin(theta).unsqueeze(0).permute(0, 2, 1), requires_grad=False
        )

    def forward(self, X):
        # shape: (271, 32*32, 271) - sum -> (271, 271)
        a = torch.sum(
            self.weights_real * self.cos_theta
            + self.weights_imaginary * self.sin_theta,
            dim=1,
        )
        attention_weights = sum_of_exps_times_tensor(a, X) / sum_of_exps(a)

        # Normalize attention weights
        return attention_weights / (
            torch.sum(attention_weights, dim=2, keepdim=True) + self.epsilon
        )


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
