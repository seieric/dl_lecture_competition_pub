import torch
import torch.nn as nn
from ..layers.subject_layer import SubjectLayer
from ..layers.megnet import MEGNet


class MEGEncoder(nn.Module):
    def __init__(self, dropout=0) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(271, 271, kernel_size=1, stride=1, padding=1)
        self.subject_layer = SubjectLayer(4, 271)
        self.megnet = MEGNet(dropout=dropout)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.conv1d(X)
        X = self.subject_layer(X, subject_idx)
        return self.megnet(X)
