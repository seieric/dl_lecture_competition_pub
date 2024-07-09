import torch
import torch.nn as nn

from .pretrain.meg_encoder import MEGEncoder


class MyModel(nn.Module):
    def __init__(self, dropout=0) -> None:
        super().__init__()
        self.encoder = MEGEncoder(dropout=dropout)
        self.fc = nn.Linear(512, 1854)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X, subject_idx)
        return self.fc(X)
