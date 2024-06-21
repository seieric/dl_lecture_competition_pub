import torch
import torch.nn as nn
import math


class SubjectLayer(nn.Module):
    def __init__(self, num_subjects, num_channels):
        super(SubjectLayer, self).__init__()
        self.num_subjects = num_subjects
        self.num_channels = num_channels

        self.Ms = nn.Parameter(torch.Tensor(num_subjects, num_channels, num_channels))
        nn.init.uniform_(self.Ms, -math.sqrt(3), math.sqrt(3))

    def forward(self, X, subject_idx):
        # subject_idx: (128)
        idx = subject_idx.view(-1, 1, 1).expand(
            X.size(0), self.num_channels, self.num_channels
        )
        # idx: (128, 271, 271)
        Ms_selected = torch.gather(self.Ms, 0, idx)
        # Ms_selected: (128, 271, 271)
        X = torch.matmul(Ms_selected, X)
        return X.squeeze(-1)
