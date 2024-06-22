import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), dropout=None):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, X):
        if self.in_channels == self.out_channels:
            X = F.gelu(self.bn1(self.conv1(X))) + X
        else:
            X = F.gelu(self.bn1(self.conv1(X)))
        X = F.gelu(self.bn2(self.conv2(X))) + X
        if self.dropout:
            X = self.dropout(X)
        return X
