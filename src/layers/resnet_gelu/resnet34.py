import torch
import torch.nn as nn
import torch.nn.functional as F

from .basicblock import BasicBlock


class ResNet34GELU(nn.Module):
    def __init__(self, in_channels, num_classes=1854, dropout=None):
        super(ResNet34GELU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.GELU(),
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.conv2_x = self._make_layer(64, 64, 3, dropout)
        self.conv3_x = self._make_layer(64, 128, 4, dropout, stride=(2, 2))
        self.conv4_x = self._make_layer(128, 256, 6, dropout, stride=(2, 2))
        self.conv5_x = self._make_layer(256, 512, 3, dropout, stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=True),
        )

    def _make_layer(
        self, in_channels, out_channels, num_blocks, dropout, stride=(1, 1)
    ):
        layers = []
        layers.append(
            BasicBlock(in_channels, out_channels, stride=stride, dropout=dropout)
        )
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.conv1(X)
        X = self.maxpool(X)
        X = self.conv2_x(X)
        X = self.conv3_x(X)
        X = self.conv4_x(X)
        X = self.conv5_x(X)
        X = self.avgpool(X)
        return self.fc(X)
