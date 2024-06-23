import torch.nn as nn
import torch.nn.functional as F


class ResNet152GELU(nn.Module):
    def __init__(self, in_channels, num_classes=1854, dropout=None):
        super(ResNet152GELU, self).__init__()

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

        self.conv2_x = self._make_layer(64, 64, 256, 3, dropout)
        self.conv3_x = self._make_layer(256, 128, 512, 8, dropout, stride=(2, 2))
        self.conv4_x = self._make_layer(512, 256, 1024, 36, dropout, stride=(2, 2))
        self.conv5_x = self._make_layer(1024, 512, 2048, 3, dropout, stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes, bias=True),
        )

    def _make_layer(
        self,
        in_channels,
        hid_channels,
        out_channels,
        num_blocks,
        dropout,
        stride=(1, 1),
    ):
        layers = []
        layers.append(
            BasicBlock(
                in_channels, hid_channels, out_channels, stride=stride, dropout=dropout
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(out_channels, hid_channels, out_channels, dropout=dropout)
            )
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


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, hid_channels, out_channels, stride=(1, 1), dropout=None
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            hid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            hid_channels,
            hid_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            hid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            hid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.bn2 = nn.BatchNorm2d(
            hid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.bn3 = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, X):
        Y = F.gelu(self.bn1(self.conv1(X)))
        Y = F.gelu(self.bn2(self.conv2(Y)))
        Y = F.gelu(self.bn3(self.conv3(Y)))
        if self.in_channels == self.out_channels:
            Y += X
        if self.dropout:
            Y = self.dropout(Y)
        return Y
