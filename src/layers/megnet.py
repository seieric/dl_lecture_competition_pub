import torch.nn as nn


class MEGNet(nn.Module):
    def __init__(
        self,
        num_classes=1854,
        num_channels=271,
        dropout=0.5,
        dropoutFunc=nn.Dropout,
    ):
        super(MEGNet, self).__init__()
        # shape: (N, 271, 1, 283)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                num_channels,
                256,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=(1, 16), groups=256, bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutFunc(dropout),
        )
        # shape: (N, 512, 1, 67)
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 8), padding=(0, 8), bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutFunc(dropout),
        )
        # shape: (N, 512, 1, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 4), padding=(0, 0), bias=False),
            nn.BatchNorm2d(1024),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutFunc(dropout),
        )
        # shape: (N, 1024, 1, 4)
        self.block4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(2048),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            dropoutFunc(dropout),
        )
        # shape: (N, 2048, 1, 2)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 2, num_classes, bias=False),
        )

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        return self.dense(X)
