import torch.nn as nn


class MEGNet(nn.Module):
    def __init__(
        self,
        num_classes=1854,
        num_channels=271,
        seq_len=283,
        dropout=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        dropoutFunc=nn.Dropout,
    ):
        super(MEGNet, self).__init__()
        # shape: (N, 271, 1, 283)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                num_channels,
                F1,
                kernel_size=(1, kernLength),
                padding=(0, kernLength // 2),
                bias=False,
            ),
            # (8, 1, 283)
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(1, 16), groups=F1, bias=False),
            # (16, 1, 268)
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutFunc(dropout),
        )
        # shape: (N, 16, 1, 67)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            # (16, 1, 60)
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropoutFunc(dropout),
        )
        # shape: (N, 16, 1, 8)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (seq_len // 32), num_classes, bias=False)

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.flatten(X)
        return self.dense(X)
