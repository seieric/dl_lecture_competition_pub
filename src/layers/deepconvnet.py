import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(
        self, num_classes=1854, num_channels=271, dropout=0.5, dropoutFunc=nn.Dropout
    ):
        super(DeepConvNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(num_channels, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 50, kernel_size=(1, 1)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            dropoutFunc(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            dropoutFunc(dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            dropoutFunc(dropout),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            dropoutFunc(dropout),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2600, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        return self.head(X)
