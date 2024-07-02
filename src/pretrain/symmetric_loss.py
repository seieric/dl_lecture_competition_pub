import torch
import torch.nn as nn


class SymmetricLoss(nn.Module):
    def __init__(self) -> None:
        super(SymmetricLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_meg: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        image_loss = self.criterion(logits_per_image, y)
        meg_loss = self.criterion(logits_per_meg, y)

        return image_loss + meg_loss
