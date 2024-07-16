# CLIPみたいなものを実装
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .meg_encoder import MEGEncoder


class MyCLIP(nn.Module):
    def __init__(self, pretrained_weights, meg_dropout=0):
        super(MyCLIP, self).__init__()
        self.image_encoder = ImageEncoder(pretrained_weights)

        self.meg_encoder = MEGEncoder(dropout=meg_dropout)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _encode_image(self, X: torch.Tensor) -> torch.Tensor:
        X = self.image_encoder(X)
        return X

    def _encode_meg(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.meg_encoder(X, subject_idx)
        return X

    def _l2_normalize(self, X: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(X**2, dim=0)
        return X / l2_norm

    def loss(
        self, image: torch.Tensor, meg: torch.Tensor, subject_idx: torch.Tensor
    ) -> torch.Tensor:
        # 画像を512次元の特徴量で表現 (batch_size, 512)
        image = self._l2_normalize(self._encode_image(image))
        # MEGを512次元の特徴量で表現 (batch_size, 512)
        meg = self._l2_normalize(self._encode_meg(meg, subject_idx))

        # (batch_size, batch_size)
        logits = (image @ meg.T) * self.temperature.exp()

        labels = torch.arange(logits.shape[0]).to(image.device)
        image_loss = F.cross_entropy(logits, labels, reduction="none")
        meg_loss = F.cross_entropy(logits.T, labels, reduction="none")

        loss = (image_loss + meg_loss) / 2.0

        return loss.mean()
