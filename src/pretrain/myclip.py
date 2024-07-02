# CLIPみたいなものを実装
import torch
import numpy as np
import torch.nn as nn
from image_encoder import ImageEncoder
from meg_encoder import MEGEncoder


class MyCLIP(nn.Module):
    def __init__(self, meg_dropout=0):
        super(MyCLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = nn.Parameter(torch.randn(2048, 512))

        self.meg_encoder = MEGEncoder(dropout=meg_dropout)
        self.meg_projection = nn.Parameter(torch.randn(4096, 512))

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _encode_image(self, X: torch.Tensor) -> torch.Tensor:
        X = self.image_encoder(X)
        return X @ self.image_projection

    def _encode_meg(self, X: torch.Tensor) -> torch.Tensor:
        X = self.meg_encoder(X)
        return X @ self.meg_projection

    def forward(
        self, image: torch.Tensor, meg: torch.Tensor, subject_idx: torch.Tensor
    ) -> torch.Tensor:
        image = self._encode_image(image)
        meg = self._encode_meg(meg, subject_idx)

        logits_per_image = (image @ meg.T) * self.temperature.exp()
        logits_per_meg = (meg @ image.T) * self.temperature.exp()

        return logits_per_image, logits_per_meg
