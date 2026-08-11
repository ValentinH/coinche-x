from __future__ import annotations

import torch
from torch import nn

from labels import RANKS, SUITS

INPUT_SIZE = 96


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            ConvNormActivation(
                input_channels,
                input_channels,
                stride=2,
                groups=input_channels,
            ),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class TinyFactorizedClassifier(nn.Module):
    """Small browser-oriented CNN with independent canonical rank/suit heads."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation(3, 32, stride=2),
            ConvNormActivation(32, 64, stride=2),
            ConvNormActivation(64, 128, stride=2),
            ConvNormActivation(128, 192, stride=2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
        )
        self.rank_head = nn.Linear(512, len(RANKS))
        self.suit_head = nn.Linear(512, len(SUITS))
        self.index_head = nn.Linear(512, 1)
        self.card_head = nn.Linear(512, len(RANKS) * len(SUITS))
        self.orientation_head = nn.Linear(512, 1)
        nn.init.zeros_(self.orientation_head.weight)
        nn.init.zeros_(self.orientation_head.bias)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = images.shape[0]
        transposed = images.transpose(-2, -1)
        rotated = torch.cat(
            [
                images,
                transposed.flip(-2),
                images.flip((-2, -1)),
                transposed.flip(-1),
            ],
            dim=0,
        )
        embeddings = self.embedding(self.features(rotated)).reshape(
            4, batch_size, -1
        )
        orientation_logits = self.orientation_head(embeddings).squeeze(-1).T
        orientation_weights = torch.softmax(orientation_logits, dim=1).T.unsqueeze(-1)
        embedding = (embeddings * orientation_weights).sum(dim=0)
        return (
            self.rank_head(embedding),
            self.suit_head(embedding),
            self.index_head(embedding).squeeze(1),
            self.card_head(embedding),
            orientation_logits,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
