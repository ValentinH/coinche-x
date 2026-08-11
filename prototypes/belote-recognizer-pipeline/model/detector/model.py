"""Tiny permissively licensed corner-index center/box network."""

from __future__ import annotations

import torch
from torch import nn


INPUT_SIZE = 384
OUTPUT_SIZE = 96
MAX_CANDIDATES = 128


def normalization_groups(channels: int) -> int:
    return next(
        groups for groups in range(min(8, channels), 0, -1)
        if channels % groups == 0
    )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class PointwiseNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(normalization_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(normalization_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(normalization_groups(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class TinyCornerDetector(nn.Module):
    """384 px RGB tile to 96x96 center logits and box regression."""

    def __init__(self):
        super().__init__()
        self.stem = ConvNormAct(3, 20, stride=2)
        self.quarter = ConvNormAct(20, 40, stride=2)
        self.eighth = nn.Sequential(
            SeparableNormAct(40, 64, stride=2),
            ResidualBlock(64),
            ResidualBlock(64, dilation=2),
        )
        self.sixteenth = nn.Sequential(
            SeparableNormAct(64, 96, stride=2),
            ResidualBlock(96),
            ResidualBlock(96, dilation=2),
            ResidualBlock(96, dilation=4),
        )
        self.decode_eighth = nn.Sequential(
            PointwiseNormAct(96 + 64, 64),
            ResidualBlock(64),
        )
        self.decode_quarter = nn.Sequential(
            PointwiseNormAct(64 + 40, 48),
            ResidualBlock(48),
            ResidualBlock(48, dilation=2),
        )
        self.center_head = nn.Sequential(
            SeparableNormAct(48, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.box_head = nn.Sequential(
            SeparableNormAct(48, 32),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        with torch.no_grad():
            self.center_head[-1].bias.fill_(-4.0)
            self.box_head[-1].bias[:2] = 0.0
            self.box_head[-1].bias[2:] = -0.85

    def forward(self, images: torch.Tensor):
        half = self.stem(images)
        quarter = self.quarter(half)
        eighth = self.eighth(quarter)
        sixteenth = self.sixteenth(eighth)
        decoded_eighth = nn.functional.interpolate(
            sixteenth, size=eighth.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_eighth = self.decode_eighth(
            torch.cat((decoded_eighth, eighth), dim=1)
        )
        decoded_quarter = nn.functional.interpolate(
            decoded_eighth,
            size=quarter.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_quarter = self.decode_quarter(
            torch.cat((decoded_quarter, quarter), dim=1)
        )
        return self.center_head(decoded_quarter), self.box_head(decoded_quarter)


class ExportDetector(nn.Module):
    """Decode local center peaks to the browser detector contract."""

    def __init__(self, detector: TinyCornerDetector):
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor):
        center_logits, regression = self.detector(images)
        scores = center_logits[:, 0].sigmoid()
        local_maximum = nn.functional.max_pool2d(
            scores[:, None], kernel_size=7, stride=1, padding=3
        )[:, 0]
        scores = torch.where(scores >= local_maximum, scores, torch.zeros_like(scores))
        offsets = regression[:, :2].sigmoid()
        sizes = regression[:, 2:].sigmoid() * 0.5
        coordinates = torch.arange(
            OUTPUT_SIZE, dtype=images.dtype, device=images.device
        )
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        center_x = (grid_x[None] + offsets[:, 0]) / OUTPUT_SIZE
        center_y = (grid_y[None] + offsets[:, 1]) / OUTPUT_SIZE
        boxes = torch.stack(
            (
                center_x - sizes[:, 0] / 2,
                center_y - sizes[:, 1] / 2,
                center_x + sizes[:, 0] / 2,
                center_y + sizes[:, 1] / 2,
            ),
            dim=-1,
        ).clamp(0, 1)
        flat_scores = scores.flatten(1)
        flat_boxes = boxes.flatten(1, 2)
        top_scores, indices = torch.topk(
            flat_scores, MAX_CANDIDATES, dim=1, sorted=True
        )
        top_boxes = torch.gather(
            flat_boxes,
            1,
            indices[:, :, None].expand(-1, -1, 4),
        )
        return top_boxes, top_scores
