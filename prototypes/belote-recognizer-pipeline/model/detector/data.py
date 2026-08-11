"""Scene-level detector data loading. No smoke-data access."""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model import INPUT_SIZE, OUTPUT_SIZE


TILE_SIZE = 960
TILE_OVERLAP = 192
MINIMUM_VISIBLE_FRACTION = 0.9


def tile_starts(length: int) -> list[int]:
    if length <= TILE_SIZE:
        return [0]
    last = length - TILE_SIZE
    starts = list(range(0, last, TILE_SIZE - TILE_OVERLAP))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


@dataclass(frozen=True)
class Tile:
    frame_index: int
    x: int
    y: int


class DetectorTiles(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        augment: bool,
        seed: int,
        expected_source: str = "andrew-tidey-cards-pack",
    ):
        self.dataset_dir = dataset_dir.resolve()
        with (self.dataset_dir / "dataset.json").open() as handle:
            self.manifest = json.load(handle)
        if self.manifest["source"]["id"] != expected_source:
            raise ValueError(
                f"Expected source {expected_source}, got "
                f"{self.manifest['source']['id']}"
            )
        self.frames = self.manifest["frames"]
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.image_cache = {}
        self.target_cache = {}
        self.tiles = [
            Tile(frame_index, x, y)
            for frame_index, frame in enumerate(self.frames)
            for y in tile_starts(frame["height"])
            for x in tile_starts(frame["width"])
        ]

    def __len__(self) -> int:
        return len(self.tiles)

    def _rng(self, index: int) -> random.Random:
        # DataLoader order does not affect augmentation.
        return random.Random(
            self.seed * 1_000_003 + self.epoch * 10_007 + index
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        tile = self.tiles[index]
        frame = self.frames[tile.frame_index]
        right = min(frame["width"], tile.x + TILE_SIZE)
        bottom = min(frame["height"], tile.y + TILE_SIZE)
        if index in self.image_cache:
            image = self.image_cache[index].copy()
        else:
            source = Image.open(self.dataset_dir / frame["image"]).convert("RGB")
            crop = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "#202020")
            crop.paste(source.crop((tile.x, tile.y, right, bottom)), (0, 0))
            crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
            image = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1).copy()
            self.image_cache[index] = image.copy()
        if index in self.target_cache:
            heatmap, offsets, sizes, active, boxes = self.target_cache[index]
        else:
            heatmap = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
            offsets = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
            sizes = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
            active = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
            scale = OUTPUT_SIZE / TILE_SIZE
            boxes = []
            for annotation in frame["annotations"]:
                if annotation["visibleFraction"] < MINIMUM_VISIBLE_FRACTION:
                    continue
                box = annotation["bbox"]
                center_x = box["x"] + box["width"] / 2
                center_y = box["y"] + box["height"] / 2
                if not (
                    tile.x <= center_x < right
                    and tile.y <= center_y < bottom
                ):
                    continue
                left = max(0, int(np.floor((box["x"] - tile.x) * scale)))
                top = max(0, int(np.floor((box["y"] - tile.y) * scale)))
                box_right = min(
                    OUTPUT_SIZE,
                    int(np.ceil((box["x"] + box["width"] - tile.x) * scale)),
                )
                box_bottom = min(
                    OUTPUT_SIZE,
                    int(np.ceil((box["y"] + box["height"] - tile.y) * scale)),
                )
                if box_right <= left or box_bottom <= top:
                    continue
                center_output_x = (center_x - tile.x) * scale
                center_output_y = (center_y - tile.y) * scale
                cell_x = min(OUTPUT_SIZE - 1, int(center_output_x))
                cell_y = min(OUTPUT_SIZE - 1, int(center_output_y))
                radius = max(
                    1.0,
                    min(box_right - left, box_bottom - top) * 0.18,
                )
                radius_cells = int(np.ceil(radius * 3))
                gaussian_left = max(0, cell_x - radius_cells)
                gaussian_right = min(OUTPUT_SIZE, cell_x + radius_cells + 1)
                gaussian_top = max(0, cell_y - radius_cells)
                gaussian_bottom = min(OUTPUT_SIZE, cell_y + radius_cells + 1)
                yy, xx = np.mgrid[
                    gaussian_top:gaussian_bottom, gaussian_left:gaussian_right
                ]
                gaussian = np.exp(
                    -(
                        (xx - center_output_x) ** 2
                        + (yy - center_output_y) ** 2
                    )
                    / (2 * radius**2)
                )
                heatmap[
                    gaussian_top:gaussian_bottom,
                    gaussian_left:gaussian_right,
                ] = np.maximum(
                    heatmap[
                        gaussian_top:gaussian_bottom,
                        gaussian_left:gaussian_right,
                    ],
                    gaussian,
                )
                heatmap[cell_y, cell_x] = 1.0
                offsets[:, cell_y, cell_x] = (
                    center_output_x - cell_x,
                    center_output_y - cell_y,
                )
                sizes[:, cell_y, cell_x] = (
                    (box_right - left) / OUTPUT_SIZE,
                    (box_bottom - top) / OUTPUT_SIZE,
                )
                active[cell_y, cell_x] = 1.0
                boxes.append((left, top, box_right, box_bottom))
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            self.target_cache[index] = (
                heatmap,
                offsets,
                sizes,
                active,
                boxes,
            )

        if self.augment:
            rng = self._rng(index)
            if rng.random() < 0.5:
                image = np.ascontiguousarray(image[:, :, ::-1])
                heatmap = np.ascontiguousarray(heatmap[:, ::-1])
                offsets = np.ascontiguousarray(offsets[:, :, ::-1])
                offsets[0] = 1.0 - offsets[0]
                sizes = np.ascontiguousarray(sizes[:, :, ::-1])
                active = np.ascontiguousarray(active[:, ::-1])
                boxes = [
                    (OUTPUT_SIZE - right, top, OUTPUT_SIZE - left, bottom)
                    for left, top, right, bottom in boxes
                ]

        target = {
            "heatmap": torch.from_numpy(heatmap[None]),
            "offsets": torch.from_numpy(offsets),
            "sizes": torch.from_numpy(sizes),
            "active": torch.from_numpy(active),
            "boxes": torch.from_numpy(np.asarray(boxes, dtype=np.float32)),
            "scene": frame["image"],
            "tile": (tile.x, tile.y),
        }
        return torch.from_numpy(image), target


class CombinedDetectorTiles(Dataset):
    def __init__(self, datasets: list[DetectorTiles]):
        self.datasets = datasets
        self.boundaries = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.boundaries.append(total)
        self.frames = [
            frame for dataset in datasets for frame in dataset.frames
        ]

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, index):
        dataset_index = bisect_right(self.boundaries, index)
        previous = self.boundaries[dataset_index - 1] if dataset_index else 0
        return self.datasets[dataset_index][index - previous]

    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
