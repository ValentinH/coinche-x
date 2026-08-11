from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from labels import RANKS, SUITS

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DETECTOR = PIPELINE_ROOT / "model" / "detector" / "detector.onnx"
ARCHIVE_SHA256 = "f824641f1cedc5fa9996342cd09b0314de26a5f3fd34fc1e9103197bbd50ac63"
RANK_DIRECTORY = {
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "J": "jack",
    "Q": "queen",
    "K": "king",
    "A": "ace",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def components(probabilities: np.ndarray) -> list[dict[str, float]]:
    mask = probabilities >= 0.75
    seen = np.zeros(mask.shape, dtype=bool)
    output = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        queue = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        points = []
        while queue:
            y, x = queue.pop()
            points.append((y, x))
            for offset_y, offset_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y = y + offset_y
                next_x = x + offset_x
                if (
                    0 <= next_y < 96
                    and 0 <= next_x < 96
                    and mask[next_y, next_x]
                    and not seen[next_y, next_x]
                ):
                    seen[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if not 9 <= len(points) <= 900:
            continue
        ys = [point[0] for point in points]
        xs = [point[1] for point in points]
        output.append(
            {
                "left": float(min(xs)),
                "top": float(min(ys)),
                "right": float(max(xs) + 1),
                "bottom": float(max(ys) + 1),
                "score": float(
                    0.7 * max(probabilities[y, x] for y, x in points)
                    + 0.3
                    * np.mean([probabilities[y, x] for y, x in points])
                ),
            }
        )
    return output


def candidate_pairs(
    detected: list[dict[str, float]],
) -> list[dict[str, float]]:
    candidates = []
    for first_index, first in enumerate(detected):
        first_center = (
            (first["left"] + first["right"]) / 2,
            (first["top"] + first["bottom"]) / 2,
        )
        for second in detected[first_index + 1 :]:
            second_center = (
                (second["left"] + second["right"]) / 2,
                (second["top"] + second["bottom"]) / 2,
            )
            distance = math.dist(first_center, second_center)
            if not 4 <= distance <= 30:
                continue
            left = min(first["left"], second["left"])
            top = min(first["top"], second["top"])
            right = max(first["right"], second["right"])
            bottom = max(first["bottom"], second["bottom"])
            width = right - left
            height = bottom - top
            if not 8 <= min(width, height) or max(width, height) > 46:
                continue
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            corner_distance = min(
                math.dist((center_x, center_y), corner)
                for corner in ((0, 0), (96, 0), (96, 96), (0, 96))
            )
            candidates.append(
                {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "score": first["score"]
                    + second["score"]
                    - 0.018 * corner_distance,
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def intersection_over_union(
    first: dict[str, float], second: dict[str, float]
) -> float:
    width = max(
        0.0, min(first["right"], second["right"]) - max(first["left"], second["left"])
    )
    height = max(
        0.0, min(first["bottom"], second["bottom"]) - max(first["top"], second["top"])
    )
    intersection = width * height
    first_area = (first["right"] - first["left"]) * (
        first["bottom"] - first["top"]
    )
    second_area = (second["right"] - second["left"]) * (
        second["bottom"] - second["top"]
    )
    return intersection / max(first_area + second_area - intersection, 1.0)


def select_candidates(probabilities: np.ndarray) -> list[dict[str, float]]:
    selected = []
    for candidate in candidate_pairs(components(probabilities)):
        if any(intersection_over_union(candidate, other) > 0.15 for other in selected):
            continue
        selected.append(candidate)
        if len(selected) == 2:
            break
    return selected


def crop_candidate(
    image: Image.Image,
    candidate: dict[str, float],
) -> Image.Image:
    scale_x = image.width / 96.0
    scale_y = image.height / 96.0
    width = (candidate["right"] - candidate["left"]) * scale_x
    height = (candidate["bottom"] - candidate["top"]) * scale_y
    padding = max(width, height) * 0.10
    box = (
        max(0, math.floor(candidate["left"] * scale_x - padding)),
        max(0, math.floor(candidate["top"] * scale_y - padding)),
        min(image.width, math.ceil(candidate["right"] * scale_x + padding)),
        min(image.height, math.ceil(candidate["bottom"] * scale_y + padding)),
    )
    return image.crop(box).resize((96, 96), Image.Resampling.LANCZOS)


def extract(arguments: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(arguments.archive) != ARCHIVE_SHA256:
        raise ValueError("Kaggle v2 archive hash mismatch")
    if arguments.output.exists():
        shutil.rmtree(arguments.output)
    crops_directory = arguments.output / "crops"
    crops_directory.mkdir(parents=True)
    sources = []
    for rank in RANKS:
        for suit in SUITS:
            directory = (
                arguments.train_root
                / f"{RANK_DIRECTORY[rank]} of {suit}"
            )
            for path in sorted(directory.glob("*.jpg")):
                sources.append((rank, suit, path))
    session = ort.InferenceSession(
        str(DETECTOR),
        providers=["CPUExecutionProvider"],
    )
    manifest = []
    for offset in range(0, len(sources), arguments.batch_size):
        batch_sources = sources[offset : offset + arguments.batch_size]
        images = []
        inputs = []
        for _rank, _suit, path in batch_sources:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            images.append(image)
            inputs.append(
                np.asarray(
                    image.resize((384, 384), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).transpose(2, 0, 1)
                / 255.0
            )
        logits = session.run(
            ["mask_logits"],
            {"images": np.stack(inputs).astype(np.float32)},
        )[0][:, 0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        for source_index, ((rank, suit, path), image, heatmap) in enumerate(
            zip(batch_sources, images, probabilities)
        ):
            for candidate_index, candidate in enumerate(
                select_candidates(heatmap)
            ):
                name = (
                    f"{rank}-{suit}--{offset + source_index:05d}"
                    f"--{candidate_index}.jpg"
                )
                output_path = crops_directory / name
                crop_candidate(image, candidate).save(
                    output_path,
                    format="JPEG",
                    quality=94,
                    subsampling=0,
                )
                manifest.append(
                    {
                        "crop": f"crops/{name}",
                        "scene": f"kaggle-train:{path.parent.name}/{path.name}",
                        "sourceScene": str(path.relative_to(arguments.train_root)),
                        "annotationId": f"{path.parent.name}/{path.name}:{candidate_index}",
                        "instanceId": f"kaggle:{path.parent.name}/{path.name}",
                        "corner": f"detector-candidate-{candidate_index}",
                        "visibleFraction": 1.0,
                        "fullyOnCanvas": True,
                        "rank": rank,
                        "suit": suit,
                        "glyph": rank,
                        "alphabet": "english",
                        "typography": "kaggle-real-train",
                        "isIndex": True,
                        "orientationClass": -1,
                        "detectorScore": candidate["score"],
                        "sourceAsset": str(path.relative_to(arguments.train_root)),
                        "sourceSha256": sha256_file(path),
                        "sha256": sha256_file(output_path),
                    }
                )
    manifest_path = arguments.output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            f"{json.dumps(row, separators=(',', ':'), sort_keys=True)}\n"
            for row in manifest
        ),
        encoding="utf-8",
    )
    counts = {
        f"{rank}-{suit}": sum(
            row["rank"] == rank and row["suit"] == suit for row in manifest
        )
        for rank in RANKS
        for suit in SUITS
    }
    summary = {
        "schemaVersion": 1,
        "role": "train",
        "inputSize": 96,
        "cropCount": len(manifest),
        "sceneCount": len({row["scene"] for row in manifest}),
        "datasetSeed": "official-train-detector-crops-v1",
        "source": {
            "id": "kaggle-gpiosenka-cards-v2",
            "role": "training",
            "license": "CC0-1.0",
            "datasetVersion": 2,
            "archiveSha256": ARCHIVE_SHA256,
            "officialPartition": "train",
        },
        "detector": {
            "path": str(DETECTOR.relative_to(PIPELINE_ROOT)),
            "sha256": sha256_file(DETECTOR),
            "threshold": 0.75,
            "candidateConstruction": "paired heatmap components nearest image corners",
        },
        "classCounts": counts,
        "manifestSha256": sha256_file(manifest_path),
    }
    (arguments.output / "summary.json").write_text(
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    arguments.archive = arguments.archive.resolve()
    arguments.train_root = arguments.train_root.resolve()
    arguments.output = arguments.output.resolve()
    if arguments.train_root.name != "train":
        raise ValueError("Only Kaggle's official TRAIN partition is allowed")
    return arguments


if __name__ == "__main__":
    print(json.dumps(extract(parse_arguments()), indent=2, sort_keys=True))
