#!/usr/bin/env python3
"""Acquire a deterministic train-only subset of the pinned MIT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


REPOSITORY = "JackFurby/playing-cards"
REVISION = "966aa7cf31d55c255e5b05b545c9a5aafa28dabd"
SEED = "coinche-detector-jackfurby-v1"
ROOT_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
USER_AGENT = "coinche-x-detector-training/1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(relative: str, destination: Path):
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                f"{ROOT_URL}/{relative}", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            os.replace(temporary, destination)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def bilinear(points, horizontal: float, vertical: float):
    top_left, top_right, bottom_left, bottom_right = points
    top = (
        top_left[0] + (top_right[0] - top_left[0]) * horizontal,
        top_left[1] + (top_right[1] - top_left[1]) * horizontal,
    )
    bottom = (
        bottom_left[0] + (bottom_right[0] - bottom_left[0]) * horizontal,
        bottom_left[1] + (bottom_right[1] - bottom_left[1]) * horizontal,
    )
    return {
        "x": top[0] + (bottom[0] - top[0]) * vertical,
        "y": top[1] + (bottom[1] - top[1]) * vertical,
    }


def corner_polygon(points, corner: str):
    if corner == "top-left":
        bounds = (0.0, 0.0, 0.28, 0.36)
    else:
        bounds = (0.72, 0.64, 1.0, 1.0)
    left, top, right, bottom = bounds
    return [
        bilinear(points, left, top),
        bilinear(points, right, top),
        bilinear(points, right, bottom),
        bilinear(points, left, bottom),
    ]


def polygon_box(points, width: int, height: int):
    left = max(0.0, min(point["x"] for point in points))
    top = max(0.0, min(point["y"] for point in points))
    right = min(float(width), max(point["x"] for point in points))
    bottom = min(float(height), max(point["y"] for point in points))
    return {
        "x": left,
        "y": top,
        "width": max(0.0, right - left),
        "height": max(0.0, bottom - top),
    }


def select(metadata, kind: str, count: int):
    ranked = sorted(
        metadata.items(),
        key=lambda item: hashlib.sha256(
            f"{SEED}:{kind}:{item[0]}".encode()
        ).digest(),
    )
    return ranked[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single", type=int, default=512)
    parser.add_argument("--three", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    upstream = output / "upstream"
    sources = {
        "README.md": upstream / "README.md",
        "splits/single/train.json": upstream / "single-train.json",
        "splits/single/val.json": upstream / "single-val.json",
        "splits/three/train.json": upstream / "three-train.json",
        "splits/three/val.json": upstream / "three-val.json",
    }
    for relative, destination in sources.items():
        fetch(relative, destination)

    selected = []
    requested = {"single": args.single, "three": args.three}
    for kind in ("single", "three"):
        metadata = json.loads((upstream / f"{kind}-train.json").read_text())
        selected.extend(
            (kind, identifier, sample)
            for identifier, sample in select(metadata, kind, requested[kind])
        )
    selected_paths = {sample["img_path"] for _, _, sample in selected}
    validation_paths = {
        sample["img_path"]
        for kind in ("single", "three")
        for sample in json.loads(
            (upstream / f"{kind}-val.json").read_text()
        ).values()
    }
    leakage = selected_paths & validation_paths
    if leakage:
        raise RuntimeError(
            f"Selected images overlap upstream validation: {sorted(leakage)[:3]}"
        )

    images = output / "images"
    download_specs = [
        (
            sample["img_path"],
            images / f"{kind}-{int(identifier):05d}.png",
        )
        for kind, identifier, sample in selected
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda spec: fetch(*spec), download_specs))

    frames = []
    annotation_id = 1
    for (kind, identifier, sample), (_, image_path) in zip(
        selected, download_specs
    ):
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = []
        for card_index, (card_points, card_class) in enumerate(
            sample["card_points"]
        ):
            for corner in ("top-left", "bottom-right"):
                polygon = corner_polygon(card_points, corner)
                box = polygon_box(polygon, width, height)
                annotations.append(
                    {
                        "id": annotation_id,
                        "instanceId": (
                            f"{kind}-{identifier}-{card_index}-{card_class}"
                        ),
                        "corner": corner,
                        "bbox": box,
                        "polygon": polygon,
                        "visibleFraction": 1.0,
                        "visibilitySource": "not-supplied-projected-train-only",
                        "upstreamCardClass": card_class,
                    }
                )
                annotation_id += 1
        frames.append(
            {
                "image": f"images/{image_path.name}",
                "width": width,
                "height": height,
                "cardCount": len(sample["card_points"]),
                "mode": f"jackfurby-{kind}",
                "upstreamImage": sample["img_path"],
                "annotations": annotations,
            }
        )

    dataset = {
        "schemaVersion": 1,
        "split": "train",
        "seed": SEED,
        "source": {
            "id": "jackfurby-playing-cards",
            "role": "training-only",
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "MIT",
        },
        "generator": {
            "version": 1,
            "annotations": "two-corner-index-regions-projected-from-card-quads",
            "normalizedRegions": {
                "topLeft": [0.0, 0.0, 0.28, 0.36],
                "bottomRight": [0.72, 0.64, 1.0, 1.0],
            },
            "visibility": "upstream-not-supplied; training-only labels",
        },
        "coverage": {
            "singleScenes": args.single,
            "threeCardScenes": args.three,
        },
        "frames": frames,
    }
    dataset_path = output / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    provenance = {
        "schemaVersion": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "MIT",
        "rootUrl": ROOT_URL,
        "selection": {
            "algorithm": "ascending SHA-256(seed:subset:sample-id)",
            "seed": SEED,
            "counts": requested,
            "trainSplitsOnly": True,
            "upstreamValidationOverlap": 0,
        },
        "upstreamMetadata": {
            relative: {
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
            for relative, destination in sources.items()
        },
        "images": [
            {
                "upstream": relative,
                "local": f"images/{destination.name}",
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
            for relative, destination in download_specs
        ],
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    checksum_paths = [
        dataset_path,
        provenance_path,
        *sources.values(),
        *(destination for _, destination in download_specs),
    ]
    lines = [
        f"{sha256(path)}  {path.relative_to(output)}" for path in checksum_paths
    ]
    (output / "dataset.sha256").write_text("\n".join(sorted(lines)) + "\n")
    print(
        json.dumps(
            {
                "scenes": len(frames),
                "single": args.single,
                "three": args.three,
                "indices": annotation_id - 1,
                "bytes": sum(path.stat().st_size for _, path in download_specs),
                "revision": REVISION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
