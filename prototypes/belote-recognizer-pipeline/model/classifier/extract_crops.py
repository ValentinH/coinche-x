from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from labels import validate_factorized_label

EXPECTED_SOURCE = {
    "train": ("train", "andrew-tidey-cards-pack", "training"),
    "validation": ("validation", "andrew-tidey-cards-pack", "training"),
    "holdout": ("holdout", "greywyvern-cardset", "holdout-only"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def safe_image_path(dataset_directory: Path, relative_path: str) -> Path:
    image_path = (dataset_directory / relative_path).resolve()
    if os.path.commonpath((dataset_directory.resolve(), image_path)) != str(
        dataset_directory.resolve()
    ):
        raise ValueError(f"Image escapes dataset directory: {relative_path}")
    return image_path


def recognizer_crop(
    image: Image.Image,
    bbox: dict[str, float],
    *,
    padding_fraction: float,
    input_size: int,
) -> Image.Image:
    width = float(bbox["width"])
    height = float(bbox["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox: {bbox}")
    padding = max(width, height) * padding_fraction
    box = (
        max(0, math.floor(float(bbox["x"]) - padding)),
        max(0, math.floor(float(bbox["y"]) - padding)),
        min(image.width, math.ceil(float(bbox["x"]) + width + padding)),
        min(image.height, math.ceil(float(bbox["y"]) + height + padding)),
    )
    return image.crop(box).resize((input_size, input_size), Image.Resampling.LANCZOS)


def verify_dataset(dataset_directory: Path, role: str) -> dict[str, Any]:
    dataset_path = dataset_directory / "dataset.json"
    labels_path = dataset_directory / "classifier-labels.jsonl"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_split, expected_source, expected_role = EXPECTED_SOURCE[role]
    actual = (
        dataset.get("split"),
        dataset.get("source", {}).get("id"),
        dataset.get("source", {}).get("role"),
    )
    if actual != (expected_split, expected_source, expected_role):
        raise ValueError(
            f"{role} source firewall rejected {actual}; expected "
            f"{(expected_split, expected_source, expected_role)}"
        )
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    return dataset


def extract(
    dataset_directory: Path,
    output_directory: Path,
    *,
    role: str,
    input_size: int,
    padding_fraction: float,
) -> dict[str, Any]:
    dataset = verify_dataset(dataset_directory, role)
    labels_path = dataset_directory / "classifier-labels.jsonl"
    annotations = {
        annotation["id"]: annotation
        for frame in dataset["frames"]
        for annotation in frame["annotations"]
    }
    crops_directory = output_directory / "crops"
    if crops_directory.exists():
        shutil.rmtree(crops_directory)
    crops_directory.mkdir(parents=True)

    manifest: list[dict[str, Any]] = []
    cached_image_path: Path | None = None
    cached_image: Image.Image | None = None
    try:
        for crop_index, label in enumerate(read_json_lines(labels_path)):
            validate_factorized_label(label["rank"], label["suit"], label["glyph"])
            annotation = annotations[label["annotationId"]]
            image_path = safe_image_path(dataset_directory, label["image"])
            if image_path != cached_image_path:
                if cached_image is not None:
                    cached_image.close()
                cached_image = Image.open(image_path).convert("RGB")
                cached_image_path = image_path
            assert cached_image is not None
            crop = recognizer_crop(
                cached_image,
                label["bbox"],
                padding_fraction=padding_fraction,
                input_size=input_size,
            )
            crop_name = f"{crop_index:06d}.png"
            crop_path = crops_directory / crop_name
            crop.save(crop_path, format="PNG", optimize=False)
            manifest.append(
                {
                    "crop": f"crops/{crop_name}",
                    "scene": f"{dataset['split']}:{label['image']}",
                    "sourceScene": label["image"],
                    "annotationId": label["annotationId"],
                    "instanceId": annotation["instanceId"],
                    "corner": annotation["corner"],
                    "visibleFraction": annotation["visibleFraction"],
                    "fullyOnCanvas": all(
                        0 <= point["x"] <= dataset["width"]
                        and 0 <= point["y"] <= dataset["height"]
                        for point in annotation["polygon"]
                    ),
                    "rank": label["rank"],
                    "suit": label["suit"],
                    "glyph": label["glyph"],
                    "alphabet": label["alphabet"],
                    "sha256": sha256_file(crop_path),
                }
            )
    finally:
        if cached_image is not None:
            cached_image.close()

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.jsonl"
    manifest_path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in manifest),
        encoding="utf-8",
    )
    summary = {
        "schemaVersion": 1,
        "role": role,
        "inputSize": input_size,
        "paddingFraction": padding_fraction,
        "cropCount": len(manifest),
        "sceneCount": len({row["scene"] for row in manifest}),
        "datasetSeed": dataset["seed"],
        "source": dataset["source"],
        "labelsSha256": sha256_file(labels_path),
        "manifestSha256": sha256_file(manifest_path),
    }
    (output_directory / "summary.json").write_text(
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(EXPECTED_SOURCE), required=True)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--padding-fraction", type=float, default=0.06)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = extract(
        arguments.dataset.resolve(),
        arguments.output.resolve(),
        role=arguments.role,
        input_size=arguments.input_size,
        padding_fraction=arguments.padding_fraction,
    )
    print(json.dumps(result, sort_keys=True))
