from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from labels import RANKS, SUITS, validate_factorized_label

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parents[1]
SOURCE_ROOT = PIPELINE_ROOT / "training" / "sources" / "andrew-tidey"
INPUT_SIZE = 96
FACE_GLYPHS = {
    "J": ("J", "V"),
    "Q": ("Q", "D"),
    "K": ("K", "R"),
}
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf"),
)
BACKGROUND_PALETTES = (
    ((22, 48, 39), (77, 105, 88)),
    ((35, 45, 57), (94, 111, 127)),
    ((67, 49, 39), (137, 105, 80)),
    ((54, 43, 61), (112, 88, 121)),
    ((176, 169, 151), (231, 226, 210)),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_seed(dataset_seed: str, label: str, index: int) -> int:
    value = hashlib.sha256(
        f"{dataset_seed}\0{label}\0{index}".encode("utf-8")
    ).digest()
    return int.from_bytes(value[:8], "big")


def resolve_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("A pinned typography source font is required")


def canonical_corner(card: Image.Image, corner: str) -> Image.Image:
    if corner == "bottom-right":
        card = card.rotate(180)
    width = round(card.width * 0.28)
    height = round(card.height * 0.36)
    return card.crop((0, 0, width, height))


def typography_variant(
    corner: Image.Image,
    *,
    rank: str,
    suit: str,
    variant: str,
    font_path: Path,
) -> tuple[Image.Image, str]:
    if rank not in FACE_GLYPHS or variant == "andrew-english":
        return corner, rank
    glyph = FACE_GLYPHS[rank][1 if variant == "rendered-french" else 0]
    output = corner.copy()
    draw = ImageDraw.Draw(output)
    paper = output.convert("RGB").getpixel(
        (min(output.width - 1, 27), min(output.height - 1, 7))
    )
    draw.rectangle((4, 3, 25, 23), fill=(*paper, 255))
    font = ImageFont.truetype(str(font_path), 18)
    color = (239, 24, 54, 255) if suit in {"diamonds", "hearts"} else (14, 17, 18, 255)
    draw.text((6, 1), glyph, font=font, fill=color, stroke_width=0)
    return output, glyph


def background(
    rng: np.random.Generator, size: int = INPUT_SIZE
) -> Image.Image:
    first, second = BACKGROUND_PALETTES[
        int(rng.integers(0, len(BACKGROUND_PALETTES)))
    ]
    vertical = rng.random() < 0.5
    coordinate = np.linspace(0.0, 1.0, size, dtype=np.float32)
    if vertical:
        blend = np.repeat(coordinate[:, None], size, axis=1)
    else:
        blend = np.repeat(coordinate[None, :], size, axis=0)
    array = (
        np.asarray(first, dtype=np.float32)[None, None, :] * (1.0 - blend[:, :, None])
        + np.asarray(second, dtype=np.float32)[None, None, :] * blend[:, :, None]
    )
    stripe_period = int(rng.integers(8, 25))
    stripe_strength = float(rng.uniform(-12.0, 12.0))
    yy, xx = np.indices((size, size))
    stripes = ((xx + yy) % stripe_period < max(1, stripe_period // 8))[:, :, None]
    array += stripes * stripe_strength
    array += rng.normal(0.0, rng.uniform(0.0, 5.0), array.shape)
    return Image.fromarray(np.uint8(np.clip(array, 0, 255)), "RGB")


def homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    matrix = []
    values = []
    for (x, y), (u, v) in zip(source, destination):
        matrix.extend(
            (
                [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y],
                [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y],
            )
        )
        values.extend((u, v))
    coefficients = np.linalg.solve(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
    return np.append(coefficients, 1.0).reshape(3, 3)


def transform_corner(
    corner: Image.Image,
    rng: np.random.Generator,
    distractors: list[Image.Image] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    patch = corner.convert("RGBA")
    angle = float(rng.uniform(-180.0, 180.0))
    shear_x = float(rng.uniform(-0.18, 0.18))
    shear_y = float(rng.uniform(-0.12, 0.12))
    canvas_size = 192
    target_height = float(rng.uniform(72.0, 132.0))
    scale_x = target_height / patch.height * float(rng.uniform(0.88, 1.12))
    scale_y = target_height / patch.height
    radians = math.radians(angle)
    rotation = np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ]
    )
    affine = (
        rotation
        @ np.asarray([[1.0, shear_x], [shear_y, 1.0]])
        @ np.asarray([[scale_x, 0.0], [0.0, scale_y]])
    )
    source = np.asarray(
        [
            [0.0, 0.0],
            [patch.width - 1.0, 0.0],
            [patch.width - 1.0, patch.height - 1.0],
            [0.0, patch.height - 1.0],
        ]
    )
    centered = source - np.asarray(
        [(patch.width - 1.0) / 2.0, (patch.height - 1.0) / 2.0]
    )
    destination = centered @ affine.T
    perspective = rng.uniform(-5.0, 5.0, size=(4, 2))
    destination += perspective
    span = np.ptp(destination, axis=0)
    fit_scale = min(1.0, 160.0 / max(float(span.max()), 1.0))
    destination = (destination - destination.mean(axis=0)) * fit_scale
    minimum = destination.min(axis=0)
    maximum = destination.max(axis=0)
    center = np.asarray(
        [
            rng.uniform(12.0 - minimum[0], 180.0 - maximum[0]),
            rng.uniform(12.0 - minimum[1], 180.0 - maximum[1]),
        ]
    )
    destination += center
    forward = homography(source, destination)
    inverse = np.linalg.inv(forward)
    inverse /= inverse[2, 2]
    coefficients = (
        inverse[0, 0],
        inverse[0, 1],
        inverse[0, 2],
        inverse[1, 0],
        inverse[1, 1],
        inverse[1, 2],
        inverse[2, 0],
        inverse[2, 1],
    )
    warped = patch.transform(
        (canvas_size, canvas_size),
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    output = background(rng, canvas_size).convert("RGBA")
    for distractor in distractors or []:
        scale = float(rng.uniform(1.0, 2.2))
        resized = distractor.convert("RGBA").resize(
            (
                max(8, round(distractor.width * scale)),
                max(12, round(distractor.height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        rotated_distractor = resized.rotate(
            float(rng.uniform(-180.0, 180.0)),
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        position = (
            int(rng.integers(-rotated_distractor.width // 2, canvas_size - 10)),
            int(rng.integers(-rotated_distractor.height // 2, canvas_size - 10)),
        )
        output.alpha_composite(rotated_distractor, position)
    output.alpha_composite(warped)
    output = output.convert("RGB")
    brightness = float(rng.uniform(0.58, 1.42))
    contrast = float(rng.uniform(0.62, 1.48))
    saturation = float(rng.uniform(0.45, 1.45))
    output = ImageEnhance.Brightness(output).enhance(brightness)
    output = ImageEnhance.Contrast(output).enhance(contrast)
    output = ImageEnhance.Color(output).enhance(saturation)
    blur = float(rng.uniform(0.0, 1.35))
    if blur > 0.12:
        output = output.filter(ImageFilter.GaussianBlur(blur))
    if rng.random() < 0.06:
        output = output.convert("L").convert("RGB")
    minimum = destination.min(axis=0)
    maximum = destination.max(axis=0)
    padding = max(float((maximum - minimum).max()) * 0.06, 2.0)
    crop_box = (
        max(0, math.floor(minimum[0] - padding)),
        max(0, math.floor(minimum[1] - padding)),
        min(canvas_size, math.ceil(maximum[0] + padding)),
        min(canvas_size, math.ceil(maximum[1] + padding)),
    )
    output = output.crop(crop_box).resize(
        (INPUT_SIZE, INPUT_SIZE),
        Image.Resampling.LANCZOS,
    )
    jpeg_quality = int(rng.integers(42, 96))
    buffer = io.BytesIO()
    output.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=2,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        result = compressed.convert("RGB")
    return result, {
        "rotationDegrees": round(angle, 5),
        "shearX": round(shear_x, 6),
        "shearY": round(shear_y, 6),
        "targetHeight": round(target_height, 5),
        "fitScale": round(fit_scale, 6),
        "destinationQuad": np.round(destination, 5).tolist(),
        "recognizerCropBox": list(crop_box),
        "perspective": np.round(perspective, 5).tolist(),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "saturation": round(saturation, 6),
        "blurRadius": round(blur, 6),
        "jpegQuality": jpeg_quality,
        "clutterCount": len(distractors or []),
    }


def hard_negative(
    cards: dict[str, Image.Image],
    rng: np.random.Generator,
    kind: str,
) -> tuple[Image.Image, str]:
    if kind == "background":
        output = background(rng)
        draw = ImageDraw.Draw(output)
        if rng.random() < 0.7:
            x = int(rng.integers(4, 54))
            y = int(rng.integers(4, 64))
            draw.rounded_rectangle(
                (x, y, x + int(rng.integers(20, 48)), y + int(rng.integers(8, 28))),
                radius=3,
                fill=(int(rng.integers(145, 245)),) * 3,
            )
        return output, "background"

    label = sorted(cards)[int(rng.integers(0, len(cards)))]
    card = cards[label]
    if kind == "center-art":
        left = round(card.width * rng.uniform(0.17, 0.32))
        top = round(card.height * rng.uniform(0.22, 0.42))
        right = min(card.width, left + round(card.width * rng.uniform(0.42, 0.68)))
        bottom = min(card.height, top + round(card.height * rng.uniform(0.35, 0.62)))
        patch = card.crop((left, top, right, bottom))
    elif kind == "card-border":
        left = 0
        top = round(card.height * rng.uniform(0.28, 0.58))
        patch = card.crop(
            (
                left,
                top,
                round(card.width * rng.uniform(0.22, 0.40)),
                min(card.height, top + round(card.height * 0.38)),
            )
        )
    elif kind == "partial-index":
        corner = canonical_corner(
            card,
            "top-left" if rng.random() < 0.5 else "bottom-right",
        )
        if rng.random() < 0.5:
            patch = corner.crop((0, 0, corner.width, round(corner.height * 0.38)))
        else:
            patch = corner.crop(
                (
                    0,
                    round(corner.height * 0.30),
                    corner.width,
                    round(corner.height * 0.72),
                )
            )
    else:
        raise ValueError(f"Unknown negative kind: {kind}")
    output, _ = transform_corner(patch, rng)
    return output, f"{kind}:{label}"


def synthesize(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.role not in {"train", "validation"}:
        raise ValueError("Only Andrew train/validation synthesis is allowed")
    if arguments.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive")
    font_path = resolve_font()
    source_hashes = {
        f"{rank}-{suit}": sha256_file(SOURCE_ROOT / f"{rank}-{suit}.png")
        for rank in RANKS
        for suit in SUITS
    }
    source_cards = {}
    for label in source_hashes:
        with Image.open(SOURCE_ROOT / f"{label}.png") as opened:
            source_cards[label] = opened.convert("RGBA")
    if arguments.output.exists():
        shutil.rmtree(arguments.output)
    crops_directory = arguments.output / "crops"
    crops_directory.mkdir(parents=True)
    manifest = []
    for rank in RANKS:
        for suit in SUITS:
            label = f"{rank}-{suit}"
            source_path = SOURCE_ROOT / f"{label}.png"
            with Image.open(source_path) as opened:
                card = opened.convert("RGBA")
            for index in range(arguments.samples_per_class):
                seed = sample_seed(arguments.seed, label, index)
                rng = np.random.default_rng(seed)
                corner_name = "top-left" if index % 2 == 0 else "bottom-right"
                variant = "shared"
                if rank in FACE_GLYPHS:
                    variant = (
                        "andrew-english",
                        "rendered-english",
                        "rendered-french",
                    )[index % 3]
                corner, glyph = typography_variant(
                    canonical_corner(card, corner_name),
                    rank=rank,
                    suit=suit,
                    variant=variant,
                    font_path=font_path,
                )
                distractors = []
                for _ in range(int(rng.integers(0, 4))):
                    distractor_label = sorted(source_cards)[
                        int(rng.integers(0, len(source_cards)))
                    ]
                    distractor_card = source_cards[distractor_label]
                    distractors.append(
                        canonical_corner(
                            distractor_card,
                            "top-left"
                            if rng.random() < 0.5
                            else "bottom-right",
                        )
                    )
                image, parameters = transform_corner(
                    corner,
                    rng,
                    distractors=distractors,
                )
                name = f"{label}--{index:04d}.jpg"
                path = crops_directory / name
                image.save(path, format="JPEG", quality=95, subsampling=0)
                validate_factorized_label(rank, suit, glyph)
                manifest.append(
                    {
                        "crop": f"crops/{name}",
                        "scene": f"{arguments.role}:{label}:{seed:016x}",
                        "sourceScene": f"andrew:{label}",
                        "annotationId": f"{label}:{index}",
                        "instanceId": f"{arguments.role}:{label}:{index}",
                        "corner": corner_name,
                        "visibleFraction": 1.0,
                        "fullyOnCanvas": True,
                        "rank": rank,
                        "suit": suit,
                        "glyph": glyph,
                        "alphabet": (
                            "french"
                            if variant == "rendered-french"
                            else "english"
                            if rank in FACE_GLYPHS
                            else "shared"
                        ),
                        "typography": variant,
                        "isIndex": True,
                        "orientationClass": int(
                            round(parameters["rotationDegrees"] / 90.0)
                        )
                        % 4,
                        "transformSeed": f"{seed:016x}",
                        "transform": parameters,
                        "sourceAsset": str(source_path.relative_to(PIPELINE_ROOT)),
                        "sourceSha256": source_hashes[label],
                        "sha256": sha256_file(path),
                    }
                )
    negative_kinds = (
        "background",
        "center-art",
        "card-border",
        "partial-index",
    )
    negative_count = (
        arguments.negative_photo_equivalents
        * arguments.negatives_per_photo_equivalent
    )
    for index in range(negative_count):
        kind = negative_kinds[index % len(negative_kinds)]
        seed = sample_seed(arguments.seed, f"negative:{kind}", index)
        rng = np.random.default_rng(seed)
        image, negative_source = hard_negative(source_cards, rng, kind)
        name = f"negative--{index:05d}.jpg"
        path = crops_directory / name
        image.save(path, format="JPEG", quality=95, subsampling=0)
        manifest.append(
            {
                "crop": f"crops/{name}",
                "scene": f"{arguments.role}:negative:{seed:016x}",
                "sourceScene": f"andrew-hard-negative:{negative_source}",
                "annotationId": f"negative:{index}",
                "instanceId": f"{arguments.role}:negative:{index}",
                "corner": "none",
                "visibleFraction": 1.0,
                "fullyOnCanvas": True,
                "rank": "7",
                "suit": "clubs",
                "glyph": "7",
                "alphabet": "negative",
                "typography": "none",
                "isIndex": False,
                "orientationClass": -1,
                "negativeKind": kind,
                "transformSeed": f"{seed:016x}",
                "sourceAsset": negative_source,
                "sha256": sha256_file(path),
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
    aggregate_source_hash = sha256_bytes(
        "".join(f"{key}:{source_hashes[key]}\n" for key in sorted(source_hashes)).encode()
    )
    summary = {
        "schemaVersion": 2,
        "role": arguments.role,
        "inputSize": INPUT_SIZE,
        "paddingFraction": 0.06,
        "cropCount": len(manifest),
        "sceneCount": len(manifest),
        "datasetSeed": arguments.seed,
        "samplesPerClass": arguments.samples_per_class,
        "positiveCount": len(RANKS) * len(SUITS) * arguments.samples_per_class,
        "hardNegativeCount": negative_count,
        "negativePhotoEquivalentCount": arguments.negative_photo_equivalents,
        "negativesPerPhotoEquivalent": arguments.negatives_per_photo_equivalent,
        "balancedClassCount": len(RANKS) * len(SUITS),
        "source": {
            "id": "andrew-tidey-cards-pack",
            "role": "training",
            "license": "CC0-1.0",
            "assetPath": str(SOURCE_ROOT.relative_to(PIPELINE_ROOT)),
            "assetsAggregateSha256": aggregate_source_hash,
        },
        "generator": {
            "id": "balanced-andrew-detector-crop-synthesis-v4",
            "cropContract": "axis-aligned transformed index bbox plus 0.06 padding",
            "bothCorners": True,
            "rotationDegrees": [-180, 180],
            "perspectivePixels": [-5, 5],
            "shearX": [-0.18, 0.18],
            "shearY": [-0.12, 0.12],
            "targetHeight": [72, 132],
            "blurRadius": [0, 1.35],
            "jpegQuality": [42, 95],
            "brightness": [0.58, 1.42],
            "contrast": [0.62, 1.48],
            "saturation": [0.45, 1.45],
            "backgrounds": len(BACKGROUND_PALETTES),
            "foreignIndexClutterCount": [0, 3],
            "faceTypography": [
                "andrew-english",
                "rendered-english",
                "rendered-french",
            ],
            "hardNegativeKinds": list(negative_kinds),
            "font": {
                "path": str(font_path),
                "sha256": sha256_file(font_path),
            },
        },
        "manifestSha256": sha256_file(manifest_path),
    }
    (arguments.output / "summary.json").write_text(
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--samples-per-class", type=int, required=True)
    parser.add_argument(
        "--negative-photo-equivalents", type=int, default=320
    )
    parser.add_argument(
        "--negatives-per-photo-equivalent", type=int, default=10
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    return arguments


if __name__ == "__main__":
    print(json.dumps(synthesize(parse_arguments()), indent=2, sort_keys=True))
