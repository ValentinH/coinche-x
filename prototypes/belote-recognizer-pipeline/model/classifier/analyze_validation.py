from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import RANKS, SUITS  # noqa: E402
from model import TinyFactorizedClassifier  # noqa: E402
from train import CropDataset, configure_reproducibility  # noqa: E402


def accuracy(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    return {
        "count": count,
        "rankAccuracy": sum(row["rankCorrect"] for row in rows) / count,
        "suitAccuracy": sum(row["suitCorrect"] for row in rows) / count,
        "cardAccuracy": sum(row["cardCorrect"] for row in rows) / count,
    }


def bucket(row: dict[str, Any], dimension: str) -> str:
    transform = row["transform"]
    if dimension == "blur":
        value = transform["blurRadius"]
        return "low" if value < 0.45 else "medium" if value < 0.90 else "high"
    if dimension == "rotationResidual":
        value = abs((transform["rotationDegrees"] + 45.0) % 90.0 - 45.0)
        return "0-15" if value < 15 else "15-30" if value < 30 else "30-45"
    if dimension == "jpeg":
        value = transform["jpegQuality"]
        return "42-59" if value < 60 else "60-79" if value < 80 else "80-95"
    if dimension == "fitScale":
        value = transform["fitScale"]
        return "<0.80" if value < 0.80 else "0.80-0.94" if value < 0.95 else "0.95-1"
    return str(row[dimension])


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    configure_reproducibility(20260730)
    dataset = CropDataset(arguments.crops, training=False, clean_only=True)
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = TinyFactorizedClassifier().eval()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    predictions = []
    offset = 0
    with torch.inference_mode():
        for images, rank_targets, suit_targets, is_index, _orientations in loader:
            (
                _rank_logits,
                _suit_logits,
                index_logits,
                card_logits,
                _orientation_logits,
            ) = model(images)
            card_predictions = card_logits.argmax(dim=1)
            rank_predictions = card_predictions // len(SUITS)
            suit_predictions = card_predictions % len(SUITS)
            for index in range(images.shape[0]):
                source = dataset.rows[offset + index]
                if not bool(is_index[index]):
                    continue
                actual_rank = RANKS[int(rank_targets[index])]
                actual_suit = SUITS[int(suit_targets[index])]
                predicted_rank = RANKS[int(rank_predictions[index])]
                predicted_suit = SUITS[int(suit_predictions[index])]
                predictions.append(
                    {
                        **source,
                        "predictedRank": predicted_rank,
                        "predictedSuit": predicted_suit,
                        "rankCorrect": actual_rank == predicted_rank,
                        "suitCorrect": actual_suit == predicted_suit,
                        "cardCorrect": actual_rank == predicted_rank
                        and actual_suit == predicted_suit,
                        "indexConfidence": float(
                            torch.sigmoid(index_logits[index])
                        ),
                    }
                )
            offset += images.shape[0]

    rank_confusion = {
        actual: {predicted: 0 for predicted in RANKS} for actual in RANKS
    }
    suit_confusion = {
        actual: {predicted: 0 for predicted in SUITS} for actual in SUITS
    }
    for row in predictions:
        rank_confusion[row["rank"]][row["predictedRank"]] += 1
        suit_confusion[row["suit"]][row["predictedSuit"]] += 1
    stratified = {}
    for dimension in (
        "blur",
        "rotationResidual",
        "jpeg",
        "fitScale",
        "corner",
        "typography",
    ):
        groups = defaultdict(list)
        for row in predictions:
            groups[bucket(row, dimension)].append(row)
        stratified[dimension] = {
            name: accuracy(rows) for name, rows in sorted(groups.items())
        }

    failures = [row for row in predictions if not row["cardCorrect"]]
    tile_width = 150
    tile_height = 126
    columns = 8
    shown = failures[:64]
    rows = max(1, math.ceil(len(shown) / columns))
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(montage)
    for index, failure in enumerate(shown):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        with Image.open(arguments.crops / failure["crop"]) as image:
            montage.paste(image.convert("RGB"), (x, y))
        draw.text(
            (x + 2, y + 98),
            f"{failure['rank']}-{failure['suit'][:1]} -> "
            f"{failure['predictedRank']}-{failure['predictedSuit'][:1]}",
            fill="black",
        )
        draw.text(
            (x + 2, y + 111),
            f"r={failure['transform']['rotationDegrees']:.0f} "
            f"b={failure['transform']['blurRadius']:.1f}",
            fill="#444444",
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    montage.save(arguments.output / "failures.jpg", quality=92)
    result = {
        "schemaVersion": 1,
        "checkpointEpoch": checkpoint["epoch"],
        "metrics": accuracy(predictions),
        "rankConfusion": rank_confusion,
        "suitConfusion": suit_confusion,
        "stratified": stratified,
        "failureCount": len(failures),
        "failureMontage": "failures.jpg",
    }
    (arguments.output / "analysis.json").write_text(
        f"{json.dumps(result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    arguments = parser.parse_args()
    arguments.checkpoint = arguments.checkpoint.resolve()
    arguments.crops = arguments.crops.resolve()
    arguments.output = arguments.output.resolve()
    return arguments


if __name__ == "__main__":
    print(json.dumps(analyze(parse_arguments()), indent=2, sort_keys=True))
