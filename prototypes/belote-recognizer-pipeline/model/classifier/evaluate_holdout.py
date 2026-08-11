from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from labels import RANKS, RANK_TO_INDEX, SUITS, SUIT_TO_INDEX, validate_factorized_label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(directory: Path) -> list[dict[str, Any]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        summary["role"] != "holdout"
        or summary["source"]["id"] != "greywyvern-cardset"
        or summary["source"]["role"] != "holdout-only"
    ):
        raise ValueError("Holdout source firewall rejected dataset")
    rows = [
        json.loads(line)
        for line in (directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_factorized_label(row["rank"], row["suit"], row["glyph"])
    return rows


def load_batch(directory: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    arrays = []
    for row in rows:
        with Image.open(directory / row["crop"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        arrays.append((array.transpose(2, 0, 1) - 0.5) / 0.5)
    return np.stack(arrays).astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def per_class(
    targets: np.ndarray, predictions: np.ndarray, classes: tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for index, label in enumerate(classes):
        selected = targets == index
        support = int(selected.sum())
        correct = int(((predictions == index) & selected).sum())
        result[label] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else 0.0,
        }
    return result


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.output.exists():
        raise FileExistsError(
            f"Refusing to repeat final holdout evaluation: {arguments.output}"
        )
    rows = read_rows(arguments.holdout_crops)
    session = ort.InferenceSession(
        str(arguments.model), providers=["CPUExecutionProvider"]
    )
    rank_logits = []
    suit_logits = []
    index_logits = []
    card_logits = []
    for offset in range(0, len(rows), arguments.batch_size):
        batch_rows = rows[offset : offset + arguments.batch_size]
        outputs = session.run(
            [
                "rank_logits",
                "suit_logits",
                "index_logit",
                "card_logits",
            ],
            {"input": load_batch(arguments.holdout_crops, batch_rows)},
        )
        rank_logits.append(outputs[0])
        suit_logits.append(outputs[1])
        index_logits.append(outputs[2])
        card_logits.append(outputs[3])
    all_rank_logits = np.concatenate(rank_logits)
    all_suit_logits = np.concatenate(suit_logits)
    all_index_logits = np.concatenate(index_logits)
    all_card_logits = np.concatenate(card_logits)
    card_predictions = all_card_logits.argmax(axis=1)
    rank_predictions = card_predictions // len(SUITS)
    suit_predictions = card_predictions % len(SUITS)
    rank_targets = np.asarray([RANK_TO_INDEX[row["rank"]] for row in rows])
    suit_targets = np.asarray([SUIT_TO_INDEX[row["suit"]] for row in rows])
    rank_probabilities = softmax(all_rank_logits)
    suit_probabilities = softmax(all_suit_logits)
    index_probabilities = 1.0 / (1.0 + np.exp(-all_index_logits))
    card_correct = (rank_predictions == rank_targets) & (
        suit_predictions == suit_targets
    )
    metrics = {
        "count": len(rows),
        "rankAccuracy": float((rank_predictions == rank_targets).mean()),
        "suitAccuracy": float((suit_predictions == suit_targets).mean()),
        "cardAccuracy": float(card_correct.mean()),
        "meanRankConfidence": float(rank_probabilities.max(axis=1).mean()),
        "meanSuitConfidence": float(suit_probabilities.max(axis=1).mean()),
        "indexRecall": float(
            (index_probabilities >= arguments.index_threshold).mean()
        ),
        "meanIndexConfidence": float(index_probabilities.mean()),
        "indexThreshold": arguments.index_threshold,
        "rankPerClass": per_class(rank_targets, rank_predictions, RANKS),
        "suitPerClass": per_class(suit_targets, suit_predictions, SUITS),
    }
    result = {
        "schemaVersion": 1,
        "evaluation": "single-final-holdout",
        "source": "greywyvern-cardset",
        "model": {
            "bytes": arguments.model.stat().st_size,
            "sha256": sha256_file(arguments.model),
        },
        "holdoutManifestSha256": sha256_file(
            arguments.holdout_crops / "manifest.jsonl"
        ),
        "metrics": metrics,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(arguments.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--holdout-crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--index-threshold", type=float, default=0.5)
    arguments = parser.parse_args()
    arguments.model = arguments.model.resolve()
    arguments.holdout_crops = arguments.holdout_crops.resolve()
    arguments.output = arguments.output.resolve()
    return arguments


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_arguments()), indent=2, sort_keys=True))
