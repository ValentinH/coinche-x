from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from labels import RANKS, RANK_TO_INDEX, SUITS, SUIT_TO_INDEX, validate_factorized_label
from model import INPUT_SIZE, TinyFactorizedClassifier, parameter_count

SEED = 20260730


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(directory: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_factorized_label(row["rank"], row["suit"], row["glyph"])
    if not rows:
        raise ValueError(f"Empty crop manifest: {directory}")
    return rows


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).sub_(0.5).div_(0.5)


def augment(image: Image.Image) -> Image.Image:
    image = image.rotate(
        random.uniform(-4.0, 4.0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(24, 28, 32),
    )
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.12))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.90, 1.15))
    if random.random() < 0.02:
        image = ImageOps.grayscale(image).convert("RGB")
    return image


class CropDataset(Dataset[tuple[torch.Tensor, int, int, float, int]]):
    def __init__(
        self,
        directory: Path,
        *,
        training: bool,
        clean_only: bool = False,
    ) -> None:
        self.directory = directory
        self.summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        self.rows = read_manifest(directory)
        if clean_only:
            self.rows = [
                row
                for row in self.rows
                if row.get("visibleFraction", 0.0) >= 0.999
                and row.get("fullyOnCanvas", False)
            ]
        if not self.rows:
            raise ValueError(f"No eligible crops: {directory}")
        self.training = training
        self.images = []
        for row in self.rows:
            image_path = self.directory / row["crop"]
            if sha256_file(image_path) != row["sha256"]:
                raise ValueError(f"Crop integrity check failed: {image_path}")
            with Image.open(image_path) as source:
                self.images.append(source.convert("RGB"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, int, int, float, int]:
        row = self.rows[index]
        image = self.images[index].copy()
        if self.training:
            image = augment(image)
        return (
            image_to_tensor(image),
            RANK_TO_INDEX[row["rank"]],
            SUIT_TO_INDEX[row["suit"]],
            float(row.get("isIndex", True)),
            int(row.get("orientationClass", -1)),
        )


def configure_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def metrics_from_predictions(
    rank_targets: list[int],
    rank_predictions: list[int],
    suit_targets: list[int],
    suit_predictions: list[int],
) -> dict[str, Any]:
    count = len(rank_targets)
    rank_correct = sum(a == b for a, b in zip(rank_targets, rank_predictions))
    suit_correct = sum(a == b for a, b in zip(suit_targets, suit_predictions))
    card_correct = sum(
        actual_rank == predicted_rank and actual_suit == predicted_suit
        for actual_rank, predicted_rank, actual_suit, predicted_suit in zip(
            rank_targets,
            rank_predictions,
            suit_targets,
            suit_predictions,
        )
    )

    def per_class(
        targets: list[int], predictions: list[int], classes: tuple[str, ...]
    ) -> dict[str, dict[str, float | int]]:
        result = {}
        for class_index, label in enumerate(classes):
            support = sum(target == class_index for target in targets)
            correct = sum(
                target == class_index and prediction == class_index
                for target, prediction in zip(targets, predictions)
            )
            result[label] = {
                "support": support,
                "correct": correct,
                "recall": correct / support if support else 0.0,
            }
        return result

    return {
        "count": count,
        "rankAccuracy": rank_correct / count,
        "suitAccuracy": suit_correct / count,
        "cardAccuracy": card_correct / count,
        "rankPerClass": per_class(rank_targets, rank_predictions, RANKS),
        "suitPerClass": per_class(suit_targets, suit_predictions, SUITS),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, int, float, int]],
    *,
    index_threshold: float,
) -> dict[str, Any]:
    model.eval()
    rank_targets: list[int] = []
    rank_predictions: list[int] = []
    suit_targets: list[int] = []
    suit_predictions: list[int] = []
    factorized_rank_predictions: list[int] = []
    factorized_suit_predictions: list[int] = []
    index_targets: list[bool] = []
    index_scores: list[float] = []
    orientation_correct = 0
    orientation_count = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    validity_criterion = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for images, ranks, suits, is_index, orientations in loader:
            (
                rank_logits,
                suit_logits,
                index_logits,
                card_logits,
                orientation_logits,
            ) = model(images)
            positive = is_index.bool()
            loss = validity_criterion(index_logits, is_index)
            if positive.any():
                loss += criterion(rank_logits[positive], ranks[positive])
                loss += criterion(suit_logits[positive], suits[positive])
                loss += criterion(
                    card_logits[positive],
                    ranks[positive] * len(SUITS) + suits[positive],
                )
            total_loss += loss.item() * images.shape[0]
            rank_targets.extend(ranks[positive].tolist())
            suit_targets.extend(suits[positive].tolist())
            card_predictions = card_logits[positive].argmax(dim=1)
            rank_predictions.extend((card_predictions // len(SUITS)).tolist())
            suit_predictions.extend((card_predictions % len(SUITS)).tolist())
            factorized_rank_predictions.extend(
                rank_logits[positive].argmax(dim=1).tolist()
            )
            factorized_suit_predictions.extend(
                suit_logits[positive].argmax(dim=1).tolist()
            )
            index_targets.extend(positive.tolist())
            index_scores.extend(torch.sigmoid(index_logits).tolist())
            known_orientation = orientations >= 0
            orientation_count += int(known_orientation.sum())
            orientation_correct += int(
                (
                    orientation_logits[known_orientation].argmax(dim=1)
                    == orientations[known_orientation]
                ).sum()
            )
    metrics = metrics_from_predictions(
        rank_targets, rank_predictions, suit_targets, suit_predictions
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["factorizedAuxiliary"] = metrics_from_predictions(
        rank_targets,
        factorized_rank_predictions,
        suit_targets,
        factorized_suit_predictions,
    )
    metrics["orientationAccuracy"] = (
        orientation_correct / orientation_count if orientation_count else None
    )
    accepted = [score >= index_threshold for score in index_scores]
    positive_count = sum(index_targets)
    negative_count = len(index_targets) - positive_count
    true_accepts = sum(
        target and prediction
        for target, prediction in zip(index_targets, accepted)
    )
    false_accepts = sum(
        not target and prediction
        for target, prediction in zip(index_targets, accepted)
    )
    dataset = loader.dataset
    assert isinstance(dataset, CropDataset)
    photo_equivalents = int(
        dataset.summary.get("negativePhotoEquivalentCount", 0)
    )
    metrics["index"] = {
        "threshold": index_threshold,
        "positiveCount": positive_count,
        "negativeCount": negative_count,
        "recall": true_accepts / positive_count if positive_count else 0.0,
        "falseAcceptRate": false_accepts / negative_count if negative_count else 0.0,
        "falseAcceptCount": false_accepts,
        "negativePhotoEquivalentCount": photo_equivalents,
        "falseAcceptsPerPhotoEquivalent": (
            false_accepts / photo_equivalents if photo_equivalents else 0.0
        ),
    }
    return metrics


def calibrate_index_threshold(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, int, float, int]],
    *,
    minimum_recall: float,
) -> float:
    model.eval()
    positive_scores = []
    with torch.inference_mode():
        for images, _ranks, _suits, is_index, _orientations in loader:
            (
                _rank_logits,
                _suit_logits,
                index_logits,
                _card_logits,
                _orientation_logits,
            ) = model(images)
            scores = torch.sigmoid(index_logits)
            positive_scores.extend(scores[is_index.bool()].tolist())
    if not positive_scores:
        raise ValueError("Index threshold calibration requires positives")
    allowed_misses = math.floor(
        len(positive_scores) * (1.0 - minimum_recall) + 1e-9
    )
    ordered = sorted(positive_scores)
    return float(ordered[min(allowed_misses, len(ordered) - 1)])


def assert_scene_level_separation(train_directory: Path, validation_directory: Path) -> None:
    train_summary = json.loads(
        (train_directory / "summary.json").read_text(encoding="utf-8")
    )
    validation_summary = json.loads(
        (validation_directory / "summary.json").read_text(encoding="utf-8")
    )
    if train_summary["role"] != "train" or validation_summary["role"] != "validation":
        raise ValueError("Expected train and validation crop roles")
    if train_summary["source"]["id"] != "andrew-tidey-cards-pack":
        raise ValueError("Training source firewall rejected non-Andrew data")
    if validation_summary["source"]["id"] != "andrew-tidey-cards-pack":
        raise ValueError("Validation source firewall rejected non-Andrew data")
    if train_summary["datasetSeed"] == validation_summary["datasetSeed"]:
        raise ValueError("Train and validation scene seeds must differ")
    if train_summary.get("samplesPerClass", 0) < 160:
        raise ValueError("Training requires at least 160 transforms per card class")
    if train_summary.get("balancedClassCount") != 32:
        raise ValueError("Training synthesis must cover exactly 32 balanced classes")
    if validation_summary.get("balancedClassCount") != 32:
        raise ValueError("Validation synthesis must cover exactly 32 balanced classes")
    train_scenes = {row["scene"] for row in read_manifest(train_directory)}
    validation_scenes = {row["scene"] for row in read_manifest(validation_directory)}
    if train_scenes & validation_scenes:
        raise ValueError("Train/validation scene leakage")
    train_seeds = {
        row["transformSeed"]
        for row in read_manifest(train_directory)
        if row.get("isIndex", True)
    }
    validation_seeds = {
        row["transformSeed"]
        for row in read_manifest(validation_directory)
        if row.get("isIndex", True)
    }
    if train_seeds & validation_seeds:
        raise ValueError("Train/validation transform seed leakage")


def assert_additional_training_sources(directories: list[Path]) -> None:
    allowed = {
        ("train", "andrew-tidey-cards-pack", "training"),
        ("train", "kaggle-gpiosenka-cards-v2", "training"),
    }
    for directory in directories:
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        source = summary["source"]
        identity = (
            summary["role"],
            source["id"],
            source["role"],
        )
        if identity not in allowed:
            raise ValueError(
                f"Additional training source firewall rejected {identity}"
            )


def export_onnx(model: nn.Module, output_path: Path) -> None:
    model.eval()
    with tempfile.TemporaryDirectory(prefix="belote-classifier-export-") as directory:
        temporary_path = Path(directory) / "classifier.onnx"
        torch.onnx.export(
            model,
            torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),
            temporary_path,
            input_names=["input"],
            output_names=[
                "rank_logits",
                "suit_logits",
                "index_logit",
                "card_logits",
                "orientation_logits",
            ],
            dynamic_axes={
                "input": {0: "batch"},
                "rank_logits": {0: "batch"},
                "suit_logits": {0: "batch"},
                "index_logit": {0: "batch"},
                "card_logits": {0: "batch"},
                "orientation_logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        exported = onnx.load(temporary_path)
        onnx.checker.check_model(exported)
        shutil.copyfile(temporary_path, output_path)


def verify_onnx_export(model: nn.Module, output_path: Path) -> float:
    values = torch.linspace(
        -1.0, 1.0, steps=2 * 3 * INPUT_SIZE * INPUT_SIZE
    ).reshape(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        expected = [output.numpy() for output in model(values)]
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        [
            "rank_logits",
            "suit_logits",
            "index_logit",
            "card_logits",
            "orientation_logits",
        ],
        {"input": values.numpy()},
    )
    maximum_error = max(
        float(np.max(np.abs(expected_output - actual_output)))
        for expected_output, actual_output in zip(expected, actual)
    )
    if maximum_error > 1e-4:
        raise ValueError(f"ONNX parity error too large: {maximum_error}")
    return maximum_error


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    configure_reproducibility(arguments.seed)
    assert_scene_level_separation(arguments.train_crops, arguments.validation_crops)
    assert_additional_training_sources(arguments.additional_train_crops)
    arguments.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output / "training-checkpoint.pt"
    train_sources = [
        CropDataset(
            arguments.train_crops,
            training=arguments.online_augment,
            clean_only=True,
        ),
        *[
            CropDataset(
                directory,
                training=arguments.online_augment,
                clean_only=False,
            )
            for directory in arguments.additional_train_crops
        ],
    ]
    train_dataset = (
        train_sources[0]
        if len(train_sources) == 1
        else ConcatDataset(train_sources)
    )
    validation_clean_dataset = CropDataset(
        arguments.validation_crops, training=False, clean_only=True
    )
    validation_all_dataset = CropDataset(
        arguments.validation_crops, training=False, clean_only=False
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_clean_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_all_loader = DataLoader(
        validation_all_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    scene_validation_clean_loader = None
    scene_validation_all_loader = None
    if arguments.scene_validation_crops is not None:
        scene_validation_clean_loader = DataLoader(
            CropDataset(
                arguments.scene_validation_crops,
                training=False,
                clean_only=True,
            ),
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=0,
        )
        scene_validation_all_loader = DataLoader(
            CropDataset(
                arguments.scene_validation_crops,
                training=False,
                clean_only=False,
            ),
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=0,
        )

    model = TinyFactorizedClassifier()
    if arguments.initial_checkpoint is not None:
        checkpoint = torch.load(
            arguments.initial_checkpoint,
            map_location="cpu",
        )
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = {
            "card_head.weight",
            "card_head.bias",
            "orientation_head.weight",
            "orientation_head.bias",
        }
        if set(incompatible.missing_keys) - allowed_missing:
            raise ValueError(
                f"Checkpoint misses unsupported keys: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                f"Checkpoint has unexpected keys: {incompatible.unexpected_keys}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    validity_criterion = nn.BCEWithLogitsLoss()
    best_score = (-math.inf, -math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history = []
    if arguments.initial_checkpoint is not None:
        initial_metrics = evaluate(
            model,
            validation_loader,
            index_threshold=arguments.index_threshold,
        )
        best_score = (
            initial_metrics["cardAccuracy"],
            initial_metrics["index"]["recall"],
            -initial_metrics["index"]["falseAcceptsPerPhotoEquivalent"],
        )
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        torch.save(
            {
                "epoch": 0,
                "score": best_score,
                "model": best_state,
            },
            checkpoint_path,
        )

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, ranks, suits, is_index, orientations in train_loader:
            optimizer.zero_grad(set_to_none=True)
            (
                rank_logits,
                suit_logits,
                index_logits,
                card_logits,
                orientation_logits,
            ) = model(images)
            positive = is_index.bool()
            loss = arguments.index_loss_weight * validity_criterion(
                index_logits, is_index
            )
            if positive.any():
                card_targets = ranks[positive] * len(SUITS) + suits[positive]
                loss += criterion(card_logits[positive], card_targets)
                loss += 0.4 * criterion(
                    rank_logits[positive], ranks[positive]
                )
                loss += 0.5 * criterion(
                    suit_logits[positive], suits[positive]
                )
            known_orientation = orientations >= 0
            if known_orientation.any():
                loss += arguments.orientation_loss_weight * criterion(
                    orientation_logits[known_orientation],
                    orientations[known_orientation],
                )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]
        scheduler.step()
        validation_metrics = evaluate(
            model,
            validation_loader,
            index_threshold=arguments.index_threshold,
        )
        score = (
            validation_metrics["cardAccuracy"],
            validation_metrics["index"]["recall"],
            -validation_metrics["index"]["falseAcceptsPerPhotoEquivalent"],
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "epoch": epoch,
                    "score": score,
                    "model": best_state,
                },
                checkpoint_path,
            )
        epoch_metrics = {
            "epoch": epoch,
            "learningRate": scheduler.get_last_lr()[0],
            "trainLoss": train_loss / len(train_dataset),
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} train_loss={epoch_metrics['trainLoss']:.4f} "
            f"val_rank={validation_metrics['rankAccuracy']:.4f} "
            f"val_suit={validation_metrics['suitAccuracy']:.4f} "
            f"val_card={validation_metrics['cardAccuracy']:.4f} "
            f"val_index={validation_metrics['index']['recall']:.4f} "
            f"val_fa/photo={validation_metrics['index']['falseAcceptsPerPhotoEquivalent']:.4f}",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    arguments.index_threshold = calibrate_index_threshold(
        model,
        validation_loader,
        minimum_recall=arguments.minimum_index_recall,
    )
    validation_metrics = evaluate(
        model,
        validation_loader,
        index_threshold=arguments.index_threshold,
    )
    validation_all_metrics = evaluate(
        model,
        validation_all_loader,
        index_threshold=arguments.index_threshold,
    )
    scene_validation_clean_metrics = (
        evaluate(
            model,
            scene_validation_clean_loader,
            index_threshold=arguments.index_threshold,
        )
        if scene_validation_clean_loader is not None
        else None
    )
    scene_validation_all_metrics = (
        evaluate(
            model,
            scene_validation_all_loader,
            index_threshold=arguments.index_threshold,
        )
        if scene_validation_all_loader is not None
        else None
    )

    validation_result = {
        "schemaVersion": 1,
        "split": "validation",
        "source": "andrew-tidey-cards-pack",
        "cleanDefinition": "visibleFraction >= 0.999 and corner polygon fully on canvas",
        "metrics": {
            "balancedClean": validation_metrics,
            "balancedAll": validation_all_metrics,
            "andrewSceneClean": scene_validation_clean_metrics,
            "andrewSceneAll": scene_validation_all_metrics,
        },
        "gates": {
            "minimumBalancedCardAccuracy": arguments.minimum_card_accuracy,
            "minimumPositiveIndexRecall": arguments.minimum_index_recall,
            "maximumFalseAcceptsPerPhotoEquivalent": arguments.maximum_false_accepts_per_photo,
        },
    }
    (arguments.output / "validation-metrics.json").write_text(
        f"{json.dumps(validation_result, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    (arguments.output / "train-history.json").write_text(
        f"{json.dumps(history, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    gate_failures = []
    if validation_metrics["cardAccuracy"] < arguments.minimum_card_accuracy:
        gate_failures.append(
            f"card accuracy {validation_metrics['cardAccuracy']:.6f}"
        )
    if validation_metrics["index"]["recall"] < arguments.minimum_index_recall:
        gate_failures.append(
            f"index recall {validation_metrics['index']['recall']:.6f}"
        )
    if (
        validation_metrics["index"]["falseAcceptsPerPhotoEquivalent"]
        > arguments.maximum_false_accepts_per_photo
    ):
        gate_failures.append(
            "false accepts/photo "
            f"{validation_metrics['index']['falseAcceptsPerPhotoEquivalent']:.6f}"
        )
    if gate_failures:
        raise ValueError("Validation freeze gate failed: " + ", ".join(gate_failures))

    onnx_path = arguments.output / "classifier.onnx"
    export_onnx(model, onnx_path)
    onnx_maximum_error = verify_onnx_export(model, onnx_path)
    if onnx_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Classifier exceeds 8 MiB hard limit")
    metadata = {
        "schemaVersion": 1,
        "architecture": "c4-best-view-joint-card-validity-v4",
        "input": {"name": "input", "shape": ["batch", 3, INPUT_SIZE, INPUT_SIZE]},
        "outputs": {
            "rank": {"name": "rank_logits", "classes": list(RANKS)},
            "suit": {"name": "suit_logits", "classes": list(SUITS)},
            "index": {
                "name": "index_logit",
                "activation": "sigmoid",
                "threshold": arguments.index_threshold,
            },
            "card": {
                "name": "card_logits",
                "classes": [
                    f"{rank}-{suit}" for rank in RANKS for suit in SUITS
                ],
                "selection": True,
            },
            "orientation": {
                "name": "orientation_logits",
                "classes": [0, 90, 180, 270],
            },
        },
        "parameterCount": parameter_count(model),
        "model": {
            "path": "classifier.onnx",
            "bytes": onnx_path.stat().st_size,
            "sha256": sha256_file(onnx_path),
            "opset": 17,
            "pytorchOnnxMaximumAbsoluteError": onnx_maximum_error,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "training": {
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "bestEpoch": best_epoch,
            "batchSize": arguments.batch_size,
            "learningRate": arguments.learning_rate,
            "weightDecay": arguments.weight_decay,
            "initialCheckpointSha256": (
                sha256_file(arguments.initial_checkpoint)
                if arguments.initial_checkpoint is not None
                else None
            ),
            "additionalTrainManifestSha256": [
                sha256_file(directory / "manifest.jsonl")
                for directory in arguments.additional_train_crops
            ],
            "trainManifestSha256": sha256_file(
                arguments.train_crops / "manifest.jsonl"
            ),
            "validationManifestSha256": sha256_file(
                arguments.validation_crops / "manifest.jsonl"
            ),
            "selectionMetric": "balanced Andrew card accuracy, index recall, then negative false accepts",
            "freezeGates": validation_result["gates"],
            "holdoutUsed": False,
        },
    }
    (arguments.output / "metadata.json").write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    checkpoint_path.unlink(missing_ok=True)
    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-crops", type=Path, required=True)
    parser.add_argument("--validation-crops", type=Path, required=True)
    parser.add_argument("--scene-validation-crops", type=Path)
    parser.add_argument(
        "--additional-train-crops",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--index-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.25)
    parser.add_argument("--online-augment", action="store_true")
    parser.add_argument("--index-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-card-accuracy", type=float, default=0.995)
    parser.add_argument("--minimum-index-recall", type=float, default=0.995)
    parser.add_argument(
        "--maximum-false-accepts-per-photo", type=float, default=0.1
    )
    arguments = parser.parse_args()
    arguments.train_crops = arguments.train_crops.resolve()
    arguments.validation_crops = arguments.validation_crops.resolve()
    arguments.additional_train_crops = [
        path.resolve() for path in arguments.additional_train_crops
    ]
    if arguments.scene_validation_crops is not None:
        arguments.scene_validation_crops = (
            arguments.scene_validation_crops.resolve()
        )
    arguments.output = arguments.output.resolve()
    if arguments.initial_checkpoint is not None:
        arguments.initial_checkpoint = arguments.initial_checkpoint.resolve()
    return arguments


if __name__ == "__main__":
    print(json.dumps(train(parse_arguments()), indent=2, sort_keys=True))
