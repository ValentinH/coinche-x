#!/usr/bin/env python3
"""Train, validate, export, and freeze the tiny detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CombinedDetectorTiles, DetectorTiles, collate
from decode import decode_centers, summarize
from model import (
    INPUT_SIZE,
    MAX_CANDIDATES,
    TinyCornerDetector,
    ExportDetector,
)


ROOT = Path(__file__).resolve().parent
SEED = 20260730


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--extra-train", type=Path, action="append", default=[])
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_deterministic():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def loss_for(outputs, targets):
    heatmap_logits, regression = outputs
    heatmaps = torch.stack([target["heatmap"] for target in targets]).to(
        heatmap_logits.device
    )
    probabilities = heatmap_logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positives = heatmaps.eq(1)
    negatives = heatmaps.lt(1)
    positive_loss = (
        torch.log(probabilities)
        * (1 - probabilities).pow(2)
        * positives
    ).sum()
    negative_loss = (
        torch.log(1 - probabilities)
        * probabilities.pow(2)
        * (1 - heatmaps).pow(4)
        * negatives
    ).sum()
    positive_count = positives.sum().clamp(min=1)
    focal = -(positive_loss + negative_loss) / positive_count
    active = torch.stack([target["active"] for target in targets]).to(
        heatmap_logits.device
    )
    offsets = torch.stack([target["offsets"] for target in targets]).to(
        heatmap_logits.device
    )
    sizes = torch.stack([target["sizes"] for target in targets]).to(
        heatmap_logits.device
    )
    active_channels = active[:, None]
    offset_loss = (
        (regression[:, :2].sigmoid() - offsets).abs()
        * active_channels
    ).sum() / positive_count
    size_loss = (
        (regression[:, 2:].sigmoid() * 0.5 - sizes).abs()
        * active_channels
    ).sum() / positive_count
    return focal + offset_loss + 5 * size_loss


def photometric_augmentation(images, epoch: int, batch_index: int):
    generator = torch.Generator().manual_seed(
        SEED + epoch * 10_007 + batch_index
    )
    shape = (len(images), 1, 1, 1)
    brightness = torch.empty(shape).uniform_(0.75, 1.25, generator=generator)
    contrast = torch.empty(shape).uniform_(0.75, 1.35, generator=generator)
    saturation = torch.empty(shape).uniform_(0.6, 1.3, generator=generator)
    brightness = brightness.to(images.device)
    contrast = contrast.to(images.device)
    saturation = saturation.to(images.device)
    spatial_mean = images.mean((2, 3), keepdim=True)
    images = (images - spatial_mean) * contrast + spatial_mean
    grayscale = images.mean(1, keepdim=True)
    images = (images - grayscale) * saturation + grayscale
    return (images * brightness).clamp(0, 1)


@torch.inference_mode()
def validation_loss(model, loader, device):
    model.eval()
    total = 0.0
    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32).div_(255)
        loss = loss_for(model(images), targets)
        total += loss.item() * len(images)
    return total / len(loader.dataset)


@torch.inference_mode()
def validation_predictions(model, loader, device):
    model.eval()
    output = []
    latencies = []
    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32).div_(255)
        started = time.perf_counter()
        heatmap_logits, regression = model(images)
        probabilities = heatmap_logits.sigmoid().cpu().numpy()[:, 0]
        regression = regression.cpu().numpy()
        latencies.append((time.perf_counter() - started) * 1_000 / len(images))
        output.extend(zip(targets, probabilities, regression))
    return output, latencies


def tune_decoder(predictions, frames):
    choices = []
    for threshold in (
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.90,
        0.95,
    ):
        decoded = [
            (
                target,
                decode_centers(
                    probabilities, regression, threshold=threshold
                ),
            )
            for target, probabilities, regression in predictions
        ]
        metrics = summarize(decoded, frames)
        choices.append((threshold, metrics))
    eligible = [
        choice
        for choice in choices
        if choice[1]["scene"]["falsePositivesPerImage"] <= 0.1
    ]
    pool = eligible or choices
    chosen = max(
        pool,
        key=lambda choice: (
            choice[1]["scene"]["recall"],
            choice[1]["scene"]["countExactRate"],
            choice[1]["scene"]["precision"],
            -choice[1]["scene"]["falsePositivesPerImage"],
        ),
    )
    curve = [
        {
            "threshold": threshold,
            "scene": metrics["scene"],
            "dense24To32": metrics["dense24To32"],
        }
        for threshold, metrics in choices
    ]
    return chosen[0], chosen[1], curve


def export_onnx(model, output: Path):
    model = ExportDetector(model.cpu().eval()).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),
        output,
        input_names=["images"],
        output_names=["boxes", "scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["boxes", "scores"],
        {"images": np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), np.float32)},
    )
    if actual[0].shape != (1, MAX_CANDIDATES, 4) or actual[1].shape != (
        1,
        MAX_CANDIDATES,
    ):
        raise RuntimeError(f"Unexpected ONNX outputs {[value.shape for value in actual]}")


def main():
    args = arguments()
    set_deterministic()
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_parts = [DetectorTiles(args.train, augment=True, seed=SEED)]
    train_parts.extend(
        DetectorTiles(
            path,
            augment=True,
            seed=SEED + index + 1,
            expected_source="jackfurby-playing-cards",
        )
        for index, path in enumerate(args.extra_train)
    )
    train_data = (
        train_parts[0]
        if len(train_parts) == 1
        else CombinedDetectorTiles(train_parts)
    )
    validation_data = DetectorTiles(args.validation, augment=False, seed=SEED)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    model = TinyCornerDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_loss = float("inf")
    checkpoint = ROOT / "detector.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        running = 0.0
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device=device, dtype=torch.float32).div_(255)
            images = photometric_augmentation(images, epoch, batch_index)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
        scheduler.step()
        val_loss = validation_loss(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "trainLoss": running / len(train_data),
            "validationLoss": val_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    predictions, latencies = validation_predictions(model, validation_loader, device)
    threshold, metrics, decoder_curve = tune_decoder(
        predictions, validation_data.frames
    )
    onnx_path = ROOT / "detector.onnx"
    export_onnx(model, onnx_path)
    config = {
        "schemaVersion": 1,
        "architecture": "tiny-stride-4-center-and-box",
        "license": "MIT",
        "input": {
            "name": "images",
            "shape": ["batch", 3, INPUT_SIZE, INPUT_SIZE],
            "range": [0, 1],
        },
        "outputs": {
            "boxes": {
                "name": "boxes",
                "shape": ["batch", MAX_CANDIDATES, 4],
                "format": "normalized-xyxy",
            },
            "scores": {
                "name": "scores",
                "shape": ["batch", MAX_CANDIDATES],
            },
        },
        "decoder": {
            "kind": "local-maximum-center-and-size",
            "threshold": threshold,
            "localMaximumKernel": 7,
            "nmsIou": 0.3,
            "boxes": "normalized xyxy after center/offset/size decoding",
            "scores": "sigmoid center probability",
        },
        "tiling": {"tileSize": 960, "overlap": 192},
        "minimumVisibleFraction": 0.9,
    }
    (ROOT / "detector-config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    metrics.update(
        {
            "split": "validation",
            "deck": "andrew-tidey-cards-pack",
            "trainScenes": len(train_data.frames),
            "trainingSources": [
                {
                    "id": dataset.manifest["source"]["id"],
                    "scenes": len(dataset.frames),
                    "datasetSha256": sha256(
                        dataset.dataset_dir / "dataset.sha256"
                    ),
                }
                for dataset in train_parts
            ],
            "validationScenes": len(validation_data.frames),
            "trainTiles": len(train_data),
            "validationTiles": len(validation_data),
            "epochs": args.epochs,
            "seed": SEED,
            "device": str(device),
            "meanModelLatencyMsPerTile": float(np.mean(latencies)),
            "decoderSweep": decoder_curve,
            "visibilityPolicy": {
                "minimumFraction": 0.9,
                "trainAnnotationsTotal": sum(
                    len(frame["annotations"]) for frame in train_data.frames
                ),
                "trainAnnotationsEvaluable": sum(
                    annotation["visibleFraction"] >= 0.9
                    for frame in train_data.frames
                    for annotation in frame["annotations"]
                ),
                "validationAnnotationsTotal": sum(
                    len(frame["annotations"]) for frame in validation_data.frames
                ),
                "validationAnnotationsEvaluable": sum(
                    annotation["visibleFraction"] >= 0.9
                    for frame in validation_data.frames
                    for annotation in frame["annotations"]
                ),
            },
            "history": history,
        }
    )
    gate_passed = (
        metrics["scene"]["recall"] >= 0.99
        and metrics["scene"]["falsePositivesPerImage"] <= 0.1
    )
    metrics["freezeGate"] = {
        "passed": gate_passed,
        "minimumSceneRecall": 0.99,
        "maximumFalsePositivesPerImage": 0.1,
    }
    (ROOT / "validation-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    frozen_files = [
        checkpoint,
        onnx_path,
        ROOT / "detector-config.json",
        ROOT / "model.py",
        ROOT / "data.py",
        ROOT / "decode.py",
        ROOT / "train.py",
        ROOT / "acquire_jackfurby.py",
        ROOT / "evaluate_holdout.py",
        ROOT / "test_detector.py",
        ROOT / "README.md",
        ROOT / "LICENSE",
        args.validation / "dataset.sha256",
        *[
            dataset.dataset_dir / "dataset.sha256"
            for dataset in train_parts
        ],
        *[
            dataset.dataset_dir / "provenance.json"
            for dataset in train_parts
            if (dataset.dataset_dir / "provenance.json").exists()
        ],
    ]
    freeze = {
        "schemaVersion": 1,
        "status": (
            "frozen-before-greywyvern"
            if gate_passed
            else "rejected-before-greywyvern"
        ),
        "files": {
            str(path.resolve()): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in frozen_files
        },
    }
    (ROOT / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frozen": gate_passed,
                "validation": metrics["scene"],
                "dense24To32": metrics["dense24To32"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
