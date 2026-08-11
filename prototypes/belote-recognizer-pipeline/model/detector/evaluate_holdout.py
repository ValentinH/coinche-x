#!/usr/bin/env python3
"""One-shot frozen GreyWyvern detector evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from data import DetectorTiles, collate
from decode import decode_centers, summarize
from model import TinyCornerDetector
from train import ROOT, validation_predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze():
    freeze = json.loads((ROOT / "freeze.json").read_text())
    if freeze["status"] != "frozen-before-greywyvern":
        raise RuntimeError("Detector is not frozen")
    for name, expected in freeze["files"].items():
        path = Path(name)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen file changed: {path}")
    return freeze


def verify_dataset(dataset_dir: Path):
    for line in (dataset_dir / "dataset.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = dataset_dir / relative
        if sha256(path) != expected:
            raise RuntimeError(f"Holdout checksum mismatch: {relative}")


def claim_once():
    marker = ROOT / "holdout-consumed.json"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(
            {
                "status": "started",
                "startedUnixSeconds": int(time.time()),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return marker


def percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def onnx_latency(dataset, session):
    latencies = []
    for index in range(len(dataset)):
        image, _ = dataset[index]
        inputs = image.numpy().astype(np.float32)[None] / 255.0
        started = time.perf_counter()
        session.run(["boxes", "scores"], {"images": inputs})
        elapsed = (time.perf_counter() - started) * 1_000
        if index >= 5:
            latencies.append(elapsed)
    return {
        "runtime": "onnxruntime CPUExecutionProvider",
        "unit": "single 384x384 tile, preprocessing excluded",
        "warmSamples": len(latencies),
        "p50Ms": percentile(latencies, 50),
        "p95Ms": percentile(latencies, 95),
        "meanMs": float(np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    freeze = verify_freeze()
    marker = claim_once()
    try:
        verify_dataset(args.holdout)
        dataset = DetectorTiles(
            args.holdout,
            augment=False,
            seed=0,
            expected_source="greywyvern-cardset",
        )
        loader = DataLoader(
            dataset,
            batch_size=12,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        model = TinyCornerDetector()
        model.load_state_dict(torch.load(ROOT / "detector.pt", map_location="cpu"))
        predictions, _ = validation_predictions(model, loader, torch.device("cpu"))
        config = json.loads((ROOT / "detector-config.json").read_text())
        threshold = config["decoder"]["threshold"]
        decoded = [
            (target, decode_centers(heatmap, regression, threshold))
            for target, heatmap, regression in predictions
        ]
        metrics = summarize(decoded, dataset.frames)
        onnx_path = ROOT / "detector.onnx"
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        result = {
            **metrics,
            "split": "holdout",
            "deck": "greywyvern-cardset",
            "scenes": len(dataset.frames),
            "tiles": len(dataset),
            "decoderThresholdFrozen": threshold,
            "modelArtifact": {
                "path": "detector.onnx",
                "bytes": onnx_path.stat().st_size,
                "sha256": sha256(onnx_path),
            },
            "latency": onnx_latency(dataset, session),
            "freeze": freeze,
        }
        (ROOT / "holdout-metrics.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        marker.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "startedUnixSeconds": json.loads(marker.read_text())[
                        "startedUnixSeconds"
                    ],
                    "metricsSha256": sha256(ROOT / "holdout-metrics.json"),
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"scene": metrics["scene"], "dense": metrics["dense24To32"]}, indent=2))
    except Exception:
        # The marker deliberately remains: a failed holdout run may not be retried.
        raise


if __name__ == "__main__":
    main()
