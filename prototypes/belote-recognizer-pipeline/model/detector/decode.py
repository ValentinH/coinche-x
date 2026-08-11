"""Connected-component decoding and detection metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from data import OUTPUT_SIZE, TILE_SIZE


def decode_centers(
    heatmap: np.ndarray,
    regression: np.ndarray,
    threshold: float,
) -> list[dict]:
    detections = []
    local_radius = 3
    padded = np.pad(
        heatmap, local_radius, mode="constant", constant_values=-1
    )
    height, width = heatmap.shape
    for y in range(height):
        for x in range(width):
            score = heatmap[y, x]
            if score < threshold:
                continue
            if score < padded[
                y : y + 2 * local_radius + 1,
                x : x + 2 * local_radius + 1,
            ].max():
                continue
            offset_x, offset_y = 1 / (1 + np.exp(-regression[:2, y, x]))
            size_x, size_y = (
                1 / (1 + np.exp(-regression[2:, y, x])) * 0.5 * OUTPUT_SIZE
            )
            center_x = x + offset_x
            center_y = y + offset_y
            detections.append(
                {
                    "box": [
                        float(max(0, center_x - size_x / 2)),
                        float(max(0, center_y - size_y / 2)),
                        float(min(width, center_x + size_x / 2)),
                        float(min(height, center_y + size_y / 2)),
                    ],
                    "score": float(score),
                }
            )
    return nms(detections, threshold=0.3)


def iou(left, right) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def nms(detections: list[dict], threshold: float = 0.35) -> list[dict]:
    kept = []
    for detection in sorted(
        detections, key=lambda item: item["score"], reverse=True
    ):
        if all(iou(detection["box"], other["box"]) <= threshold for other in kept):
            kept.append(detection)
    return kept


def match(detections: list[dict], ground_truth: list, threshold: float = 0.5):
    matched = set()
    true_positives = 0
    for detection in sorted(
        detections, key=lambda item: item["score"], reverse=True
    ):
        candidates = [
            (iou(detection["box"], box), index)
            for index, box in enumerate(ground_truth)
            if index not in matched
        ]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= threshold:
            matched.add(index)
            true_positives += 1
    return true_positives, len(detections) - true_positives, len(ground_truth) - true_positives


def summarize(
    decoded_tiles: list[tuple[dict, list[dict]]],
    frames: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    tile_tp = tile_fp = tile_fn = 0
    by_scene = defaultdict(list)
    for target, detections in decoded_tiles:
        truths = target["boxes"].numpy().tolist()
        tp, fp, fn = match(detections, truths, iou_threshold)
        tile_tp += tp
        tile_fp += fp
        tile_fn += fn
        scale = TILE_SIZE / OUTPUT_SIZE
        tile_x, tile_y = target["tile"]
        for detection in detections:
            left, top, right, bottom = detection["box"]
            by_scene[target["scene"]].append(
                {
                    **detection,
                    "box": [
                        left * scale + tile_x,
                        top * scale + tile_y,
                        right * scale + tile_x,
                        bottom * scale + tile_y,
                    ],
                }
            )

    scene_tp = scene_fp = scene_fn = 0
    scene_results = []
    for frame in frames:
        truths = [
            [
                annotation["bbox"]["x"],
                annotation["bbox"]["y"],
                annotation["bbox"]["x"] + annotation["bbox"]["width"],
                annotation["bbox"]["y"] + annotation["bbox"]["height"],
            ]
            for annotation in frame["annotations"]
            if annotation["visibleFraction"] >= 0.9
        ]
        detections = nms(by_scene[frame["image"]])
        tp, fp, fn = match(detections, truths, iou_threshold)
        scene_tp += tp
        scene_fp += fp
        scene_fn += fn
        scene_results.append(
            {
                "image": frame["image"],
                "cardCount": frame["cardCount"],
                "groundTruthCount": len(truths),
                "detectionCount": len(detections),
                "countExact": len(truths) == len(detections),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    def metrics(tp, fp, fn):
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    exact_count = sum(result["countExact"] for result in scene_results)
    dense_results = [
        result for result in scene_results if 24 <= result["cardCount"] <= 32
    ]
    dense_tp = sum(result["tp"] for result in dense_results)
    dense_fp = sum(result["fp"] for result in dense_results)
    dense_fn = sum(result["fn"] for result in dense_results)
    return {
        "iouThreshold": iou_threshold,
        "tile": metrics(tile_tp, tile_fp, tile_fn),
        "scene": {
            **metrics(scene_tp, scene_fp, scene_fn),
            "images": len(frames),
            "falsePositivesPerImage": scene_fp / len(frames) if frames else 0.0,
            "countExactImages": exact_count,
            "countExactRate": exact_count / len(frames) if frames else 0.0,
        },
        "dense24To32": {
            **metrics(dense_tp, dense_fp, dense_fn),
            "images": len(dense_results),
            "falsePositivesPerImage": (
                dense_fp / len(dense_results) if dense_results else 0.0
            ),
            "countExactImages": sum(
                result["countExact"] for result in dense_results
            ),
            "countExactRate": (
                sum(result["countExact"] for result in dense_results)
                / len(dense_results)
                if dense_results
                else 0.0
            ),
        },
        "sceneResults": scene_results,
    }
