#!/usr/bin/env python3
"""Generic single-card rectification and corner-index extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "generated" / "photo-source"
MAX_WORKING_SIDE = 1400
CARD_RATIO = 1.40
MAX_CANDIDATE_QUADS = 48
LABEL_HASH_WINDOWS = ((36, 40), (40, 44))
LABEL_HASH_CORNERS = ("topLeft", "bottomRight")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def contour_candidates(image: np.ndarray) -> list[tuple[float, np.ndarray, dict[str, float]]]:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_WORKING_SIDE / max(height, width))
    working = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = max(12, int(0.55 * median))
    upper = min(245, int(1.35 * median))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)
    fine_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fine_edges = []
    for lower_factor, upper_factor in ((0.20, 0.65), (0.32, 0.95)):
        fine = cv2.Canny(
            blurred,
            max(8, int(lower_factor * median)),
            max(24, int(upper_factor * median)),
        )
        fine_edges.append(
            cv2.morphologyEx(
                fine,
                cv2.MORPH_CLOSE,
                fine_kernel,
                iterations=1,
            )
        )

    border = np.concatenate(
        [
            working[: max(2, working.shape[0] // 30)].reshape(-1, 3),
            working[-max(2, working.shape[0] // 30) :].reshape(-1, 3),
            working[:, : max(2, working.shape[1] // 30)].reshape(-1, 3),
            working[:, -max(2, working.shape[1] // 30) :].reshape(-1, 3),
        ]
    )
    border_color = np.median(
        cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3),
        axis=0,
    )
    lab = cv2.cvtColor(working, cv2.COLOR_RGB2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - border_color, axis=2)
    color_mask = np.uint8(distance > max(10.0, float(np.percentile(distance, 62)))) * 255
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    hsv = cv2.cvtColor(working, cv2.COLOR_RGB2HSV)
    broad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    white_masks = []
    for quantile in (50, 58, 78, 88, 90, 92):
        brightness_floor = max(
            105.0, float(np.percentile(hsv[:, :, 2], quantile))
        )
        white_mask = np.uint8(
            (hsv[:, :, 2] >= brightness_floor) & (hsv[:, :, 1] <= 105)
        ) * 255
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_CLOSE, broad_kernel, iterations=3
        )
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        white_masks.append(white_mask)

    contours: list[np.ndarray] = []
    for mask in (*fine_edges, edges, color_mask, *white_masks):
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)

    image_area = float(working.shape[0] * working.shape[1])
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    for contour in contours:
        contour_area = abs(float(cv2.contourArea(contour)))
        if not 0.015 * image_area <= contour_area <= 0.92 * image_area:
            continue
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        if min(rect_width, rect_height) < 36:
            continue
        portrait_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)
        if not 1.18 <= portrait_ratio <= 1.75:
            continue
        rectangle_area = rect_width * rect_height
        rectangularity = min(1.0, contour_area / max(rectangle_area, 1.0))
        if rectangularity < 0.42:
            continue
        area_fraction = rectangle_area / image_area
        if area_fraction >= 0.94:
            continue
        ratio_fit = math.exp(-4.0 * abs(math.log(portrait_ratio / CARD_RATIO)))
        score = area_fraction * (0.35 + 0.65 * rectangularity) * (0.45 + 0.55 * ratio_fit)
        box = cv2.boxPoints(rect) / scale
        candidates.append(
            (
                score,
                order_quad(box),
                {
                    "areaFraction": round(area_fraction, 5),
                    "rectangularity": round(rectangularity, 5),
                    "aspectRatio": round(portrait_ratio, 5),
                    "score": round(score, 6),
                },
            )
        )
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return candidates


def warp_card(
    image: np.ndarray,
    quad: np.ndarray,
    rotation: int = 0,
    destination_width: int = 512,
) -> np.ndarray:
    source = np.roll(order_quad(quad), -(rotation % 4), axis=0)
    destination_height = round(destination_width * CARD_RATIO)
    destination = np.array(
        [
            [0, 0],
            [destination_width - 1, 0],
            [destination_width - 1, destination_height - 1],
            [0, destination_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (destination_width, destination_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def quad_touches_frame(
    quad: np.ndarray,
    image_shape: tuple[int, ...],
    margin_fraction: float = 0.012,
) -> bool:
    height, width = image_shape[:2]
    margin_x = width * margin_fraction
    margin_y = height * margin_fraction
    minimum = np.min(quad, axis=0)
    maximum = np.max(quad, axis=0)
    touched_sides = sum(
        (
            minimum[0] <= margin_x,
            minimum[1] <= margin_y,
            maximum[0] >= width - margin_x,
            maximum[1] >= height - margin_y,
        )
    )
    return touched_sides >= 2


def corner_foreground_diagnostics(
    crop: np.ndarray,
    paper_color: np.ndarray,
) -> dict[str, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    margin = max(3, round(min(gray.shape) * 0.10))
    interior = gray[margin:-margin, margin:-margin]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    distance = np.linalg.norm(
        lab[margin:-margin, margin:-margin] - paper_color,
        axis=2,
    )
    foreground = distance > 30
    paper = distance < 18
    edges = cv2.Canny(interior, 25, 90)
    strong_edges = cv2.Canny(interior, 50, 150)
    foreground_fraction = float(np.mean(foreground))
    paper_fraction = float(np.mean(paper))
    edge_fraction = float(cv2.countNonZero(edges) / edges.size)
    strong_edge_fraction = float(
        cv2.countNonZero(strong_edges) / strong_edges.size
    )
    component_mask = np.uint8(
        np.linalg.norm(lab - paper_color, axis=2) > 30
    )
    component_zone = component_mask[
        round(crop.shape[0] * 0.06) : round(crop.shape[0] * 0.92),
        round(crop.shape[1] * 0.06) : round(crop.shape[1] * 0.69),
    ]
    component_zone = cv2.morphologyEx(
        component_zone,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    component_count, _, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(component_zone, connectivity=8)
    )
    compact_components = 0
    zone_height, zone_width = component_zone.shape
    for index in range(1, component_count):
        x, y, width, height, area = component_stats[index]
        center_x, center_y = component_centroids[index]
        if (
            6 <= area <= component_zone.size * 0.18
            and width <= zone_width * 0.58
            and height <= zone_height * 0.58
            and 0.06 <= center_x / zone_width <= 0.86
            and 0.04 <= center_y / zone_height <= 0.90
        ):
            compact_components += 1
    index_structure_usable = compact_components >= 2
    score = (
        0.55 * min(1.0, foreground_fraction / 0.18)
        + 0.25 * min(1.0, paper_fraction / 0.68)
        + 0.20 * min(1.0, edge_fraction / 0.075)
    )
    return {
        "foregroundFraction": foreground_fraction,
        "paperFraction": paper_fraction,
        "edgeFraction": edge_fraction,
        "strongEdgeFraction": strong_edge_fraction,
        "compactIndexComponentCount": compact_components,
        "indexStructureUsable": index_structure_usable,
        "score": score,
    }


def diagonal_similarity(first: np.ndarray, second: np.ndarray) -> float:
    normalized = []
    for crop in (first, second):
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        normalized.append((gray - gray.mean()) / (gray.std() + 1e-6))
    return max(
        float(np.mean(normalized[0] * np.roll(normalized[1], (dy, dx), (0, 1))))
        for dy in range(-4, 5)
        for dx in range(-4, 5)
    )


def estimate_paper_color(
    hsv: np.ndarray,
    lab: np.ndarray,
) -> np.ndarray:
    height, width = hsv.shape[:2]
    border_band = np.zeros((height, width), dtype=bool)
    band_x = max(4, round(width * 0.22))
    band_y = max(4, round(height * 0.16))
    border_band[:band_y] = True
    border_band[-band_y:] = True
    border_band[:, :band_x] = True
    border_band[:, -band_x:] = True
    light_neutral_score = (
        hsv[:, :, 2].astype(np.float32)
        - 0.65 * hsv[:, :, 1].astype(np.float32)
    )
    border_scores = light_neutral_score[border_band]
    threshold = float(np.percentile(border_scores, 62))
    candidates = border_band & (light_neutral_score >= threshold)
    return np.median(lab[candidates], axis=0)


def semantic_diagnostics(card: np.ndarray) -> dict[str, float | bool]:
    hsv = cv2.cvtColor(card, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(card, cv2.COLOR_RGB2LAB).astype(np.float32)
    height, width = card.shape[:2]
    interior = hsv[
        round(height * 0.12) : round(height * 0.88),
        round(width * 0.12) : round(width * 0.88),
    ]
    neutral_light_fraction = float(
        np.mean((interior[:, :, 2] >= 72) & (interior[:, :, 1] <= 175))
    )
    paper_color = estimate_paper_color(hsv, lab)
    paper_distance = np.linalg.norm(lab - paper_color, axis=2)
    paper_band = np.zeros((height, width), dtype=bool)
    paper_band[
        round(height * 0.04) : round(height * 0.16),
        round(width * 0.05) : round(width * 0.95),
    ] = True
    paper_band[
        round(height * 0.84) : round(height * 0.96),
        round(width * 0.05) : round(width * 0.95),
    ] = True
    paper_band[
        round(height * 0.05) : round(height * 0.95),
        round(width * 0.04) : round(width * 0.16),
    ] = True
    paper_band[
        round(height * 0.05) : round(height * 0.95),
        round(width * 0.84) : round(width * 0.96),
    ] = True
    paper_coverage = float(np.mean(paper_distance[paper_band] < 18))
    gray = cv2.cvtColor(card, cv2.COLOR_RGB2GRAY)
    card_edges = cv2.Canny(gray, 25, 90)
    border_width = max(3, round(min(height, width) * 0.045))
    border_mask = np.zeros_like(card_edges, dtype=bool)
    border_mask[:border_width] = True
    border_mask[-border_width:] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_edge_fraction = float(
        np.count_nonzero(card_edges[border_mask]) / np.count_nonzero(border_mask)
    )
    top_left_crop = corner_crop(card)
    bottom_right_crop = corner_crop(card, bottom_right=True)
    top_left = corner_foreground_diagnostics(top_left_crop, paper_color)
    bottom_right = corner_foreground_diagnostics(
        bottom_right_crop,
        paper_color,
    )
    index_similarity = diagonal_similarity(top_left_crop, bottom_right_crop)
    diagonal_score = min(top_left["score"], bottom_right["score"])
    paper_structure_usable = paper_coverage >= 0.48 or (
        paper_coverage >= 0.44
        and top_left["indexStructureUsable"]
        and bottom_right["indexStructureUsable"]
        and top_left["paperFraction"] >= 0.50
        and bottom_right["paperFraction"] >= 0.50
        and index_similarity >= 0.80
    )
    semantic_score = (
        0.58 * diagonal_score
        + 0.30 * min(1.0, paper_coverage / 0.72)
        + 0.08 * min(1.0, neutral_light_fraction / 0.72)
        + 0.04 * min(1.0, border_edge_fraction / 0.06)
    )
    usable = (
        paper_structure_usable
        and top_left["edgeFraction"] >= 0.004
        and bottom_right["edgeFraction"] >= 0.004
        and top_left["strongEdgeFraction"] >= 0.004
        and bottom_right["strongEdgeFraction"] >= 0.004
        and top_left["foregroundFraction"] >= 0.025
        and bottom_right["foregroundFraction"] >= 0.025
        and top_left["paperFraction"] >= 0.25
        and bottom_right["paperFraction"] >= 0.25
        and top_left["indexStructureUsable"]
        and bottom_right["indexStructureUsable"]
        and index_similarity >= 0.25
    )
    return {
        "semanticScore": semantic_score,
        "semanticUsable": usable,
        "neutralLightFraction": neutral_light_fraction,
        "paperCoverage": paper_coverage,
        "paperStructureUsable": paper_structure_usable,
        "borderEdgeFraction": border_edge_fraction,
        "topLeftEdgeFraction": top_left["edgeFraction"],
        "bottomRightEdgeFraction": bottom_right["edgeFraction"],
        "topLeftStrongEdgeFraction": top_left["strongEdgeFraction"],
        "bottomRightStrongEdgeFraction": bottom_right["strongEdgeFraction"],
        "topLeftForegroundFraction": top_left["foregroundFraction"],
        "bottomRightForegroundFraction": bottom_right["foregroundFraction"],
        "topLeftPaperFraction": top_left["paperFraction"],
        "bottomRightPaperFraction": bottom_right["paperFraction"],
        "topLeftCompactIndexComponentCount": top_left[
            "compactIndexComponentCount"
        ],
        "bottomRightCompactIndexComponentCount": bottom_right[
            "compactIndexComponentCount"
        ],
        "topLeftIndexStructureUsable": top_left["indexStructureUsable"],
        "bottomRightIndexStructureUsable": bottom_right[
            "indexStructureUsable"
        ],
        "diagonalSimilarity": index_similarity,
    }


def select_card_candidate(
    image: np.ndarray,
    candidates: list[tuple[float, np.ndarray, dict[str, float]]],
) -> tuple[np.ndarray, np.ndarray, int, float, dict[str, Any]] | None:
    variants = []
    for geometry_score, quad, geometry in candidates[:MAX_CANDIDATE_QUADS]:
        if geometry["areaFraction"] >= 0.94 or quad_touches_frame(quad, image.shape):
            continue
        for rotation in range(4):
            ordered = np.roll(order_quad(quad), -(rotation % 4), axis=0)
            mapped_width = float(np.linalg.norm(ordered[1] - ordered[0]))
            mapped_height = float(np.linalg.norm(ordered[2] - ordered[1]))
            if mapped_width < 1.0 or mapped_height < 1.0:
                continue
            mapped_ratio = mapped_height / max(mapped_width, 1.0)
            orientation_fit = math.exp(
                -4.0 * abs(math.log(mapped_ratio / CARD_RATIO))
            )
            preview = warp_card(
                image,
                quad,
                rotation=rotation,
                destination_width=256,
            )
            semantic = semantic_diagnostics(preview)
            score = (
                semantic["semanticScore"]
                + 0.10 * orientation_fit
                + min(0.06, geometry_score * 0.09)
            )
            variants.append(
                (
                    score,
                    quad,
                    rotation,
                    geometry,
                    {**semantic, "orientationFit": orientation_fit},
                )
            )
    if not variants:
        return None
    score, quad, rotation, geometry, semantic = max(
        variants,
        key=lambda item: (
            item[4]["semanticUsable"],
            item[0],
            -item[2],
        ),
    )
    card = warp_card(image, quad, rotation=rotation)
    final_semantic = semantic_diagnostics(card)
    diagnostics = {
        **geometry,
        **{
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in final_semantic.items()
        },
        "rotation": rotation,
        "orientationFit": round(float(semantic["orientationFit"]), 6),
        "candidateCount": len(candidates),
        "variantCount": len(variants),
    }
    return card, quad, rotation, score, diagnostics


def center_fallback(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    box_height = min(height * 0.82, width * CARD_RATIO * 0.82)
    box_width = box_height / CARD_RATIO
    left = (width - box_width) / 2
    top = (height - box_height) / 2
    quad = np.array(
        [
            [left, top],
            [left + box_width, top],
            [left + box_width, top + box_height],
            [left, top + box_height],
        ],
        dtype=np.float32,
    )
    return warp_card(image, quad), quad


def save_rgb(path: Path, image: np.ndarray, *, max_side: int | None = None) -> None:
    output = Image.fromarray(image)
    if max_side and max(output.size) > max_side:
        output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        output.save(path, quality=88, optimize=True)
    else:
        output.save(path, optimize=True)


def corner_crop(card: np.ndarray, bottom_right: bool = False) -> np.ndarray:
    if bottom_right:
        card = cv2.rotate(card, cv2.ROTATE_180)
    side = max(24, round(card.shape[1] * 0.43))
    crop = card[:side, :side]
    return cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)


def process_asset(source_root: Path, gallery_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    source = source_root / entry["localPath"]
    with Image.open(source) as opened:
        image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    candidates = contour_candidates(image)
    selected = select_card_candidate(image, candidates)
    if selected:
        card, quad, _rotation, score, diagnostics = selected
        detected = bool(diagnostics["semanticUsable"])
        method = (
            "semantic-quad-and-rotation"
            if detected
            else "flagged-semantic-quad"
        )
    else:
        card, quad = center_fallback(image)
        score = 0.0
        semantic = semantic_diagnostics(card)
        diagnostics = {
            "areaFraction": 0.0,
            "rectangularity": 0.0,
            "aspectRatio": round(card.shape[0] / card.shape[1], 5),
            "score": 0.0,
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in semantic.items()
            },
            "rotation": 0,
            "candidateCount": len(candidates),
            "variantCount": 0,
        }
        detected = False
        method = "flagged-center-fallback"

    asset_root = gallery_root / "assets" / entry["id"]
    full_path = asset_root / "photo.jpg"
    card_path = asset_root / "card.jpg"
    top_left_path = asset_root / "top-left.png"
    bottom_right_path = asset_root / "bottom-right.png"
    save_rgb(full_path, image, max_side=640)
    save_rgb(card_path, card, max_side=512)
    save_rgb(top_left_path, corner_crop(card))
    save_rgb(bottom_right_path, corner_crop(card, bottom_right=True))

    output = {
        **entry,
        "sourceSha256Verified": sha256(source) == entry["sha256"],
        "usableForClassifier": detected,
        "method": method,
        "score": round(score, 6),
        "diagnostics": diagnostics,
        "sourceSize": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
        "gallery": {
            "photo": str(full_path.relative_to(gallery_root)),
            "card": str(card_path.relative_to(gallery_root)),
            "topLeft": str(top_left_path.relative_to(gallery_root)),
            "bottomRight": str(bottom_right_path.relative_to(gallery_root)),
        },
        "derivedSha256": {
            "photo": sha256(full_path),
            "card": sha256(card_path),
            "topLeft": sha256(top_left_path),
            "bottomRight": sha256(bottom_right_path),
        },
    }
    return output


def corner_perceptual_hash(
    path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read corner crop {path}")
    region = image[:height, :width]
    normalized = cv2.resize(
        region,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    coefficients = cv2.dct(normalized)[:8, :8]
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    return coefficients > threshold


def infer_corner_label(
    target: dict[str, Any],
    corner: str,
    width: int,
    height: int,
    references: list[dict[str, Any]],
    hashes: dict[tuple[str, str, int, int], np.ndarray],
) -> dict[str, Any]:
    target_hash = hashes[(target["id"], corner, width, height)]
    distances_by_label: dict[str, dict[str, int]] = {}
    for reference in references:
        if (
            reference["id"] == target["id"]
            or reference["captureFamily"] == target["captureFamily"]
        ):
            continue
        distance = min(
            int(
                np.count_nonzero(
                    target_hash
                    != hashes[
                        (
                            reference["id"],
                            reference_corner,
                            width,
                            height,
                        )
                    ]
                )
            )
            for reference_corner in LABEL_HASH_CORNERS
        )
        family_distances = distances_by_label.setdefault(
            reference["label"],
            {},
        )
        previous = family_distances.get(reference["captureFamily"])
        if previous is None or distance < previous:
            family_distances[reference["captureFamily"]] = distance

    ranking = []
    for label, family_distances in distances_by_label.items():
        support = sorted(
            (
                {"captureFamily": family, "distance": distance}
                for family, distance in family_distances.items()
            ),
            key=lambda item: (item["distance"], item["captureFamily"]),
        )
        if len(support) < 2:
            continue
        ranking.append(
            {
                "label": label,
                "score": support[0]["distance"] + support[1]["distance"],
                "support": support[:3],
            }
        )
    ranking.sort(key=lambda item: (item["score"], item["label"]))
    if len(ranking) < 2:
        return {"label": None, "confident": False}

    best, runner_up = ranking[:2]
    second_distance = best["support"][1]["distance"]
    margin = runner_up["score"] - best["score"]
    confident = (
        best["score"] <= 32
        and second_distance <= 16
        and margin >= 4
    )
    return {
        "label": best["label"] if confident else None,
        "candidateLabel": best["label"],
        "confident": confident,
        "score": best["score"],
        "runnerUpLabel": runner_up["label"],
        "runnerUpScore": runner_up["score"],
        "margin": margin,
        "support": best["support"],
    }


def apply_label_consensus(
    outputs: list[dict[str, Any]],
    gallery_root: Path,
) -> None:
    references = [
        entry for entry in outputs if entry["usableForClassifier"]
    ]
    hashes = {
        (entry["id"], corner, width, height): corner_perceptual_hash(
            gallery_root / entry["gallery"][corner],
            width,
            height,
        )
        for entry in references
        for corner in LABEL_HASH_CORNERS
        for width, height in LABEL_HASH_WINDOWS
    }
    for target in references:
        predictions = []
        for width, height in LABEL_HASH_WINDOWS:
            for corner in LABEL_HASH_CORNERS:
                predictions.append(
                    {
                        "window": {"width": width, "height": height},
                        "corner": corner,
                        **infer_corner_label(
                            target,
                            corner,
                            width,
                            height,
                            references,
                            hashes,
                        ),
                    }
                )
        strong_labels = [
            prediction["label"]
            for prediction in predictions
            if prediction["confident"] and prediction["margin"] >= 6
        ]
        label_counts = {
            label: strong_labels.count(label)
            for label in set(strong_labels)
        }
        inferred_label, agreement_count = (
            max(
                label_counts.items(),
                key=lambda item: (item[1], item[0]),
            )
            if label_counts
            else (None, 0)
        )
        if agreement_count < 3:
            inferred_label = None
        target["diagnostics"]["labelConsensus"] = {
            "inferredLabel": inferred_label,
            "agreementCount": agreement_count,
            "expectedLabel": target["label"],
            "leaveCaptureFamilyOut": target["captureFamily"],
            "predictions": predictions,
        }
        if inferred_label is not None and inferred_label != target["label"]:
            target["usableForClassifier"] = False
            target["method"] = "flagged-label-mismatch"


def preprocess(source_root: Path) -> dict[str, Any]:
    provenance_path = source_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    gallery_root = source_root / "gallery"
    if gallery_root.exists():
        shutil.rmtree(gallery_root)
    gallery_root.mkdir(parents=True)

    outputs = [
        process_asset(source_root, gallery_root, entry)
        for entry in sorted(provenance["entries"], key=lambda item: item["id"])
    ]
    apply_label_consensus(outputs, gallery_root)
    manifest = {
        "schemaVersion": 1,
        "dataset": provenance["dataset"],
        "sourceBytes": provenance["sourceBytes"],
        "assetCount": len(outputs),
        "usableCount": sum(item["usableForClassifier"] for item in outputs),
        "failedCount": sum(not item["usableForClassifier"] for item in outputs),
        "labelMismatchCount": sum(
            item["method"] == "flagged-label-mismatch"
            for item in outputs
        ),
        "entries": outputs,
    }
    (gallery_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    classifier_path = source_root / "classifier-crops.jsonl"
    with classifier_path.open("w") as output:
        for entry in outputs:
            if not entry["usableForClassifier"]:
                continue
            for corner in ("topLeft", "bottomRight"):
                output.write(
                    json.dumps(
                        {
                            "id": f'{entry["id"]}--{corner}',
                            "label": entry["label"],
                            "rank": entry["rank"],
                            "suit": entry["suit"],
                            "split": entry["split"],
                            "captureFamily": entry["captureFamily"],
                            "corner": corner,
                            "path": str(
                                Path("gallery") / entry["gallery"][corner]
                            ),
                            "sourceSha256": entry["sha256"],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    for name in ("index.html", "main.js", "styles.css"):
        shutil.copy2(HERE / "gallery" / name, gallery_root / name)
    print(
        f'Photo crops ready: {manifest["usableCount"]}/{manifest["assetCount"]} '
        f'detected, {manifest["failedCount"]} flagged fallbacks.'
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    preprocess(args.source.resolve())


if __name__ == "__main__":
    main()
