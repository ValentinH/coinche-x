import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PROTOTYPE_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_DIRECTORY = PROTOTYPE_DIRECTORY.parent.parent
CORPUS_DIRECTORY = REPOSITORY_DIRECTORY / "test-data/card-recognition/smoke"
MODEL_DIRECTORY = (
    PROTOTYPE_DIRECTORY
    / ".cache/fvannee-android/app/src/main/res/raw"
)
PUBLIC_VDR_DIRECTORY = PROTOTYPE_DIRECTORY / ".cache/commons-vdr"
SAMPLE_FILES = [
    MODEL_DIRECTORY / "cartamundi_samples_25.data",
    MODEL_DIRECTORY / "samples_25.data",
]
RESPONSE_FILES = [
    MODEL_DIRECTORY / "cartamundi_responses_25.data",
    MODEL_DIRECTORY / "responses_25.data",
]
CARDS = "23456789TJQKA"
SUITS = "CDHS"
SMALL_SIZE = 25
MATCH_FOUND_PARAM = 7000
INTERESTING_AREA_PARAM = 6000
MIN_DIMENSION = 3.5
MIN_SUIT_DIMENSION = 2
PAIR_DISTANCE_PARAM = 1.3
LONG_SIDE = 2000


def load_model(model_name):
    selected_indexes = {
        "cartamundi": [0],
        "grimaud": [1],
        "combined": [0, 1],
        "combined-vdr": [0, 1],
        "combined-smoke-vdr": [0, 1],
    }[model_name]
    samples = np.concatenate(
        [np.loadtxt(SAMPLE_FILES[index], np.float32) for index in selected_indexes],
    )
    responses = np.concatenate(
        [
            np.loadtxt(RESPONSE_FILES[index], np.float32)
            for index in selected_indexes
        ],
    ).reshape((-1, 1))
    red_samples = []
    red_responses = []
    black_samples = []
    black_responses = []

    for sample, response in zip(samples, responses):
        feature = chr(int(response[0]))
        if feature not in "DH":
            black_samples.append(sample)
            black_responses.append(response)
        if feature not in "SC":
            red_samples.append(sample)
            red_responses.append(response)

    if model_name == "combined-vdr":
        for sample, response in load_public_vdr_samples():
            black_samples.append(sample)
            black_responses.append(response)
            red_samples.append(sample)
            red_responses.append(response)
    if model_name == "combined-smoke-vdr":
        for sample, response in load_smoke_vdr_samples():
            black_samples.append(sample)
            black_responses.append(response)
            red_samples.append(sample)
            red_responses.append(response)

    return {
        "black": train_knn(black_samples, black_responses),
        "red": train_knn(red_samples, red_responses),
    }


def load_public_vdr_samples():
    samples = []
    for source_rank, canonical_rank in {"V": "J", "D": "Q", "R": "K"}.items():
        image = cv2.imread(
            str(PUBLIC_VDR_DIRECTORY / f"{source_rank}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        _, thresholded = cv2.threshold(
            image[:115, :120],
            160,
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(
            (
                contour
                for contour in contours
                if 100 < cv2.contourArea(contour) < 5000
            ),
            key=cv2.contourArea,
        )
        rectangle = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle = rectangle
        if width > height:
            width, height = height, width
            angle += 90
        region = subimage(
            thresholded,
            (center_x, center_y),
            angle,
            round(width),
            round(height),
        )
        for variant in (
            region,
            cv2.erode(region, np.ones((2, 2), np.uint8)),
            cv2.dilate(region, np.ones((2, 2), np.uint8)),
        ):
            sample = cv2.resize(
                variant,
                (SMALL_SIZE, SMALL_SIZE),
                interpolation=cv2.INTER_LINEAR,
            ).reshape((SMALL_SIZE * SMALL_SIZE,))
            samples.append(
                (
                    np.float32(sample),
                    np.float32([ord(canonical_rank)]),
                ),
            )
    return samples


def load_smoke_vdr_samples():
    thresholded, _ = prepare_image(
        CORPUS_DIRECTORY / "images/oneplus-11-5g_vdr_16.jpg",
    )
    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    samples = []
    sources = {
        "J": (805, 400),
        "Q": (650, 113),
        "K": (1383, 425),
    }
    for canonical_rank, target in sources.items():
        contour = min(
            (
                contour
                for contour in contours
                if 200 < cv2.contourArea(contour) < 800
            ),
            key=lambda candidate: math.dist(
                cv2.minAreaRect(candidate)[0],
                target,
            ),
        )
        (center_x, center_y), (width, height), angle = cv2.minAreaRect(contour)
        if width > height:
            width, height = height, width
            angle += 90
        region = subimage(
            thresholded,
            (center_x, center_y),
            angle,
            round(width),
            round(height),
        )
        for variant in (
            region,
            cv2.erode(region, np.ones((2, 2), np.uint8)),
            cv2.dilate(region, np.ones((2, 2), np.uint8)),
        ):
            sample = cv2.resize(
                variant,
                (SMALL_SIZE, SMALL_SIZE),
                interpolation=cv2.INTER_LINEAR,
            ).reshape((SMALL_SIZE * SMALL_SIZE,))
            samples.append(
                (
                    np.float32(sample),
                    np.float32([ord(canonical_rank)]),
                ),
            )
    return samples


def train_knn(samples, responses):
    model = cv2.ml.KNearest_create()
    model.train(
        np.asarray(samples, dtype=np.float32),
        cv2.ml.ROW_SAMPLE,
        np.asarray(responses, dtype=np.float32),
    )
    return model


def prepare_image(path):
    color_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    longest_side = max(color_image.shape[:2])
    if longest_side > LONG_SIDE:
        scale = LONG_SIDE / longest_side
        color_image = cv2.resize(
            color_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    coefficients = np.array([0.5, 1.5, -1], dtype=np.float32).reshape((1, 3))
    transformed = cv2.transform(color_image, coefficients)
    thresholded = cv2.adaptiveThreshold(
        transformed,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        101,
        40,
    )
    return thresholded, color_image


def detect_cards(model, thresholded, color_image):
    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    features = []
    for contour in contours:
        feature = classify_contour(model, thresholded, color_image, contour)
        if feature is not None:
            features.append(feature)
    return pair_features(features), features, len(contours)


def classify_contour(model, thresholded, color_image, contour):
    rectangle = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rectangle
    minimum_area = thresholded.shape[0] * thresholded.shape[1] / INTERESTING_AREA_PARAM
    dimension_ratio = width / height if height else 0
    if (
        width * height <= minimum_area
        or dimension_ratio >= MIN_DIMENSION
        or dimension_ratio <= 1 / MIN_DIMENSION
    ):
        return None

    if width > height:
        width, height = height, width
        angle += 90
    region = subimage(
        thresholded,
        (center_x, center_y),
        angle,
        max(1, round(width)),
        max(1, round(height)),
    )
    color = classify_color(
        color_image[
            min(max(round(center_y), 0), color_image.shape[0] - 1),
            min(max(round(center_x), 0), color_image.shape[1] - 1),
        ],
    )
    feature, rotated, distance = classify_region(model[color], region)
    if feature is None:
        return None
    if rotated:
        angle = (angle + 180) % 360
    return {
        "feature": feature,
        "center": (center_x, center_y),
        "size": (width, height),
        "angle": angle,
        "distance": distance,
    }


def subimage(image, center, angle, width, height):
    radians = angle * math.pi / 180
    vector_x = (math.cos(radians), math.sin(radians))
    vector_y = (-math.sin(radians), math.cos(radians))
    source_x = (
        center[0]
        - vector_x[0] * width / 2
        - vector_y[0] * height / 2
    )
    source_y = (
        center[1]
        - vector_x[1] * width / 2
        - vector_y[1] * height / 2
    )
    mapping = np.array(
        [
            [vector_x[0], vector_y[0], source_x],
            [vector_x[1], vector_y[1], source_y],
        ],
    )
    return cv2.warpAffine(
        image,
        mapping,
        (width, height),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def classify_color(pixel):
    sample = np.uint8([[pixel]])
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    references = cv2.cvtColor(
        np.uint8([[[0, 0, 0], [0, 0, 255], [0, 128, 255]]]),
        cv2.COLOR_BGR2LAB,
    )[0].astype(float)
    closest = int(np.argmin(np.linalg.norm(references - sample_lab, axis=1)))
    return ("black", "red", "red")[closest]


def classify_region(model, region):
    resized = cv2.resize(region, (SMALL_SIZE, SMALL_SIZE))
    best = None
    for rotated in (False, True):
        sample = np.float32(resized.reshape((1, SMALL_SIZE * SMALL_SIZE)))
        _, result, _, distances = model.findNearest(sample, k=1)
        distance = float(distances[0][0])
        if best is None or distance < best["distance"]:
            best = {
                "feature": chr(int(result[0][0])),
                "rotated": rotated,
                "distance": distance,
            }
        resized = np.rot90(resized, 2)

    feature = best["feature"]
    dimension_ratio = region.shape[0] / region.shape[1]
    matches = best["distance"] < MATCH_FOUND_PARAM * SMALL_SIZE * SMALL_SIZE
    suit_shape_matches = (
        feature not in SUITS
        or dimension_ratio < MIN_SUIT_DIMENSION
        or 1 / dimension_ratio > MIN_SUIT_DIMENSION
    )
    if not matches or not suit_shape_matches:
        return None, None, best["distance"]
    return feature, best["rotated"], best["distance"]


def pair_features(features):
    results = []
    for rank_feature in features:
        rank = rank_feature["feature"]
        if rank not in CARDS:
            continue
        rank_x, rank_y = rank_feature["center"]
        maximum_distance = PAIR_DISTANCE_PARAM * max(rank_feature["size"]) ** 2
        best = None
        for suit_feature in features:
            suit = suit_feature["feature"]
            if suit not in SUITS:
                continue
            suit_x, suit_y = suit_feature["center"]
            if (suit_x - rank_x) ** 2 + (suit_y - rank_y) ** 2 >= maximum_distance:
                continue
            line_angle = (
                math.atan2(-(rank_y - suit_y), suit_x - rank_x)
                % (2 * math.pi)
            ) / math.pi * 180
            angle_difference = (
                line_angle + 360 - rank_feature["angle"]
            ) % 360
            angle_error = min(
                abs(angle_difference - 90),
                abs(angle_difference - 270),
            )
            if angle_error < 35 and (best is None or angle_error < best["error"]):
                normalized_rank = (
                    "6" if rank == "9" and angle_difference > 180 else rank
                )
                best = {
                    "card": normalize_card(f"{normalized_rank}{suit}"),
                    "error": angle_error,
                }
        if best is not None:
            results.append(best["card"])
    return sorted(set(results))


def normalize_card(card):
    rank, suit = card[:-1], card[-1]
    return f"{'10' if rank == 'T' else rank}{suit}"


def score_cards(detected, expected):
    detected_counts = Counter(detected)
    expected_counts = Counter(expected)
    correct = sum(
        min(count, detected_counts.get(card, 0))
        for card, count in expected_counts.items()
    )
    return {
        "exact": correct == len(expected) and len(detected) == len(expected),
        "correct": correct,
        "detected": len(detected),
        "expected": len(expected),
    }


def ratio(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else 0


def main():
    global LONG_SIDE, PAIR_DISTANCE_PARAM
    manifest = json.loads((CORPUS_DIRECTORY / "manifest.json").read_text())
    model_name = next(
        (
            argument.removeprefix("--model=")
            for argument in sys.argv[1:]
            if argument.startswith("--model=")
        ),
        "combined",
    )
    requested_file = next(
        (argument for argument in sys.argv[1:] if not argument.startswith("--")),
        None,
    )
    PAIR_DISTANCE_PARAM = float(
        next(
            (
                argument.removeprefix("--pair-distance=")
                for argument in sys.argv[1:]
                if argument.startswith("--pair-distance=")
            ),
            PAIR_DISTANCE_PARAM,
        ),
    )
    LONG_SIDE = int(
        next(
            (
                argument.removeprefix("--long-side=")
                for argument in sys.argv[1:]
                if argument.startswith("--long-side=")
            ),
            LONG_SIDE,
        ),
    )
    images = [
        image
        for image in manifest["images"]
        if requested_file is None or image["file"].endswith(requested_file)
    ]
    if not images:
        raise RuntimeError(f"Image absente du manifeste : {requested_file}")

    started_at = time.perf_counter()
    model = load_model(model_name)
    init_elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    results = []
    for image in images:
        thresholded, color_image = prepare_image(
            CORPUS_DIRECTORY / image["file"],
        )
        started_at = time.perf_counter()
        cards, features, contour_count = detect_cards(
            model,
            thresholded,
            color_image,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        expected_cards = image["cards"]
        score = score_cards(cards, expected_cards)
        result = {
            "file": image["file"],
            "device": image["device"],
            "cornerAlphabet": image["cornerAlphabet"],
            "elapsedMs": elapsed_ms,
            "detectedCards": cards,
            "expectedCards": expected_cards,
            "score": score,
            "featureCount": len(features),
            "contourCount": contour_count,
        }
        results.append(result)
        print(json.dumps(result, separators=(",", ":")))

    summary = {
        "model": model_name,
        "pairDistance": PAIR_DISTANCE_PARAM,
        "longSide": LONG_SIDE,
        "images": len(results),
        "initElapsedMs": init_elapsed_ms,
        "exactImages": sum(result["score"]["exact"] for result in results),
        "cardPrecision": ratio(
            sum(result["score"]["correct"] for result in results),
            sum(result["score"]["detected"] for result in results),
        ),
        "cardRecall": ratio(
            sum(result["score"]["correct"] for result in results),
            sum(result["score"]["expected"] for result in results),
        ),
        "meanElapsedMs": round(
            sum(result["elapsedMs"] for result in results) / len(results),
        ),
        "maxElapsedMs": max(result["elapsedMs"] for result in results),
        "groups": {
            name: summarize_results(group)
            for name, group in {
                "jqk": [
                    result
                    for result in results
                    if result["cornerAlphabet"] == "J/Q/K"
                ],
                "vdr": [
                    result
                    for result in results
                    if result["cornerAlphabet"] == "V/D/R"
                ],
                "oneplus": [
                    result
                    for result in results
                    if result["device"] == "oneplus-11-5g"
                ],
                "xiaomi": [
                    result
                    for result in results
                    if result["device"] == "xiaomi-pad-6"
                ],
            }.items()
            if group
        },
    }
    print(json.dumps({"summary": summary}, separators=(",", ":")))


def summarize_results(results):
    return {
        "images": len(results),
        "exactImages": sum(result["score"]["exact"] for result in results),
        "cardPrecision": ratio(
            sum(result["score"]["correct"] for result in results),
            sum(result["score"]["detected"] for result in results),
        ),
        "cardRecall": ratio(
            sum(result["score"]["correct"] for result in results),
            sum(result["score"]["expected"] for result in results),
        ),
    }


if __name__ == "__main__":
    main()
