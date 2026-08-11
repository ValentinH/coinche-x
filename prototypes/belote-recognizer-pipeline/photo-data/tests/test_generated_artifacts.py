import json
import os
import unittest
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "generated" / "photo-source"
SOURCE_ROOT = Path(os.environ.get("PHOTO_DATA_SOURCE", DEFAULT_SOURCE))


def difference_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise AssertionError(f"cannot read {path}")
    resized = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    return resized[:, 1:] > resized[:, :-1]


@unittest.skipUnless(
    (SOURCE_ROOT / "gallery" / "manifest.json").is_file(),
    "generated photo artifacts are not present",
)
class GeneratedArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (SOURCE_ROOT / "gallery" / "manifest.json").read_text()
        )
        cls.provenance = json.loads(
            (SOURCE_ROOT / "provenance.json").read_text()
        )
        cls.crops = [
            json.loads(line)
            for line in (SOURCE_ROOT / "classifier-crops.jsonl")
            .read_text()
            .splitlines()
            if line
        ]

    def test_sources_are_globally_unique(self):
        hashes = [entry["sha256"] for entry in self.manifest["entries"]]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_train_validation_are_near_duplicate_free(self):
        accepted = [
            entry
            for entry in self.manifest["entries"]
            if entry["usableForClassifier"]
        ]
        hashes = {
            entry["id"]: difference_hash(SOURCE_ROOT / entry["localPath"])
            for entry in accepted
        }
        near_duplicates = []
        train = [entry for entry in accepted if entry["split"] == "train"]
        validation = [
            entry for entry in accepted if entry["split"] == "validation"
        ]
        for train_entry in train:
            for validation_entry in validation:
                distance = int(
                    np.count_nonzero(
                        hashes[train_entry["id"]]
                        != hashes[validation_entry["id"]]
                    )
                )
                if distance <= 8:
                    near_duplicates.append(
                        (
                            train_entry["id"],
                            validation_entry["id"],
                            distance,
                        )
                    )
        self.assertEqual(near_duplicates, [])

    def test_all_assets_have_semantically_usable_non_frame_quads(self):
        self.assertGreaterEqual(self.manifest["assetCount"], 128)
        self.assertEqual(
            self.manifest["usableCount"],
            sum(entry["usableForClassifier"] for entry in self.manifest["entries"]),
        )
        unusable = [
            entry["id"]
            for entry in self.manifest["entries"]
            if entry["usableForClassifier"]
            and (
                entry["diagnostics"]["areaFraction"] >= 0.94
                or not entry["diagnostics"].get("semanticUsable", False)
                or not entry["diagnostics"].get(
                    "topLeftIndexStructureUsable", False
                )
                or not entry["diagnostics"].get(
                    "bottomRightIndexStructureUsable", False
                )
            )
        ]
        self.assertEqual(unusable, [])
        covered = {
            (entry["label"], entry["split"])
            for entry in self.manifest["entries"]
            if entry["usableForClassifier"]
        }
        labels = {entry["label"] for entry in self.manifest["entries"]}
        missing = {
            (label, split)
            for label in labels
            for split in ("train", "validation")
            if (label, split) not in covered
        }
        self.assertEqual(missing, set())

    def test_both_corner_crops_have_edges(self):
        self.assertEqual(len(self.crops), self.manifest["usableCount"] * 2)
        corners_by_source = defaultdict(set)
        edge_free = []
        for crop in self.crops:
            corners_by_source[crop["sourceSha256"]].add(crop.get("corner"))
            image = cv2.imread(
                str(SOURCE_ROOT / crop["path"]), cv2.IMREAD_GRAYSCALE
            )
            if image is None or cv2.countNonZero(cv2.Canny(image, 50, 150)) == 0:
                edge_free.append(crop["id"])
        self.assertEqual(edge_free, [])
        self.assertTrue(
            all(
                corners == {"topLeft", "bottomRight"}
                for corners in corners_by_source.values()
            )
        )

    def test_label_consensus_rejects_mislabeled_card_without_rejecting_true_alternate(self):
        by_id = {entry["id"]: entry for entry in self.manifest["entries"]}
        rejected_by_hash = {
            entry["sha256"]: entry
            for entry in self.provenance["rejectedLabelEntries"]
        }
        for sha256 in (
            "26c711641efa9a4ad26bdf309727d5964fbef1357e2a3e09c46471c9bd94675f",
            "a20befdbd6a7e545c4ee79aca79289ea05233f54d72753fe35891ca2c40d86f6",
        ):
            self.assertEqual(
                rejected_by_hash[sha256]["rejection"]["method"],
                "flagged-label-mismatch",
            )
            self.assertEqual(
                rejected_by_hash[sha256]["rejection"]["inferredLabel"],
                "8-clubs",
            )

        replacement = by_id["8-spades--march-20240314"]
        self.assertFalse(replacement["usableForClassifier"])
        self.assertEqual(
            replacement["method"],
            "flagged-label-mismatch",
        )

        true_alternate = by_id["8-spades--img-dated"]
        self.assertTrue(true_alternate["usableForClassifier"])


if __name__ == "__main__":
    unittest.main()
