import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import (  # noqa: E402
    contour_candidates,
    corner_crop,
    semantic_diagnostics,
    warp_card,
)


class PreprocessTest(unittest.TestCase):
    def synthetic_photo(self):
        image = Image.new("RGB", (900, 700), "#7f9a68")
        card = Image.new("RGB", (300, 430), "white")
        draw = ImageDraw.Draw(card)
        draw.rounded_rectangle((1, 1, 298, 428), 20, outline="#dadada", width=4)
        draw.text((24, 20), "10", fill="#d91e2d")
        draw.ellipse((35, 75, 72, 112), fill="#d91e2d")
        rotated = card.rotate(17, expand=True, fillcolor="#7f9a68")
        image.paste(rotated, (260, 110))
        return np.asarray(image)

    def test_generic_rectangle_and_corner_crop(self):
        image = self.synthetic_photo()
        candidates = contour_candidates(image)
        self.assertTrue(candidates)
        card = warp_card(image, candidates[0][1])
        self.assertGreater(card.shape[0], card.shape[1])
        self.assertEqual(corner_crop(card).shape, (64, 64, 3))
        self.assertEqual(corner_crop(card, bottom_right=True).shape, (64, 64, 3))

    def test_output_is_deterministic(self):
        image = self.synthetic_photo()
        first = corner_crop(warp_card(image, contour_candidates(image)[0][1]))
        second = corner_crop(warp_card(image, contour_candidates(image)[0][1]))
        self.assertTrue(np.array_equal(first, second))

    def test_semantic_gate_requires_outer_corner_index_structure(self):
        indexed_card = np.full((717, 512, 3), 245, dtype=np.uint8)
        cv2.putText(
            indexed_card,
            "10",
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (15, 15, 15),
            4,
            cv2.LINE_AA,
        )
        cv2.circle(indexed_card, (62, 126), 23, (15, 15, 15), -1)
        indexed_card[-220:, -220:] = cv2.rotate(
            indexed_card[:220, :220],
            cv2.ROTATE_180,
        )

        central_pips = np.full_like(indexed_card, 245)
        cv2.circle(central_pips, (168, 150), 34, (15, 15, 15), -1)
        cv2.circle(
            central_pips,
            (512 - 168, 717 - 150),
            34,
            (15, 15, 15),
            -1,
        )

        self.assertTrue(semantic_diagnostics(indexed_card)["semanticUsable"])
        self.assertFalse(semantic_diagnostics(central_pips)["semanticUsable"])


if __name__ == "__main__":
    unittest.main()
