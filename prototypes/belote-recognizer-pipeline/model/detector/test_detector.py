#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from data import tile_starts
from decode import decode_centers
from model import ExportDetector, TinyCornerDetector


class DetectorTest(unittest.TestCase):
    def test_browser_tile_starts(self):
        self.assertEqual(tile_starts(1536), [0, 576])
        self.assertEqual(tile_starts(2048), [0, 768, 1088])

    def test_tiny_model_and_export_contract(self):
        model = TinyCornerDetector().eval()
        self.assertLess(sum(value.numel() for value in model.parameters()), 250_000)
        with torch.inference_mode():
            boxes, scores = ExportDetector(model)(torch.zeros(2, 3, 384, 384))
        self.assertEqual(tuple(boxes.shape), (2, 128, 4))
        self.assertEqual(tuple(scores.shape), (2, 128))
        self.assertTrue(torch.all((0 <= boxes) & (boxes <= 1)))
        self.assertTrue(torch.all(scores[:, :-1] >= scores[:, 1:]))

    def test_center_decoder(self):
        heatmap = np.zeros((96, 96), dtype=np.float32)
        heatmap[30, 40] = 0.9
        regression = np.zeros((4, 96, 96), dtype=np.float32)
        detections = decode_centers(heatmap, regression, threshold=0.5)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["score"], 0.9)
        left, top, right, bottom = detections[0]["box"]
        self.assertAlmostEqual((left + right) / 2, 40.5)
        self.assertAlmostEqual((top + bottom) / 2, 30.5)


if __name__ == "__main__":
    unittest.main()
