from __future__ import annotations

import unittest

import torch

from labels import canonical_rank, validate_factorized_label
from model import INPUT_SIZE, TinyFactorizedClassifier, parameter_count


class ClassifierTest(unittest.TestCase):
    def test_french_aliases_are_canonical(self) -> None:
        self.assertEqual(canonical_rank("V"), "J")
        self.assertEqual(canonical_rank("D"), "Q")
        self.assertEqual(canonical_rank("R"), "K")
        validate_factorized_label("J", "clubs", "V")

    def test_factorized_output_contract_and_budget(self) -> None:
        model = TinyFactorizedClassifier().eval()
        with torch.inference_mode():
            (
                rank_logits,
                suit_logits,
                index_logit,
                card_logits,
                orientation_logits,
            ) = model(torch.zeros(3, 3, INPUT_SIZE, INPUT_SIZE))
        self.assertEqual(tuple(rank_logits.shape), (3, 8))
        self.assertEqual(tuple(suit_logits.shape), (3, 4))
        self.assertEqual(tuple(index_logit.shape), (3,))
        self.assertEqual(tuple(card_logits.shape), (3, 32))
        self.assertEqual(tuple(orientation_logits.shape), (3, 4))
        self.assertLess(parameter_count(model), 2_000_000)


if __name__ == "__main__":
    unittest.main()
