import assert from "node:assert/strict";
import test from "node:test";
import {
  CARDS,
  canonicalRankForGlyph,
  cardId,
} from "../src/domain.ts";
import {
  MAX_COMBINED_MODEL_BYTES,
  validateModelManifest,
} from "../src/model-contract.ts";
import {
  deduplicateCards,
  nonMaximumSuppression,
  tileStarts,
} from "../src/recognizer.ts";

test("Belote domain has exactly 32 canonical cards", () => {
  assert.equal(CARDS.length, 32);
  assert.equal(new Set(CARDS.map(cardId)).size, 32);
});

test("English and French face glyphs share canonical ranks", () => {
  assert.equal(canonicalRankForGlyph("J"), "J");
  assert.equal(canonicalRankForGlyph("V"), "J");
  assert.equal(canonicalRankForGlyph("Q"), "Q");
  assert.equal(canonicalRankForGlyph("D"), "Q");
  assert.equal(canonicalRankForGlyph("K"), "K");
  assert.equal(canonicalRankForGlyph("R"), "K");
});

test("tiling covers the far edge without a tiny final tile", () => {
  assert.deepEqual(tileStarts(1920, 960, 192), [0, 768, 960]);
  assert.deepEqual(tileStarts(700, 960, 192), [0]);
});

test("cross-tile NMS merges the same global corner", () => {
  const kept = nonMaximumSuppression(
    [
      {
        box: { x: 800, y: 50, width: 80, height: 120 },
        confidence: 0.8,
        tileX: 0,
        tileY: 0,
      },
      {
        box: { x: 32, y: 50, width: 80, height: 120 },
        confidence: 0.9,
        tileX: 768,
        tileY: 0,
      },
    ],
    0.35,
  );
  assert.equal(kept.length, 1);
  assert.equal(kept[0].confidence, 0.9);
});

test("duplicate corner readings collapse by physical card identity", () => {
  const common = {
    card: { rank: "J", suit: "hearts" },
    detectorConfidence: 0.9,
    rankConfidence: 0.9,
    suitConfidence: 0.9,
  };
  const result = deduplicateCards([
    {
      ...common,
      box: { x: 10, y: 10, width: 20, height: 30 },
      confidence: 0.7,
    },
    {
      ...common,
      box: { x: 100, y: 100, width: 20, height: 30 },
      confidence: 0.8,
    },
  ]);
  assert.equal(result.length, 1);
  assert.equal(result[0].confidence, 0.8);
});

test("model manifest enforces factorization and mobile footprint", () => {
  const artifact = {
    url: "/model.onnx",
    sha256: "a".repeat(64),
    bytes: MAX_COMBINED_MODEL_BYTES / 2,
  };
  assert.doesNotThrow(() =>
    validateModelManifest({
      schemaVersion: 1,
      detector: {
        ...artifact,
        inputSize: 640,
        outputBoxes: "boxes",
        outputScores: "scores",
      },
      classifier: {
        ...artifact,
        inputSize: 96,
        outputRankLogits: "rank_logits",
        outputSuitLogits: "suit_logits",
      },
      rankClasses: ["7", "8", "9", "10", "J", "Q", "K", "A"],
      suitClasses: ["clubs", "diamonds", "hearts", "spades"],
    }),
  );
});
