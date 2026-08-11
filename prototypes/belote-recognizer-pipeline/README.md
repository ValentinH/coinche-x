# Belote learned-recognizer pipeline slice

Reproducible data + browser integration seam for the chosen two-stage model:

`full photo → 1536–1920 px working image → overlapping tiles → corner-index detector → index crops → factorized rank/suit classifier → NMS + 32-card dedupe`

No OCR/KNN/template recognizer. Rejected ONNX candidates and their metrics are
kept as failure evidence. They are not wired into the browser evaluator.

Project paused. See [the milestone](../../docs/card-recognition-milestone.md).

## What is judgeable now

- 32 canonical cards: `7 8 9 10 J Q K A × ♣ ♦ ♥ ♠`.
- English `J/Q/K` and generated French `V/D/R`; both map to canonical
  `J/Q/K`.
- Hash-pinned CC0 training deck. Different public-domain holdout deck.
- Deterministic 1536×2048 JPEG scenes: 1–32 cards, dense 24–32 scenes,
  rotation, scale, shear, overlap, partial visibility, JPEG variation,
  table/paper hard negatives.
- COCO corner-index detection labels plus JSONL factorized classifier labels.
- Visibility-aware labels; covered indices below 52% omitted.
- Browser-only full-photo orchestration in `src/recognizer.ts`. Learned
  detector/classifier backends implement `src/model-contract.ts`.
- 1920 px cap, overlapping tiling, cross-tile NMS, one batched classifier call,
  canonical dedupe, timing diagnostics, 5 s warm-budget flag.
- Model manifest rejects combined detector + classifier artifacts above 8 MiB.

## Run

Requires Node 22+, ImageMagick 7, and `unzip`.

```sh
pnpm assets:acquire
pnpm test
pnpm dataset:train
pnpm dataset:validation
pnpm dataset:holdout
pnpm dataset:verify
```

Generated datasets are ignored. Each contains:

- `images/*.jpg`
- `dataset.json`: provenance, scene/card metadata, visible corner polygons
- `annotations.coco.json`: single `corner-index` detection class
- `classifier-labels.jsonl`: canonical rank, suit, visible glyph/alphabet
- `preview.jpg`
- `dataset.sha256`

The first 32 frames deliberately contain 1 through 32 cards. The verifier
requires all 32 canonical identities, `J/Q/K/V/D/R`, and dense scenes once a
split has at least 32 frames.

## Model export contract

Detector input: square RGB tiles. Outputs named `[batch, candidates, 4]`
normalized `xyxy` boxes and `[batch, candidates]` scores.

Classifier input: square RGB corner crops. Outputs rank logits in
`7,8,9,10,J,Q,K,A` order and suit logits in
`clubs,diamonds,hearts,spades` order. Visual `V/D/R` remain training glyphs;
outputs stay canonical.

Run both ONNX models in an ORT Web/WASM worker and adapt results to
`CornerDetector` / `CardClassifier`. `BeloteRecognizer.recognize()` accepts a
browser `CanvasImageSource` (including `ImageBitmap`) and returns full-photo
coordinates.

## Honest remaining gate

Replace the failed detector/classifier approach, wire the selected model into
the evaluator, then measure exact card set and real-mobile latency. This slice
does not claim real-photo recognition or the 5 s target.

See `provenance/README.md` and `provenance/sources.json`. Data firewall:
project smoke images/truth are never inputs to acquisition, generation,
thresholds, templates, or validation.
