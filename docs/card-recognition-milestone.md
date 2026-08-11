# Card recognition milestone — paused 2026-08-11

## Target

- Browser-only inference.
- `test-data/card-recognition/smoke`: 20/20 exact sets, 336/336 cards, zero
  misses, zero extras.
- Full 32-card photo in at most 5 seconds on a real mobile device after warmup.
- Smoke corpus never used for training, templates, tuning, or sample logic.

## Current result

**Target not reached.** The browser evaluator still uses a declared empty
placeholder: 0/20 exact sets and 0/336 cards. No end-to-end learned model or
real-mobile latency result exists.

Useful preserved work:

- `prototypes/card-recognition`: browser UI, manifest/hash validation, exact-set
  metrics, latency/evolution display, and production/evaluation firewall.
- `prototypes/belote-recognizer-pipeline`: deterministic licensed asset and
  synthetic-data pipeline, browser orchestration contracts, model trainers,
  tests, small rejected models, and validation evidence.
- Untouched smoke corpus remained evaluation-only.

## Why the attempted model was rejected

The two-stage corner detector plus rank/suit classifier did not generalize
enough to justify browser integration.

| Component | Best durable evidence | Required | Verdict |
| --- | ---: | ---: | --- |
| Detector, strict threshold | 2.38% scene recall, 0 FP/image | >=99% recall, <=0.1 FP/image | Fail |
| Detector, useful threshold | 84.99% recall, 3.97 FP/image | same | Fail |
| Classifier, clean synthetic validation | 76.75% exact card | near-perfect end-to-end | Fail |
| Smoke exact-set | 0/20 placeholder | 20/20 | Not integrated |
| Mobile warm latency | unmeasured | <=5 s | Not verified |

Detector evidence:
`prototypes/belote-recognizer-pipeline/model/detector/validation-metrics.json`.
Classifier evidence:
`prototypes/belote-recognizer-pipeline/model/classifier/artifacts/validation-metrics.json`.

The strongest diagnosis was domain mismatch: isolated synthetic corner crops
look unlike detector-aligned corners in cluttered photos. The committed
classifier ONNX is the earlier `tiny-spatial-factorized-v2` baseline; current
classifier training code contains an unfinished later experiment and must not
be assumed compatible with that artifact.

## Data caveat

The real-photo acquisition code and tests are preserved, not the ignored bulk
downloads. Its latest local selection had 152 usable photos and 304 crops, but
strict review rejected the split: 61 accepted cross-split EXIF pairs were at
most 60 seconds apart. Filename families are not capture sessions. Selection
also depends on ignored generated state, so exact recreation is not locked.

Before reuse: add immutable capture-session IDs, keep each session in one
split, and commit a small selection/quarantine lock with source hashes and
rejection reasons.

## Resume point

1. Keep the evaluator and its smoke firewall unchanged.
2. Repair and lock the real-photo split before training.
3. Prefer a direct 32-class corner detector over the failed generic
   detector-to-classifier seam; validate first on external real photos.
4. Integrate only after independent recall/false-positive gates pass.
5. Run the untouched smoke benchmark and a real-device browser latency test.

Generated datasets, caches, virtual environments, intermediate checkpoints,
and build outputs are intentionally excluded from Git.

## Verification at pause

- Evaluator: 4 Node tests; TypeScript/Vite build passed.
- Pipeline: 6 Node tests, self-test, TypeScript/Vite build passed.
- Classifier: 2 Python tests and 1 Node adapter test passed.
- Detector: 3 Python tests passed.
- Photo data: 4 Node and 8 Python tests passed.

These checks validate infrastructure, not recognition accuracy.
