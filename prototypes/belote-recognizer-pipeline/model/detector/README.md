# Tiny corner-index detector

Custom MIT-licensed PyTorch detector. No Ultralytics or AGPL dependency.

Contract:

- input `images`: float32 RGB `[batch, 3, 384, 384]`, range `[0, 1]`;
- output `boxes`: `[batch, 128, 4]` normalized `xyxy`;
- output `scores`: `[batch, 128]`;
- browser adapter resizes each 960 px recognizer tile to 384 px, filters the
  frozen score threshold, and scales normalized boxes back to the tile.

Training uses scene-level Andrew Tidey train/validation splits. Labels below
90% visible are excluded. The independent GreyWyvern split is accepted only by
the one-shot evaluator after hashes for code, model, config, and Andrew data
have been frozen.

```sh
../../generated/classifier-venv/bin/python train.py \
  --train generated/train \
  --extra-train generated/jackfurby-train \
  --validation generated/validation \
  --epochs 40 --batch-size 12

../../generated/classifier-venv/bin/python evaluate_holdout.py \
  --holdout ../../generated/classifier-holdout
```
