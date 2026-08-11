# Tiny factorized corner classifier

Genuine learned `96×96 RGB → rank logits + suit logits` classifier. Rank order:
`7,8,9,10,J,Q,K,A`; suit order: `clubs,diamonds,hearts,spades`.
Training glyphs `V/D/R` map to canonical `J/Q/K`.
`ort-web-adapter.mjs` provides the batched `CardClassifier` browser adapter.
Use recognizer `cropPadding: 0.06`; crop extraction matches that geometry.

Data firewall:

- Andrew Tidey generated scenes only for training and model selection.
- Different scene seeds and directories enforce scene-level train/validation separation.
- GreyWyvern is accepted only by the final evaluator.
- The final evaluator creates its result exclusively and refuses a second run.
- Project smoke data is never read.

Reproduce from the pipeline root with Python 3.9+:

```sh
python3 -m venv generated/classifier-venv
generated/classifier-venv/bin/pip install -r model/classifier/requirements.txt

node scripts/generate-dataset.mjs \
  --split train --count 64 --seed classifier-andrew-train-v1 \
  --out generated/classifier-train
node scripts/generate-dataset.mjs \
  --split validation --count 32 --seed classifier-andrew-validation-v1 \
  --out generated/classifier-validation

generated/classifier-venv/bin/python model/classifier/extract_crops.py \
  --dataset generated/classifier-train --output generated/classifier-crops/train \
  --role train
generated/classifier-venv/bin/python model/classifier/extract_crops.py \
  --dataset generated/classifier-validation \
  --output generated/classifier-crops/validation --role validation

generated/classifier-venv/bin/python model/classifier/train.py \
  --train-crops generated/classifier-crops/train \
  --validation-crops generated/classifier-crops/validation \
  --output model/classifier/artifacts
```

After architecture, augmentation, and hyperparameters are frozen, run the
holdout exactly once:

```sh
node scripts/generate-dataset.mjs \
  --split holdout --count 64 --seed classifier-greywyvern-final-v1 \
  --out generated/classifier-holdout
generated/classifier-venv/bin/python model/classifier/extract_crops.py \
  --dataset generated/classifier-holdout \
  --output generated/classifier-crops/holdout --role holdout
generated/classifier-venv/bin/python model/classifier/evaluate_holdout.py \
  --model model/classifier/artifacts/classifier.onnx \
  --holdout-crops generated/classifier-crops/holdout \
  --output model/classifier/artifacts/holdout-metrics.json
```

Run unit tests:

```sh
generated/classifier-venv/bin/python -m unittest \
  discover -s model/classifier -p 'test_*.py'
node --test model/classifier/*.test.mjs
```
