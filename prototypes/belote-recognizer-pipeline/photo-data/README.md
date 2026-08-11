# Real-photo training source

Pinned source: `drFarid/French-Playing-Cards` at
`8931a981f24b78d7dc0e528f42a6e9d784ad23ea` (MIT in immutable Hugging Face
metadata).

`source.json` maps French labels to the 32-card Belote deck. Acquisition chooses
the smallest hash-unique photo in six filename-family buckets per class. This
split is provisional: EXIF review found likely adjacent captures across train
and validation. Do not train from it until immutable session IDs and a selection
lock replace the filename buckets.

```sh
node photo-data/acquire.mjs
python3 -m pip install --target /tmp/coinche-photo-python -r photo-data/requirements.txt
PYTHONPATH=/tmp/coinche-photo-python python3 photo-data/preprocess.py
python3 -m http.server 4180 --directory generated/photo-source/gallery
```

If the semantic gate flags sources, advance only those IDs within their pinned
filename family, then preprocess again:

```sh
node photo-data/acquire.mjs --replace-failed
PYTHONPATH=/tmp/coinche-photo-python python3 photo-data/preprocess.py
```

Bulk source and derived data live under ignored `generated/photo-source`.
`provenance.json` records every immutable URL, expected/actual SHA-256, byte
count, label, provisional `captureFamily` filename bucket, and split.
Preprocessing detects a generic card-shaped rectangle, rectifies perspective,
requires compact rank/suit
structure in both outer corners, emits two normalized 64×64 index crops, and
flags rather than silently trusting its geometry fallback.
