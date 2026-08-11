import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { CARDS, FRENCH_FACE_GLYPH, RANKS, SUITS } from "./lib/cards.mjs";

const directories = process.argv.slice(2).map((directory) => resolve(directory));
if (directories.length === 0) {
  throw new Error("Usage: node scripts/verify-dataset.mjs <dataset> [...]");
}

function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function verifyChecksums(directory) {
  const lines = (await readFile(join(directory, "dataset.sha256"), "utf8"))
    .trim()
    .split("\n");
  const hashes = new Map();
  for (const line of lines) {
    const match = /^([a-f0-9]{64}) {2}(.+)$/.exec(line);
    if (!match) throw new Error(`${directory}: malformed checksum line`);
    const [, expected, relativePath] = match;
    const data = await readFile(join(directory, relativePath));
    const actual = sha256(data);
    if (actual !== expected) {
      throw new Error(`${directory}: checksum mismatch for ${relativePath}`);
    }
    hashes.set(relativePath, actual);
  }
  return hashes;
}

function assertBox(box, width, height, context) {
  const values = [box.x, box.y, box.width, box.height];
  if (!values.every(Number.isFinite)) throw new Error(`${context}: non-finite box`);
  if (
    box.x < 0 ||
    box.y < 0 ||
    box.width <= 0 ||
    box.height <= 0 ||
    box.x + box.width > width + 0.001 ||
    box.y + box.height > height + 0.001
  ) {
    throw new Error(`${context}: out-of-frame box`);
  }
}

const summaries = [];
const imageHashesBySplit = new Map();

for (const directory of directories) {
  const dataset = await readJson(join(directory, "dataset.json"));
  const coco = await readJson(join(directory, "annotations.coco.json"));
  const classifierLines = (await readFile(
    join(directory, "classifier-labels.jsonl"),
    "utf8",
  ))
    .trim()
    .split("\n")
    .filter(Boolean)
    .map(JSON.parse);
  const checksums = await verifyChecksums(directory);

  if (dataset.schemaVersion !== 1) throw new Error(`${directory}: schema`);
  if (!["train", "validation", "holdout"].includes(dataset.split)) {
    throw new Error(`${directory}: split`);
  }
  const shouldBeHoldout = dataset.split === "holdout";
  if (
    dataset.source.role !== (shouldBeHoldout ? "holdout-only" : "training") ||
    dataset.source.path.startsWith(
      shouldBeHoldout ? "training/" : "holdout/",
    )
  ) {
    throw new Error(`${directory}: source firewall violation`);
  }
  if (
    Math.max(dataset.width, dataset.height) < 1600 ||
    Math.max(dataset.width, dataset.height) > 2048
  ) {
    throw new Error(`${directory}: expected 1600–2048 px full-photo scenes`);
  }
  if (dataset.frames.length === 0) throw new Error(`${directory}: empty`);
  if (coco.images.length !== dataset.frames.length) {
    throw new Error(`${directory}: COCO image count`);
  }

  const annotations = dataset.frames.flatMap((frame) => frame.annotations);
  if (
    annotations.length !== coco.annotations.length ||
    annotations.length !== classifierLines.length
  ) {
    throw new Error(`${directory}: annotation count mismatch`);
  }

  for (const [frameIndex, frame] of dataset.frames.entries()) {
    if (frame.cardCount !== 1 + (frameIndex % 32)) {
      throw new Error(`${directory}: non-deterministic card-count curriculum`);
    }
    if (frame.cardCount >= 24 && frame.mode !== "dense") {
      throw new Error(`${directory}: missing dense-scene mode`);
    }
    await stat(join(directory, frame.image));
    if (!checksums.has(frame.image)) {
      throw new Error(`${directory}: image absent from checksum manifest`);
    }
    for (const annotation of frame.annotations) {
      assertBox(annotation.bbox, frame.width, frame.height, frame.image);
      if (!RANKS.includes(annotation.card.rank) || !SUITS.includes(annotation.card.suit)) {
        throw new Error(`${directory}: non-canonical card`);
      }
      const expectedFrenchGlyph = FRENCH_FACE_GLYPH[annotation.card.rank];
      if (
        annotation.alphabet === "french" &&
        annotation.glyph !== expectedFrenchGlyph
      ) {
        throw new Error(`${directory}: French alias mismatch`);
      }
      if (!annotation.sourceAsset.startsWith(dataset.source.path)) {
        throw new Error(`${directory}: annotation crossed source firewall`);
      }
      if (
        !Number.isFinite(annotation.visibleFraction) ||
        annotation.visibleFraction < dataset.generator.minimumVisibleFraction
      ) {
        throw new Error(`${directory}: invalid visibility label`);
      }
    }
  }

  if (dataset.frames.length >= 32) {
    const expectedCards = new Set(CARDS.map((card) => card.id));
    const actualCards = new Set(dataset.coverage.canonicalCards);
    if (
      expectedCards.size !== actualCards.size ||
      [...expectedCards].some((card) => !actualCards.has(card))
    ) {
      throw new Error(`${directory}: incomplete 32-card coverage`);
    }
    const requiredGlyphs = new Set(["7", "8", "9", "10", "J", "Q", "K", "V", "D", "R", "A"]);
    if (dataset.split !== "holdout") {
      for (const glyph of requiredGlyphs) {
        if (!dataset.coverage.glyphs.includes(glyph)) {
          throw new Error(`${directory}: missing glyph ${glyph}`);
        }
      }
    }
    const counts = new Set(dataset.frames.slice(0, 32).map((frame) => frame.cardCount));
    if (counts.size !== 32) throw new Error(`${directory}: missing 1–32 scene counts`);
  }

  const imageHashes = [...checksums]
    .filter(([path]) => path.startsWith("images/"))
    .map(([, hash]) => hash);
  imageHashesBySplit.set(dataset.split, imageHashes);
  summaries.push({
    split: dataset.split,
    frames: dataset.frames.length,
    annotations: annotations.length,
    cards: dataset.coverage.canonicalCards.length,
    glyphs: dataset.coverage.glyphs.join(","),
    source: dataset.source.id,
  });
}

const splits = [...imageHashesBySplit];
for (let left = 0; left < splits.length; left += 1) {
  for (let right = left + 1; right < splits.length; right += 1) {
    const [leftName, leftHashes] = splits[left];
    const [rightName, rightHashes] = splits[right];
    const rightSet = new Set(rightHashes);
    if (leftHashes.some((hash) => rightSet.has(hash))) {
      throw new Error(`${leftName}/${rightName}: duplicate generated image`);
    }
  }
}

for (const summary of summaries) {
  console.log(
    `${summary.split}: ${summary.frames} frames, ${summary.annotations} indices, ${summary.cards}/32 cards, glyphs=${summary.glyphs}, source=${summary.source}`,
  );
}
