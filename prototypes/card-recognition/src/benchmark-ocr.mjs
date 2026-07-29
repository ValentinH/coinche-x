import { mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";
import sharp from "sharp";
import { createWorker, OEM, PSM } from "tesseract.js";

const prototypeDirectory = resolve(import.meta.dirname, "..");
const repositoryDirectory = resolve(prototypeDirectory, "../..");
const corpusDirectory = resolve(
  repositoryDirectory,
  "test-data/card-recognition/smoke",
);
const manifest = JSON.parse(
  await readFile(resolve(corpusDirectory, "manifest.json"), "utf8"),
);
const cacheDirectory = resolve(prototypeDirectory, ".cache/tesseract");
await mkdir(cacheDirectory, { recursive: true });
const requestedFile = process.argv[2];
const images = requestedFile
  ? manifest.images.filter(({ file }) => file.endsWith(requestedFile))
  : manifest.images;

if (images.length === 0) {
  throw new Error(`Image absente du manifeste : ${requestedFile}`);
}

const worker = await createWorker("eng", OEM.LSTM_ONLY, {
  cachePath: cacheDirectory,
});
await worker.setParameters({
  tessedit_char_whitelist: "78910AJQKVDR",
  tessedit_pageseg_mode: PSM.SPARSE_TEXT,
  user_defined_dpi: "300",
});
const results = [];

for (const image of images) {
  const sourcePath = resolve(corpusDirectory, image.file);
  const preparedImage = await sharp(sourcePath)
    .rotate()
    .resize({
      width: 1600,
      height: 1600,
      fit: "inside",
      withoutEnlargement: true,
    })
    .grayscale()
    .normalise()
    .sharpen()
    .png()
    .toBuffer();
  const startedAt = performance.now();
  const {
    data: { blocks, text },
  } = await worker.recognize(preparedImage, {}, { blocks: true, text: true });
  const elapsedMs = Math.round(performance.now() - startedAt);
  const tokens = text
    .toUpperCase()
    .match(/10|[789AJQKVDR]/g)
    ?.map((rank) => normalizeRank(rank)) ?? [];
  const expectedCards = manifest.cardSets[image.cardSet];
  const expectedRanks = expectedCards.map((card) =>
    normalizeRank(card.slice(0, -1)),
  );
  const score = scoreMultiset(tokens, expectedRanks);
  results.push({ image, elapsedMs, score });

  console.log(
    JSON.stringify({
      file: image.file,
      elapsedMs,
      expectedCount: image.cardCount,
      detectedRanks: tokens,
      expectedRanks,
      score,
      rawText: text.replaceAll(/\s+/g, " ").trim(),
      words: parseWords(blocks),
    }),
  );
}

await worker.terminate();
console.log(
  JSON.stringify({
    summary: {
      images: results.length,
      exactImages: results.filter(({ score }) => score.exact).length,
      rankPrecision: ratio(
        results.reduce((total, { score }) => total + score.correct, 0),
        results.reduce((total, { score }) => total + score.detected, 0),
      ),
      rankRecall: ratio(
        results.reduce((total, { score }) => total + score.correct, 0),
        results.reduce((total, { score }) => total + score.expected, 0),
      ),
      meanElapsedMs: Math.round(
        results.reduce((total, { elapsedMs }) => total + elapsedMs, 0) /
          results.length,
      ),
      maxElapsedMs: Math.max(
        ...results.map(({ elapsedMs }) => elapsedMs),
      ),
    },
  }),
);

function normalizeRank(rank) {
  return { V: "J", D: "Q", R: "K" }[rank] ?? rank;
}

function scoreMultiset(detected, expected) {
  const detectedCounts = countValues(detected);
  const expectedCounts = countValues(expected);
  const correct = [...expectedCounts].reduce(
    (total, [rank, count]) =>
      total + Math.min(count, detectedCounts.get(rank) ?? 0),
    0,
  );

  return {
    exact:
      correct === expected.length && detected.length === expected.length,
    correct,
    detected: detected.length,
    expected: expected.length,
  };
}

function countValues(values) {
  const counts = new Map();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return counts;
}

function ratio(numerator, denominator) {
  return denominator === 0
    ? 0
    : Math.round((numerator / denominator) * 1000) / 1000;
}

function parseWords(blocks) {
  return (blocks ?? []).flatMap((block) =>
    block.paragraphs.flatMap((paragraph) =>
      paragraph.lines.flatMap((line) =>
        line.words.map(({ text, confidence, bbox }) => ({
          text,
          confidence,
          box: bbox,
        })),
      ),
    ),
  );
}
