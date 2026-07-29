globalThis.Module = {};
const ocr = await import("@paddlejs-models/ocr");

const corpusBase = "/test-data/card-recognition/smoke";
const manifest = await fetch(`${corpusBase}/manifest.json`).then((response) =>
  response.json(),
);
const parameters = new URLSearchParams(location.search);
const requestedFile = parameters.get("file");
const images = parameters.has("all")
  ? manifest.images
  : manifest.images.filter(({ file }) =>
      file.endsWith(requestedFile ?? "xiaomi-pad-6_jqk_16.jpg"),
    );
const status = document.querySelector("#status");
const output = document.querySelector("#results");
const source = document.querySelector("#source");
const results = [];

if (images.length === 0) {
  throw new Error(`Image absente du manifeste : ${requestedFile}`);
}

const initStartedAt = performance.now();
await ocr.init();
const initElapsedMs = Math.round(performance.now() - initStartedAt);
status.textContent = `Modèles chargés en ${initElapsedMs} ms`;

for (const image of images) {
  status.textContent = `Analyse ${results.length + 1}/${images.length} : ${image.file}`;
  await loadImage(source, `${corpusBase}/${image.file}`);
  const startedAt = performance.now();
  const recognition = await ocr.recognize(source);
  const elapsedMs = Math.round(performance.now() - startedAt);
  const tokens = recognition.text
    .flatMap((text) => text.toUpperCase().match(/10|[789AJQKVDR]/g) ?? [])
    .map(normalizeRank);
  const expectedCards = manifest.cardSets[image.cardSet];
  const expectedRanks = expectedCards.map((card) =>
    normalizeRank(card.slice(0, -1)),
  );
  const score = scoreMultiset(tokens, expectedRanks);
  const result = {
    file: image.file,
    elapsedMs,
    expectedCount: image.cardCount,
    detectedText: recognition.text,
    detectedRanks: tokens,
    expectedRanks,
    score,
    points: recognition.points,
  };
  results.push(result);
  output.textContent += `${JSON.stringify(result)}\n`;
}

const summary = {
  images: results.length,
  initElapsedMs,
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
  maxElapsedMs: Math.max(...results.map(({ elapsedMs }) => elapsedMs)),
};
output.textContent += `${JSON.stringify({ summary })}\n`;
status.textContent = "Terminé";
document.title = `DONE ${summary.exactImages}/${summary.images}`;

function loadImage(element, sourceUrl) {
  return new Promise((resolve, reject) => {
    element.onload = resolve;
    element.onerror = reject;
    element.src = sourceUrl;
  });
}

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
    exact: correct === expected.length && detected.length === expected.length,
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
