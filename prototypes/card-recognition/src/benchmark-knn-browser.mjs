import cvModule from "@techstark/opencv-js";

const corpusBase = "/test-data/card-recognition/smoke";
const modelUrl =
  "/prototypes/card-recognition/.cache/fvannee-model.bin";
const manifest = await fetch(`${corpusBase}/manifest.json`).then((response) =>
  response.json(),
);
const parameters = new URLSearchParams(location.search);
const requestedFile = parameters.get("file");
const longSide = Number(parameters.get("longSide") ?? 2000);
const images = parameters.has("all")
  ? manifest.images
  : manifest.images.filter(({ file }) =>
      file.endsWith(requestedFile ?? "oneplus-11-5g_jqk_04.jpg"),
    );
const status = document.querySelector("#status");
const output = document.querySelector("#results");
const source = document.querySelector("#source");
const results = [];

if (images.length === 0) {
  throw new Error(`Image absente du manifeste : ${requestedFile}`);
}

const initStartedAt =
  globalThis.benchmarkNavigationStartedAt ?? performance.now();
const [cv, model] = await Promise.all([getOpenCv(), loadModel(modelUrl)]);
const initElapsedMs = Math.round(performance.now() - initStartedAt);
status.textContent = `OpenCV et ${model.count} gabarits chargés en ${initElapsedMs} ms`;

for (const image of images) {
  status.textContent = `Analyse ${results.length + 1}/${images.length} : ${image.file}`;
  await loadImage(source, `${corpusBase}/${image.file}`);
  const startedAt = performance.now();
  const detection = detectCards(source, cv, model);
  const elapsedMs = Math.round(performance.now() - startedAt);
  const expectedCards = image.cards;
  const score = scoreCards(detection.cards, expectedCards);
  const result = {
    file: image.file,
    device: image.device,
    cornerAlphabet: image.cornerAlphabet,
    elapsedMs,
    detectedCards: detection.cards,
    expectedCards,
    score,
    featureCount: detection.featureCount,
    contourCount: detection.contourCount,
    ...(parameters.has("debug") ? { features: detection.features } : {}),
  };
  results.push(result);
  output.textContent += `${JSON.stringify(result)}\n`;
}

const summary = {
  images: results.length,
  initElapsedMs,
  exactImages: results.filter(({ score }) => score.exact).length,
  cardPrecision: ratio(
    results.reduce((total, { score }) => total + score.correct, 0),
    results.reduce((total, { score }) => total + score.detected, 0),
  ),
  cardRecall: ratio(
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

async function getOpenCv() {
  if (cvModule instanceof Promise) {
    return cvModule;
  }
  if (cvModule.Mat) {
    return cvModule;
  }
  return new Promise((resolve) => {
    cvModule.onRuntimeInitialized = () => resolve(cvModule);
  });
}

async function loadModel(url) {
  const buffer = await fetch(url).then((response) => response.arrayBuffer());
  const view = new DataView(buffer);
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "CFK1") {
    throw new Error("Modèle KNN invalide");
  }
  const count = view.getUint16(4, true);
  const sampleSize = view.getUint16(6, true);
  const black = [];
  const red = [];
  let offset = 8;
  for (let index = 0; index < count; index += 1) {
    const label = String.fromCharCode(view.getUint8(offset));
    offset += 1;
    const sample = new Uint8Array(buffer, offset, sampleSize);
    offset += sampleSize;
    const row = { label, sample };
    if (!"DH".includes(label)) {
      black.push(row);
    }
    if (!"SC".includes(label)) {
      red.push(row);
    }
  }
  return { black, count, red, sampleSize };
}

function detectCards(image, cv, model) {
  const original = cv.imread(image);
  const scale = Math.min(1, longSide / Math.max(original.cols, original.rows));
  const color = new cv.Mat();
  cv.resize(
    original,
    color,
    new cv.Size(0, 0),
    scale,
    scale,
    cv.INTER_AREA,
  );
  original.delete();
  const coefficients = cv.matFromArray(
    1,
    4,
    cv.CV_32F,
    [-1, 1.5, 0.5, 0],
  );
  const transformed = new cv.Mat();
  cv.transform(color, transformed, coefficients);
  coefficients.delete();
  const thresholded = new cv.Mat();
  cv.adaptiveThreshold(
    transformed,
    thresholded,
    255,
    cv.ADAPTIVE_THRESH_MEAN_C,
    cv.THRESH_BINARY,
    101,
    40,
  );
  transformed.delete();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(
    thresholded,
    contours,
    hierarchy,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE,
  );
  hierarchy.delete();
  const contourCount = contours.size();
  const features = [];
  const minimumArea =
    (thresholded.rows * thresholded.cols) / 6000;
  for (let index = 0; index < contours.size(); index += 1) {
    const contour = contours.get(index);
    const rectangle = cv.minAreaRect(contour);
    contour.delete();
    let { width, height } = rectangle.size;
    const dimensionRatio = width / height;
    if (
      width * height <= minimumArea ||
      dimensionRatio >= 3.5 ||
      dimensionRatio <= 1 / 3.5
    ) {
      continue;
    }
    let angle = rectangle.angle;
    if (width > height) {
      [width, height] = [height, width];
      angle += 90;
    }
    const region = subimage(
      cv,
      thresholded,
      rectangle.center,
      angle,
      Math.max(1, Math.round(width)),
      Math.max(1, Math.round(height)),
    );
    const pixel = color.ucharPtr(
      clamp(Math.round(rectangle.center.y), 0, color.rows - 1),
      clamp(Math.round(rectangle.center.x), 0, color.cols - 1),
    );
    const rows = classifyColor(pixel) === "red" ? model.red : model.black;
    const classification = classifyRegion(cv, region, rows, model.sampleSize);
    region.delete();
    if (classification == null) {
      continue;
    }
    features.push({
      angle: classification.rotated ? (angle + 180) % 360 : angle,
      center: rectangle.center,
      feature: classification.label,
      size: { width, height },
    });
  }
  contours.delete();
  thresholded.delete();
  color.delete();
  return {
    cards: pairFeatures(features),
    features,
    featureCount: features.length,
    contourCount,
  };
}

function subimage(cv, image, center, angle, width, height) {
  const radians = (angle * Math.PI) / 180;
  const vectorX = [Math.cos(radians), Math.sin(radians)];
  const vectorY = [-Math.sin(radians), Math.cos(radians)];
  const sourceX =
    center.x - vectorX[0] * (width / 2) - vectorY[0] * (height / 2);
  const sourceY =
    center.y - vectorX[1] * (width / 2) - vectorY[1] * (height / 2);
  const mapping = cv.matFromArray(2, 3, cv.CV_32F, [
    vectorX[0],
    vectorY[0],
    sourceX,
    vectorX[1],
    vectorY[1],
    sourceY,
  ]);
  const result = new cv.Mat();
  cv.warpAffine(
    image,
    result,
    mapping,
    new cv.Size(width, height),
    cv.WARP_INVERSE_MAP,
    cv.BORDER_REPLICATE,
    new cv.Scalar(0),
  );
  mapping.delete();
  return result;
}

function classifyColor([red, green, blue]) {
  const references = [
    ["black", 0, 0, 0],
    ["red", 255, 0, 0],
    ["red", 255, 128, 0],
  ];
  return references.reduce(
    (best, [name, referenceRed, referenceGreen, referenceBlue]) => {
      const distance =
        (red - referenceRed) ** 2 +
        (green - referenceGreen) ** 2 +
        (blue - referenceBlue) ** 2;
      return distance < best.distance ? { distance, name } : best;
    },
    { distance: Number.POSITIVE_INFINITY, name: "black" },
  ).name;
}

function classifyRegion(cv, region, rows, sampleSize) {
  const side = Math.sqrt(sampleSize);
  const resized = new cv.Mat();
  cv.resize(region, resized, new cv.Size(side, side));
  let best = nearest(resized.data, rows);
  const flipped = new cv.Mat();
  cv.flip(resized, flipped, -1);
  const inverted = nearest(flipped.data, rows);
  resized.delete();
  flipped.delete();
  if (inverted.distance < best.distance) {
    best = { ...inverted, rotated: true };
  }
  const dimensionRatio = region.rows / region.cols;
  const suitShapeMatches =
    !"CDHS".includes(best.label) ||
    dimensionRatio < 2 ||
    1 / dimensionRatio > 2;
  return best.distance < 7000 * sampleSize && suitShapeMatches ? best : null;
}

function nearest(sample, rows) {
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestLabel = "";
  for (const row of rows) {
    let distance = 0;
    for (let index = 0; index < sample.length; index += 1) {
      const difference = sample[index] - row.sample[index];
      distance += difference * difference;
      if (distance >= bestDistance) {
        break;
      }
    }
    if (distance < bestDistance) {
      bestDistance = distance;
      bestLabel = row.label;
    }
  }
  return {
    distance: bestDistance,
    label: bestLabel,
    rotated: false,
  };
}

function pairFeatures(features) {
  const cards = [];
  for (const rankFeature of features) {
    if (!"23456789TJQKA".includes(rankFeature.feature)) {
      continue;
    }
    const maximumDistance =
      1.3 * Math.max(rankFeature.size.width, rankFeature.size.height) ** 2;
    let best;
    for (const suitFeature of features) {
      if (!"CDHS".includes(suitFeature.feature)) {
        continue;
      }
      const offsetX = suitFeature.center.x - rankFeature.center.x;
      const offsetY = suitFeature.center.y - rankFeature.center.y;
      if (offsetX ** 2 + offsetY ** 2 >= maximumDistance) {
        continue;
      }
      const lineAngle =
        ((((Math.atan2(offsetY, offsetX) % (2 * Math.PI)) +
          2 * Math.PI) %
          (2 * Math.PI)) /
          Math.PI) *
        180;
      const angleDifference =
        (lineAngle + 360 - rankFeature.angle) % 360;
      const angleError = Math.min(
        Math.abs(angleDifference - 90),
        Math.abs(angleDifference - 270),
      );
      if (angleError < 35 && (best == null || angleError < best.error)) {
        const rank =
          rankFeature.feature === "9" && angleDifference > 180
            ? "6"
            : rankFeature.feature;
        best = {
          card: `${rank === "T" ? "10" : rank}${suitFeature.feature}`,
          error: angleError,
        };
      }
    }
    if (best != null) {
      cards.push(best.card);
    }
  }
  return [...new Set(cards)].sort();
}

function scoreCards(detected, expected) {
  const detectedCounts = countValues(detected);
  const expectedCounts = countValues(expected);
  const correct = [...expectedCounts].reduce(
    (total, [card, count]) =>
      total + Math.min(count, detectedCounts.get(card) ?? 0),
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

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function loadImage(element, sourceUrl) {
  return new Promise((resolve, reject) => {
    element.onload = resolve;
    element.onerror = reject;
    element.src = sourceUrl;
  });
}
