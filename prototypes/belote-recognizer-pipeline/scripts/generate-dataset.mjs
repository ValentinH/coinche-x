import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdtempSync,
  rmSync,
} from "node:fs";
import {
  mkdir,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { CARDS, FRENCH_FACE_GLYPH } from "./lib/cards.mjs";
import { createRandom } from "./lib/random.mjs";
import { frenchCornerOverlay } from "./lib/svg.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const WIDTH = 1536;
const HEIGHT = 2048;
const DEFAULT_SEEDS = {
  train: "coinche-train-v1",
  validation: "coinche-validation-v1",
  holdout: "coinche-holdout-v1",
};
const SOURCE_BY_SPLIT = {
  train: {
    id: "andrew-tidey-cards-pack",
    role: "training",
    path: "training/sources/andrew-tidey",
  },
  validation: {
    id: "andrew-tidey-cards-pack",
    role: "training",
    path: "training/sources/andrew-tidey",
  },
  holdout: {
    id: "greywyvern-cardset",
    role: "holdout-only",
    path: "holdout/sources/greywyvern",
  },
};

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Expected --key value, received ${key ?? "<end>"}`);
    }
    values[key.slice(2)] = value;
  }
  const split = values.split ?? "train";
  if (!["train", "validation", "holdout"].includes(split)) {
    throw new Error("--split must be train, validation, or holdout");
  }
  const count = Number(values.count ?? 32);
  if (!Number.isSafeInteger(count) || count < 1 || count > 100_000) {
    throw new Error("--count must be an integer between 1 and 100000");
  }
  const output = resolve(root, values.out ?? `generated/${split}`);
  if (output === root || output === "/" || basename(output).length === 0) {
    throw new Error("Refusing broad output path");
  }
  return {
    split,
    count,
    output,
    seed: values.seed ?? DEFAULT_SEEDS[split],
  };
}

function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

function pngDimensions(data) {
  if (data.toString("ascii", 1, 4) !== "PNG") throw new Error("Expected PNG source");
  return { width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
}

function matrixForCard(card, random, mode, index, count) {
  let targetWidth;
  if (mode === "dense") {
    const columns = Math.ceil(Math.sqrt((count * WIDTH) / HEIGHT));
    const rows = Math.ceil(count / columns);
    targetWidth = Math.min(WIDTH / columns, HEIGHT / rows / 1.36) * random.between(0.58, 0.78);
  } else if (count <= 3) {
    targetWidth = random.between(270, 420);
  } else {
    targetWidth = random.between(180, 300);
  }
  const scaleX = targetWidth / card.width;
  const scaleY = scaleX * random.between(0.96, 1.04);
  const shear = random.between(-0.12, 0.12);
  const angle =
    mode === "dense"
      ? random.between(-0.5, 0.5)
      : random.between(-Math.PI, Math.PI);
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const a = cosine * scaleX;
  const b = sine * scaleX;
  const c = (cosine * shear - sine) * scaleY;
  const d = (sine * shear + cosine) * scaleY;

  let centerX;
  let centerY;
  if (mode === "fan") {
    const fanAngle = count === 1 ? 0 : (index / (count - 1) - 0.5) * Math.PI;
    centerX = WIDTH * 0.5 + Math.sin(fanAngle) * random.between(180, 390);
    centerY = HEIGHT * 0.5 + Math.cos(fanAngle) * random.between(80, 270);
  } else if (mode === "dense") {
    const columns = Math.ceil(Math.sqrt((count * WIDTH) / HEIGHT));
    const rows = Math.ceil(count / columns);
    const column = index % columns;
    const row = Math.floor(index / columns);
    const cellWidth = WIDTH / columns;
    const cellHeight = HEIGHT / rows;
    centerX = (column + 0.5) * cellWidth + random.between(-0.13, 0.13) * cellWidth;
    centerY = (row + 0.5) * cellHeight + random.between(-0.13, 0.13) * cellHeight;
  } else {
    const horizontalMargin = Math.max(180, targetWidth * 0.78);
    const verticalMargin = Math.max(220, targetWidth * 0.95);
    centerX = random.between(horizontalMargin, WIDTH - horizontalMargin);
    centerY = random.between(verticalMargin, HEIGHT - verticalMargin);
  }

  return {
    a,
    b,
    c,
    d,
    e: centerX - a * card.width * 0.5 - c * card.height * 0.5,
    f: centerY - b * card.width * 0.5 - d * card.height * 0.5,
  };
}

function transformPoint(matrix, point) {
  return {
    x: matrix.a * point.x + matrix.c * point.y + matrix.e,
    y: matrix.b * point.x + matrix.d * point.y + matrix.f,
  };
}

function transformRectangle(matrix, x, y, width, height) {
  return [
    transformPoint(matrix, { x, y }),
    transformPoint(matrix, { x: x + width, y }),
    transformPoint(matrix, { x: x + width, y: y + height }),
    transformPoint(matrix, { x, y: y + height }),
  ];
}

function polygonBox(points) {
  const left = Math.max(0, Math.min(...points.map((point) => point.x)));
  const top = Math.max(0, Math.min(...points.map((point) => point.y)));
  const right = Math.min(WIDTH, Math.max(...points.map((point) => point.x)));
  const bottom = Math.min(HEIGHT, Math.max(...points.map((point) => point.y)));
  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

function cross(a, b, point) {
  return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
}

function pointInConvexPolygon(point, polygon) {
  let sign = 0;
  for (let index = 0; index < polygon.length; index += 1) {
    const value = cross(
      polygon[index],
      polygon[(index + 1) % polygon.length],
      point,
    );
    if (Math.abs(value) < 0.001) continue;
    const nextSign = Math.sign(value);
    if (sign !== 0 && nextSign !== sign) return false;
    sign = nextSign;
  }
  return true;
}

function interpolateQuad(quad, horizontal, vertical) {
  const top = {
    x: quad[0].x + (quad[1].x - quad[0].x) * horizontal,
    y: quad[0].y + (quad[1].y - quad[0].y) * horizontal,
  };
  const bottom = {
    x: quad[3].x + (quad[2].x - quad[3].x) * horizontal,
    y: quad[3].y + (quad[2].y - quad[3].y) * horizontal,
  };
  return {
    x: top.x + (bottom.x - top.x) * vertical,
    y: top.y + (bottom.y - top.y) * vertical,
  };
}

function visibleFraction(indexQuad, laterCardQuads) {
  let visible = 0;
  let total = 0;
  for (let row = 0; row < 7; row += 1) {
    for (let column = 0; column < 7; column += 1) {
      const point = interpolateQuad(indexQuad, (column + 0.5) / 7, (row + 0.5) / 7);
      total += 1;
      const onCanvas =
        point.x >= 0 && point.x < WIDTH && point.y >= 0 && point.y < HEIGHT;
      const covered = laterCardQuads.some((quad) =>
        pointInConvexPolygon(point, quad),
      );
      if (onCanvas && !covered) visible += 1;
    }
  }
  return visible / total;
}

function backgroundSpec(random, frameIndex) {
  const palettes = [
    ["#173b2f", "#31594b"],
    ["#273849", "#526578"],
    ["#513a2d", "#775b48"],
    ["#43354c", "#6e5878"],
  ];
  const [dark, light] = random.pick(palettes);
  const stripe = random.between(220, 360);
  return {
    dark,
    light,
    stripe,
    stripeWidth: random.between(1, 4),
    paperX: random.between(30, WIDTH - 480),
    paperY: random.between(30, HEIGHT - 320),
    paperWidth: random.between(300, 470),
    paperHeight: random.between(150, 300),
    frameIndex,
  };
}

async function loadDeck(source) {
  const directory = join(root, source.path);
  const deck = [];
  for (const card of CARDS) {
    const path = join(directory, `${card.id}.png`);
    const data = await readFile(path);
    const dimensions = pngDimensions(data);
    deck.push({
      ...card,
      ...dimensions,
      absolutePath: path,
      path: relative(root, path).replaceAll("\\", "/"),
      sha256: sha256(data),
    });
  }
  return deck;
}

function renderJpeg(background, cards, output, quality, split) {
  const temporaryDirectory = mkdtempSync(join(tmpdir(), "belote-scene-"));
  try {
    const backgroundPath = join(temporaryDirectory, "background.png");
    const stripeCommands = [];
    for (
      let offset = -HEIGHT;
      offset < WIDTH + HEIGHT;
      offset += background.stripe
    ) {
      stripeCommands.push(`line ${offset},${HEIGHT} ${offset + HEIGHT},0`);
    }
    const paperRight = background.paperX + background.paperWidth;
    const paperBottom = background.paperY + background.paperHeight;
    const draw = [
      `stroke ${background.light}`,
      "stroke-opacity 0.22",
      `stroke-width ${background.stripeWidth}`,
      ...stripeCommands,
      "stroke none",
      "fill rgba(242,237,223,0.34)",
      `roundrectangle ${background.paperX},${background.paperY} ${paperRight},${paperBottom} 10,10`,
      "fill none",
      "stroke rgba(48,57,66,0.45)",
      "stroke-width 6",
      `line ${background.paperX + 22},${background.paperY + 48} ${Math.min(paperRight - 20, background.paperX + 320)},${background.paperY + 48}`,
      `line ${background.paperX + 22},${background.paperY + 82} ${Math.min(paperRight - 20, background.paperX + 255 + background.frameIndex * 4)},${background.paperY + 82}`,
    ].join(" ");
    execFileSync(
      "magick",
      [
        "-size",
        `${WIDTH}x${HEIGHT}`,
        `gradient:${background.dark}-${background.light}`,
        "-draw",
        draw,
        "-colorspace",
        "sRGB",
        backgroundPath,
      ],
      { maxBuffer: 20 * 1024 * 1024 },
    );
    const layerPaths = [];
    for (let index = 0; index < cards.length; index += 1) {
      const card = cards[index];
      let cardPath = card.sourceCard.absolutePath;
      if (card.alphabet === "french") {
        const overlayPath = join(temporaryDirectory, `overlay-${index}.png`);
        const patchedCardPath = join(temporaryDirectory, `card-${index}.png`);
        const overlay = `<svg xmlns="http://www.w3.org/2000/svg" width="${card.sourceCard.width}" height="${card.sourceCard.height}" viewBox="0 0 ${card.sourceCard.width} ${card.sourceCard.height}">${frenchCornerOverlay(
          card.sourceCard.width,
          card.sourceCard.height,
          card.glyph,
          card.sourceCard.suit,
          split === "holdout" ? "holdout" : "training",
        )}</svg>`;
        execFileSync("magick", ["-background", "none", "svg:-", overlayPath], {
          input: overlay,
          maxBuffer: 5 * 1024 * 1024,
        });
        execFileSync(
          "magick",
          [cardPath, overlayPath, "-compose", "over", "-composite", patchedCardPath],
          { maxBuffer: 5 * 1024 * 1024 },
        );
        cardPath = patchedCardPath;
      }
      const layerPath = join(temporaryDirectory, `layer-${index}.png`);
      const { a, b, c, d, e, f } = card.matrix;
      execFileSync(
        "magick",
        [
          cardPath,
          "-alpha",
          "set",
          "-virtual-pixel",
          "transparent",
          "+distort",
          "AffineProjection",
          `${a},${b},${c},${d},${e},${f}`,
          layerPath,
        ],
        { maxBuffer: 10 * 1024 * 1024 },
      );
      layerPaths.push(layerPath);
    }
    const compositeArguments = [
      backgroundPath,
      ...layerPaths,
      "-background",
      "none",
      "-layers",
      "merge",
    ];
    compositeArguments.push(
      "-strip",
      "-colorspace",
      "sRGB",
      "-sampling-factor",
      "4:2:0",
      "-quality",
      String(quality),
      output,
    );
    execFileSync("magick", compositeArguments, { maxBuffer: 20 * 1024 * 1024 });
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
}

function renderContactSheet(images, output) {
  const temporaryDirectory = mkdtempSync(join(tmpdir(), "belote-preview-"));
  try {
    const thumbnailPaths = [];
    for (const [index, image] of images.slice(0, 16).entries()) {
      const thumbnailPath = join(temporaryDirectory, `thumb-${index}.jpg`);
      execFileSync(
        "magick",
        [
          image,
          "-thumbnail",
          "240x320",
          "-gravity",
          "center",
          "-background",
          "#171b20",
          "-extent",
          "240x320",
          thumbnailPath,
        ],
        { maxBuffer: 10 * 1024 * 1024 },
      );
      thumbnailPaths.push(thumbnailPath);
    }
    const rowPaths = [];
    for (let index = 0; index < thumbnailPaths.length; index += 4) {
      const rowPath = join(temporaryDirectory, `row-${rowPaths.length}.jpg`);
      execFileSync(
        "magick",
        [...thumbnailPaths.slice(index, index + 4), "+append", rowPath],
        { maxBuffer: 20 * 1024 * 1024 },
      );
      rowPaths.push(rowPath);
    }
    execFileSync("magick", [...rowPaths, "-append", output], {
      maxBuffer: 20 * 1024 * 1024,
    });
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
}

const options = parseArguments(process.argv.slice(2));
const source = SOURCE_BY_SPLIT[options.split];
const random = createRandom(options.seed);
const deck = await loadDeck(source);
const imagesDirectory = join(options.output, "images");

await mkdir(options.output, { recursive: true });
await rm(imagesDirectory, { recursive: true, force: true });
await mkdir(imagesDirectory, { recursive: true });

const frames = [];
const cocoImages = [];
const cocoAnnotations = [];
const classifierLabels = [];
const imagePaths = [];
let annotationId = 1;
const faceVariantCount = { J: 0, Q: 0, K: 0 };

for (let frameIndex = 0; frameIndex < options.count; frameIndex += 1) {
  const cardCount = 1 + (frameIndex % 32);
  const mode =
    cardCount >= 17
      ? "dense"
      : cardCount >= 4 && random.float() < 0.38
        ? "fan"
        : "scatter";
  const cards = [];
  const sceneDeck = random.shuffle(deck);

  for (let cardIndex = 0; cardIndex < cardCount; cardIndex += 1) {
    const sourceCard = sceneDeck[cardIndex];
    let alphabet = "shared";
    let glyph = sourceCard.rank;
    if (sourceCard.rank in FRENCH_FACE_GLYPH) {
      const useFrench = faceVariantCount[sourceCard.rank] % 2 === 1;
      faceVariantCount[sourceCard.rank] += 1;
      alphabet = useFrench ? "french" : "english";
      glyph = useFrench ? FRENCH_FACE_GLYPH[sourceCard.rank] : sourceCard.rank;
    }
    const matrix = matrixForCard(sourceCard, random, mode, cardIndex, cardCount);
    const cardQuad = transformRectangle(
      matrix,
      0,
      0,
      sourceCard.width,
      sourceCard.height,
    );
    const indexWidth = sourceCard.width * 0.28;
    const indexHeight = sourceCard.height * 0.36;
    const indexQuads = [
      {
        corner: "top-left",
        quad: transformRectangle(matrix, 0, 0, indexWidth, indexHeight),
      },
      {
        corner: "bottom-right",
        quad: transformRectangle(
          matrix,
          sourceCard.width - indexWidth,
          sourceCard.height - indexHeight,
          indexWidth,
          indexHeight,
        ),
      },
    ];
    cards.push({
      sourceCard,
      matrix,
      cardQuad,
      indexQuads,
      alphabet,
      glyph,
      instanceId: `${frameIndex}-${cardIndex}-${sourceCard.id}`,
    });
  }

  const background = backgroundSpec(random, frameIndex);
  const imageName = `${String(frameIndex).padStart(5, "0")}.jpg`;
  const imagePath = join(imagesDirectory, imageName);
  renderJpeg(background, cards, imagePath, random.integer(70, 92), options.split);
  imagePaths.push(imagePath);

  const frameAnnotations = [];
  for (let cardIndex = 0; cardIndex < cards.length; cardIndex += 1) {
    const card = cards[cardIndex];
    const laterQuads = cards.slice(cardIndex + 1).map((later) => later.cardQuad);
    for (const index of card.indexQuads) {
      const visibility = visibleFraction(index.quad, laterQuads);
      const box = polygonBox(index.quad);
      if (visibility < 0.52 || box.width < 10 || box.height < 10) continue;

      const annotation = {
        id: annotationId,
        instanceId: card.instanceId,
        corner: index.corner,
        bbox: box,
        polygon: index.quad,
        visibleFraction: visibility,
        card: {
          rank: card.sourceCard.rank,
          suit: card.sourceCard.suit,
        },
        glyph: card.glyph,
        alphabet: card.alphabet,
        sourceAsset: card.sourceCard.path,
        sourceSha256: card.sourceCard.sha256,
      };
      frameAnnotations.push(annotation);
      classifierLabels.push({
        image: `images/${imageName}`,
        annotationId,
        bbox: box,
        rank: card.sourceCard.rank,
        suit: card.sourceCard.suit,
        glyph: card.glyph,
        alphabet: card.alphabet,
        split: options.split,
      });
      cocoAnnotations.push({
        id: annotationId,
        image_id: frameIndex + 1,
        category_id: 1,
        bbox: [box.x, box.y, box.width, box.height],
        area: box.width * box.height,
        segmentation: [index.quad.flatMap((point) => [point.x, point.y])],
        iscrowd: 0,
        attributes: {
          instanceId: card.instanceId,
          corner: index.corner,
          rank: card.sourceCard.rank,
          suit: card.sourceCard.suit,
          glyph: card.glyph,
          alphabet: card.alphabet,
          visibleFraction: visibility,
        },
      });
      annotationId += 1;
    }
  }

  frames.push({
    image: `images/${imageName}`,
    width: WIDTH,
    height: HEIGHT,
    cardCount,
    mode,
    annotations: frameAnnotations,
  });
  cocoImages.push({
    id: frameIndex + 1,
    file_name: `images/${imageName}`,
    width: WIDTH,
    height: HEIGHT,
  });
}

const coverage = {
  canonicalCards: [...new Set(classifierLabels.map((label) => `${label.rank}-${label.suit}`))].sort(),
  glyphs: [...new Set(classifierLabels.map((label) => label.glyph))].sort(),
  suits: [...new Set(classifierLabels.map((label) => label.suit))].sort(),
};
const dataset = {
  schemaVersion: 1,
  split: options.split,
  seed: options.seed,
  width: WIDTH,
  height: HEIGHT,
  source: {
    id: source.id,
    role: source.role,
    path: source.path,
  },
  generator: {
    version: 1,
    compositor: "affine-card-scenes",
    annotations: "two-visible-corner-indices",
    minimumVisibleFraction: 0.52,
  },
  coverage,
  frames,
};
const coco = {
  info: {
    description: `Belote corner-index ${options.split}`,
    version: "1",
  },
  images: cocoImages,
  annotations: cocoAnnotations,
  categories: [{ id: 1, name: "corner-index" }],
};
const datasetPath = join(options.output, "dataset.json");
const cocoPath = join(options.output, "annotations.coco.json");
const classifierPath = join(options.output, "classifier-labels.jsonl");
await writeFile(datasetPath, `${JSON.stringify(dataset, null, 2)}\n`);
await writeFile(cocoPath, `${JSON.stringify(coco, null, 2)}\n`);
await writeFile(
  classifierPath,
  `${classifierLabels.map((label) => JSON.stringify(label)).join("\n")}\n`,
);
renderContactSheet(imagePaths, join(options.output, "preview.jpg"));

const checksumTargets = [
  ...imagePaths,
  datasetPath,
  cocoPath,
  classifierPath,
  join(options.output, "preview.jpg"),
];
const checksums = [];
for (const path of checksumTargets) {
  const file = await readFile(path);
  checksums.push(
    `${sha256(file)}  ${relative(options.output, path).replaceAll("\\", "/")}`,
  );
}
await writeFile(
  join(options.output, "dataset.sha256"),
  `${checksums.sort().join("\n")}\n`,
);

const imageCount = (await readdir(imagesDirectory)).length;
const bytes = (await Promise.all(imagePaths.map((path) => stat(path)))).reduce(
  (sum, entry) => sum + entry.size,
  0,
);
console.log(
  `${options.split}: ${imageCount} images, ${classifierLabels.length} visible indices, ${(bytes / 1024 / 1024).toFixed(2)} MiB`,
);
