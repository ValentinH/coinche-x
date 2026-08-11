import { cardId, type Card } from "./domain.ts";
import type {
  CardClassifier,
  CornerDetector,
  CornerProposal,
  PixelBox,
} from "./model-contract.ts";

type BrowserCanvas = HTMLCanvasElement | OffscreenCanvas;
type BrowserContext =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D;

export interface RecognizedCard {
  card: Card;
  box: PixelBox;
  confidence: number;
  detectorConfidence: number;
  rankConfidence: number;
  suitConfidence: number;
}

export interface RecognitionDiagnostics {
  sourceWidth: number;
  sourceHeight: number;
  workingWidth: number;
  workingHeight: number;
  tileCount: number;
  proposalCount: number;
  detectionMs: number;
  classificationMs: number;
  totalMs: number;
  withinWarmBudget: boolean;
}

export interface RecognitionResult {
  cards: readonly RecognizedCard[];
  diagnostics: RecognitionDiagnostics;
}

export interface RecognizerOptions {
  maxLongEdge?: number;
  tileSize?: number;
  tileOverlap?: number;
  detectorThreshold?: number;
  classifierThreshold?: number;
  nmsThreshold?: number;
  cropPadding?: number;
}

interface LocatedProposal extends CornerProposal {
  tileX: number;
  tileY: number;
}

const DEFAULTS: Required<RecognizerOptions> = {
  maxLongEdge: 1920,
  tileSize: 960,
  tileOverlap: 192,
  detectorThreshold: 0.35,
  classifierThreshold: 0.45,
  nmsThreshold: 0.35,
  cropPadding: 0.22,
};

function createCanvas(width: number, height: number): BrowserCanvas {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function context(canvas: BrowserCanvas): BrowserContext {
  const value = canvas.getContext("2d", {
    alpha: false,
    willReadFrequently: true,
  });
  if (!value) throw new Error("2D canvas unavailable");
  return value;
}

function sourceDimensions(source: CanvasImageSource): {
  width: number;
  height: number;
} {
  const candidate = source as CanvasImageSource & {
    naturalWidth?: number;
    naturalHeight?: number;
    videoWidth?: number;
    videoHeight?: number;
    displayWidth?: number;
    displayHeight?: number;
    width?: number;
    height?: number;
  };
  const width =
    candidate.naturalWidth ??
    candidate.videoWidth ??
    candidate.displayWidth ??
    candidate.width;
  const height =
    candidate.naturalHeight ??
    candidate.videoHeight ??
    candidate.displayHeight ??
    candidate.height;
  if (!width || !height) throw new Error("Image source has no dimensions");
  return { width, height };
}

export function tileStarts(
  length: number,
  tileSize: number,
  overlap: number,
): readonly number[] {
  if (length <= tileSize) return [0];
  if (overlap < 0 || overlap >= tileSize) throw new Error("Invalid tile overlap");
  const last = length - tileSize;
  const starts = [];
  for (let value = 0; value < last; value += tileSize - overlap) {
    starts.push(value);
  }
  if (starts.at(-1) !== last) starts.push(last);
  return starts;
}

function intersectionOverUnion(left: PixelBox, right: PixelBox): number {
  const intersectionWidth = Math.max(
    0,
    Math.min(left.x + left.width, right.x + right.width) -
      Math.max(left.x, right.x),
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(left.y + left.height, right.y + right.height) -
      Math.max(left.y, right.y),
  );
  const intersection = intersectionWidth * intersectionHeight;
  const union =
    left.width * left.height + right.width * right.height - intersection;
  return union > 0 ? intersection / union : 0;
}

export function nonMaximumSuppression(
  proposals: readonly LocatedProposal[],
  threshold: number,
): readonly LocatedProposal[] {
  const sorted = [...proposals].sort((left, right) => right.confidence - left.confidence);
  const kept: LocatedProposal[] = [];
  for (const proposal of sorted) {
    const globalBox = {
      ...proposal.box,
      x: proposal.box.x + proposal.tileX,
      y: proposal.box.y + proposal.tileY,
    };
    const overlaps = kept.some((other) =>
      intersectionOverUnion(globalBox, {
        ...other.box,
        x: other.box.x + other.tileX,
        y: other.box.y + other.tileY,
      }) > threshold,
    );
    if (!overlaps) kept.push(proposal);
  }
  return kept;
}

export function deduplicateCards(
  recognitions: readonly RecognizedCard[],
): readonly RecognizedCard[] {
  const best = new Map<string, RecognizedCard>();
  for (const recognition of recognitions) {
    const id = cardId(recognition.card);
    if (!best.has(id) || best.get(id)!.confidence < recognition.confidence) {
      best.set(id, recognition);
    }
  }
  return [...best.values()].sort(
    (left, right) => left.box.y - right.box.y || left.box.x - right.box.x,
  );
}

export class BeloteRecognizer {
  readonly #detector: CornerDetector;
  readonly #classifier: CardClassifier;
  readonly #options: Required<RecognizerOptions>;

  constructor(
    detector: CornerDetector,
    classifier: CardClassifier,
    options: RecognizerOptions = {},
  ) {
    this.#detector = detector;
    this.#classifier = classifier;
    this.#options = { ...DEFAULTS, ...options };
  }

  async recognize(source: CanvasImageSource): Promise<RecognitionResult> {
    const started = performance.now();
    const sourceSize = sourceDimensions(source);
    const scale = Math.min(
      1,
      this.#options.maxLongEdge / Math.max(sourceSize.width, sourceSize.height),
    );
    const workingWidth = Math.round(sourceSize.width * scale);
    const workingHeight = Math.round(sourceSize.height * scale);
    const workingCanvas = createCanvas(workingWidth, workingHeight);
    const workingContext = context(workingCanvas);
    workingContext.drawImage(source, 0, 0, workingWidth, workingHeight);

    const xStarts = tileStarts(
      workingWidth,
      this.#options.tileSize,
      this.#options.tileOverlap,
    );
    const yStarts = tileStarts(
      workingHeight,
      this.#options.tileSize,
      this.#options.tileOverlap,
    );
    const tiles: ImageData[] = [];
    const tileLocations: { x: number; y: number; width: number; height: number }[] =
      [];
    for (const y of yStarts) {
      for (const x of xStarts) {
        const width = Math.min(this.#options.tileSize, workingWidth - x);
        const height = Math.min(this.#options.tileSize, workingHeight - y);
        const tileCanvas = createCanvas(
          this.#options.tileSize,
          this.#options.tileSize,
        );
        const tileContext = context(tileCanvas);
        tileContext.fillStyle = "#202020";
        tileContext.fillRect(
          0,
          0,
          this.#options.tileSize,
          this.#options.tileSize,
        );
        tileContext.drawImage(
          workingCanvas,
          x,
          y,
          width,
          height,
          0,
          0,
          width,
          height,
        );
        tiles.push(
          tileContext.getImageData(
            0,
            0,
            this.#options.tileSize,
            this.#options.tileSize,
          ),
        );
        tileLocations.push({ x, y, width, height });
      }
    }

    const detectionStarted = performance.now();
    const detections = await this.#detector.detect(tiles);
    if (detections.length !== tiles.length) {
      throw new Error("Detector returned the wrong batch size");
    }
    const located = detections.flatMap((proposals, tileIndex) => {
      const tile = tileLocations[tileIndex];
      return proposals
        .filter(
          (proposal) =>
            proposal.confidence >= this.#options.detectorThreshold &&
            proposal.box.width > 0 &&
            proposal.box.height > 0 &&
            proposal.box.x < tile.width &&
            proposal.box.y < tile.height,
        )
        .map((proposal) => ({
          ...proposal,
          tileX: tile.x,
          tileY: tile.y,
        }));
    });
    const proposals = nonMaximumSuppression(located, this.#options.nmsThreshold);
    const detectionMs = performance.now() - detectionStarted;

    const crops = proposals.map((proposal) => {
      const box = {
        ...proposal.box,
        x: proposal.box.x + proposal.tileX,
        y: proposal.box.y + proposal.tileY,
      };
      const padding = Math.max(box.width, box.height) * this.#options.cropPadding;
      const left = Math.max(0, box.x - padding);
      const top = Math.max(0, box.y - padding);
      const right = Math.min(workingWidth, box.x + box.width + padding);
      const bottom = Math.min(workingHeight, box.y + box.height + padding);
      const cropCanvas = createCanvas(
        this.#classifier.inputSize,
        this.#classifier.inputSize,
      );
      const cropContext = context(cropCanvas);
      cropContext.fillStyle = "#ffffff";
      cropContext.fillRect(
        0,
        0,
        this.#classifier.inputSize,
        this.#classifier.inputSize,
      );
      cropContext.drawImage(
        workingCanvas,
        left,
        top,
        right - left,
        bottom - top,
        0,
        0,
        this.#classifier.inputSize,
        this.#classifier.inputSize,
      );
      return cropContext.getImageData(
        0,
        0,
        this.#classifier.inputSize,
        this.#classifier.inputSize,
      );
    });

    const classificationStarted = performance.now();
    const classifications =
      crops.length === 0 ? [] : await this.#classifier.classify(crops);
    if (classifications.length !== crops.length) {
      throw new Error("Classifier returned the wrong batch size");
    }
    const classificationMs = performance.now() - classificationStarted;
    const recognitions = classifications.flatMap((classification, index) => {
      const proposal = proposals[index];
      if (
        classification.rankConfidence < this.#options.classifierThreshold ||
        classification.suitConfidence < this.#options.classifierThreshold
      ) {
        return [];
      }
      const confidence = Math.cbrt(
        proposal.confidence *
          classification.rankConfidence *
          classification.suitConfidence,
      );
      return [
        {
          card: { rank: classification.rank, suit: classification.suit },
          box: {
            x: (proposal.box.x + proposal.tileX) / scale,
            y: (proposal.box.y + proposal.tileY) / scale,
            width: proposal.box.width / scale,
            height: proposal.box.height / scale,
          },
          confidence,
          detectorConfidence: proposal.confidence,
          rankConfidence: classification.rankConfidence,
          suitConfidence: classification.suitConfidence,
        },
      ];
    });
    const totalMs = performance.now() - started;
    return {
      cards: deduplicateCards(recognitions),
      diagnostics: {
        sourceWidth: sourceSize.width,
        sourceHeight: sourceSize.height,
        workingWidth,
        workingHeight,
        tileCount: tiles.length,
        proposalCount: located.length,
        detectionMs,
        classificationMs,
        totalMs,
        withinWarmBudget: totalMs <= 5_000,
      },
    };
  }
}
