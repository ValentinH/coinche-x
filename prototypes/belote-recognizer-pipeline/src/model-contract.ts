import type { Rank, Suit } from "./domain.ts";

export interface PixelBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CornerProposal {
  box: PixelBox;
  confidence: number;
}

export interface CornerClassification {
  rank: Rank;
  suit: Suit;
  rankConfidence: number;
  suitConfidence: number;
  alphabet?: "english" | "french";
}

export interface CornerDetector {
  readonly inputSize: number;
  detect(images: readonly ImageData[]): Promise<readonly (readonly CornerProposal[])[]>;
}

export interface CardClassifier {
  readonly inputSize: number;
  classify(images: readonly ImageData[]): Promise<readonly CornerClassification[]>;
}

export interface ModelArtifact {
  url: string;
  sha256: string;
  bytes: number;
}

export interface RecognizerModelManifest {
  schemaVersion: 1;
  detector: ModelArtifact & {
    inputSize: number;
    outputBoxes: string;
    outputScores: string;
  };
  classifier: ModelArtifact & {
    inputSize: number;
    outputRankLogits: string;
    outputSuitLogits: string;
    outputAlphabetLogits?: string;
  };
  rankClasses: readonly Rank[];
  suitClasses: readonly Suit[];
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
export const MAX_COMBINED_MODEL_BYTES = 8 * 1024 * 1024;

export function validateModelManifest(manifest: RecognizerModelManifest): void {
  if (manifest.schemaVersion !== 1) throw new Error("Unsupported model manifest");
  if (manifest.detector.inputSize < 128 || manifest.classifier.inputSize < 32) {
    throw new Error("Implausible model input size");
  }
  if (
    !SHA256_PATTERN.test(manifest.detector.sha256) ||
    !SHA256_PATTERN.test(manifest.classifier.sha256)
  ) {
    throw new Error("Model SHA-256 must be lowercase hex");
  }
  if (manifest.detector.bytes + manifest.classifier.bytes > MAX_COMBINED_MODEL_BYTES) {
    throw new Error("Combined models exceed the 8 MiB mobile budget");
  }
  if (new Set(manifest.rankClasses).size !== 8 || new Set(manifest.suitClasses).size !== 4) {
    throw new Error("Expected factorized 8-rank/4-suit classes");
  }
}
