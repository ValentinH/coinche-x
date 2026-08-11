import type {
  CanonicalCard,
  RecognitionDiagnostic,
} from "../recognition/index.ts";

export interface EvaluationSample {
  id: string;
  filename: string;
  image: Blob;
  expectedSha256: string;
  expectedCards: readonly CanonicalCard[];
}

export interface EvaluationDataset {
  corpusId: string;
  samples: readonly EvaluationSample[];
}

export type SampleStatus =
  | "evaluated"
  | "integrity-failure"
  | "recognition-failure";

export interface CardSetDiff {
  matchedCards: readonly CanonicalCard[];
  missingCards: readonly CanonicalCard[];
  extraCards: readonly CanonicalCard[];
  duplicateCards: readonly CanonicalCard[];
  invalidCardIds: readonly string[];
  exact: boolean;
}

export interface SampleEvaluation {
  id: string;
  filename: string;
  expectedCards: readonly CanonicalCard[];
  predictedCards: readonly CanonicalCard[];
  expectedSha256: string;
  actualSha256?: string;
  status: SampleStatus;
  failure?: string;
  diff: CardSetDiff;
  latencyMs?: number;
  diagnostics: readonly RecognitionDiagnostic[];
}

export interface EvaluationMetrics {
  totalSamples: number;
  exactSets: number;
  exactSetAccuracy: number;
  matchedCards: number;
  missingCards: number;
  extraPredictions: number;
  cardAccuracy: number;
  integrityFailures: number;
  recognitionFailures: number;
  latency: {
    measuredSamples: number;
    meanMs: number;
    p50Ms: number;
    p95Ms: number;
    maxMs: number;
  };
}

export interface EvaluationRun {
  id: string;
  corpusId: string;
  adapterId: string;
  startedAt: string;
  finishedAt: string;
  valid: boolean;
  metrics: EvaluationMetrics;
  samples: readonly SampleEvaluation[];
}

export interface EvaluationProgress {
  completed: number;
  total: number;
  filename: string;
}
