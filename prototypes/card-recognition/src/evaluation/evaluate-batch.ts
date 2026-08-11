import {
  recognizeImage,
  type BrowserInferenceAdapter,
} from "../recognition/index.ts";
import { compareCardSets, emptyFailedDiff } from "./compare-card-sets.ts";
import type {
  EvaluationDataset,
  EvaluationProgress,
  EvaluationRun,
  SampleEvaluation,
} from "./contracts.ts";
import { sha256Hex } from "./hash.ts";
import { computeEvaluationMetrics } from "./metrics.ts";

interface EvaluateBatchOptions {
  hashBlob?: (blob: Blob) => Promise<string>;
  now?: () => Date;
  createRunId?: () => string;
  onProgress?: (progress: EvaluationProgress) => void;
  signal?: AbortSignal;
}

export async function evaluateBatch(
  dataset: EvaluationDataset,
  adapter: BrowserInferenceAdapter,
  options: EvaluateBatchOptions = {},
): Promise<EvaluationRun> {
  const now = options.now ?? (() => new Date());
  const hashBlob = options.hashBlob ?? sha256Hex;
  const startedAt = now();
  const samples: SampleEvaluation[] = [];

  for (const [index, sample] of dataset.samples.entries()) {
    options.signal?.throwIfAborted();
    let actualSha256: string;

    try {
      actualSha256 = (await hashBlob(sample.image)).toLowerCase();
    } catch (error) {
      samples.push({
        id: sample.id,
        filename: sample.filename,
        expectedCards: sample.expectedCards,
        predictedCards: [],
        expectedSha256: sample.expectedSha256,
        status: "integrity-failure",
        failure: `SHA-256 impossible : ${errorMessage(error)}`,
        diff: emptyFailedDiff(sample.expectedCards),
        diagnostics: [],
      });
      reportProgress(options, index, dataset, sample.filename);
      continue;
    }

    if (actualSha256 !== sample.expectedSha256) {
      samples.push({
        id: sample.id,
        filename: sample.filename,
        expectedCards: sample.expectedCards,
        predictedCards: [],
        expectedSha256: sample.expectedSha256,
        actualSha256,
        status: "integrity-failure",
        failure: "Empreinte SHA-256 différente : inférence bloquée.",
        diff: emptyFailedDiff(sample.expectedCards),
        diagnostics: [],
      });
      reportProgress(options, index, dataset, sample.filename);
      continue;
    }

    try {
      const prediction = await recognizeImage(sample.image, adapter, {
        signal: options.signal,
      });
      samples.push({
        id: sample.id,
        filename: sample.filename,
        expectedCards: sample.expectedCards,
        predictedCards: prediction.cards,
        expectedSha256: sample.expectedSha256,
        actualSha256,
        status: "evaluated",
        diff: compareCardSets(sample.expectedCards, prediction),
        latencyMs: prediction.diagnostics.latencyMs,
        diagnostics: prediction.diagnostics.messages,
      });
    } catch (error) {
      samples.push({
        id: sample.id,
        filename: sample.filename,
        expectedCards: sample.expectedCards,
        predictedCards: [],
        expectedSha256: sample.expectedSha256,
        actualSha256,
        status: "recognition-failure",
        failure: errorMessage(error),
        diff: emptyFailedDiff(sample.expectedCards),
        diagnostics: [],
      });
    }

    reportProgress(options, index, dataset, sample.filename);
  }

  const finishedAt = now();

  return {
    id: options.createRunId?.() ?? crypto.randomUUID(),
    corpusId: dataset.corpusId,
    adapterId: adapter.id,
    startedAt: startedAt.toISOString(),
    finishedAt: finishedAt.toISOString(),
    valid: samples.every((sample) => sample.status !== "integrity-failure"),
    metrics: computeEvaluationMetrics(samples),
    samples,
  };
}

function reportProgress(
  options: EvaluateBatchOptions,
  index: number,
  dataset: EvaluationDataset,
  filename: string,
): void {
  options.onProgress?.({
    completed: index + 1,
    total: dataset.samples.length,
    filename,
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
