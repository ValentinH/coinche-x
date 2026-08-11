import type {
  EvaluationMetrics,
  SampleEvaluation,
} from "./contracts.ts";

export function computeEvaluationMetrics(
  samples: readonly SampleEvaluation[],
): EvaluationMetrics {
  const matchedCards = sum(samples, (sample) => sample.diff.matchedCards.length);
  const missingCards = sum(samples, (sample) => sample.diff.missingCards.length);
  const extraPredictions = sum(
    samples,
    (sample) =>
      sample.diff.extraCards.length +
      sample.diff.duplicateCards.length +
      sample.diff.invalidCardIds.length,
  );
  const cardDenominator = matchedCards + missingCards + extraPredictions;
  const latencies = samples
    .flatMap((sample) =>
      sample.latencyMs === undefined ? [] : [sample.latencyMs],
    )
    .sort((left, right) => left - right);

  return {
    totalSamples: samples.length,
    exactSets: samples.filter((sample) => sample.diff.exact).length,
    exactSetAccuracy: ratio(
      samples.filter((sample) => sample.diff.exact).length,
      samples.length,
    ),
    matchedCards,
    missingCards,
    extraPredictions,
    cardAccuracy: ratio(matchedCards, cardDenominator),
    integrityFailures: samples.filter(
      (sample) => sample.status === "integrity-failure",
    ).length,
    recognitionFailures: samples.filter(
      (sample) => sample.status === "recognition-failure",
    ).length,
    latency: {
      measuredSamples: latencies.length,
      meanMs: ratio(sum(latencies, (latency) => latency), latencies.length),
      p50Ms: percentile(latencies, 50),
      p95Ms: percentile(latencies, 95),
      maxMs: latencies.at(-1) ?? 0,
    },
  };
}

function percentile(sortedValues: readonly number[], value: number): number {
  if (sortedValues.length === 0) {
    return 0;
  }

  const index = Math.max(0, Math.ceil((value / 100) * sortedValues.length) - 1);
  return sortedValues[index] ?? 0;
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

function sum<T>(values: readonly T[], select: (value: T) => number): number {
  return values.reduce((total, value) => total + select(value), 0);
}
