import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

import {
  placeholderAdapter,
  recognizeImage,
  type BrowserInferenceAdapter,
} from "../src/recognition/index.ts";
import type {
  EvaluationDataset,
  EvaluationRun,
} from "../src/evaluation/contracts.ts";
import { evaluateBatch } from "../src/evaluation/evaluate-batch.ts";
import { renderEvaluationDashboard } from "../src/evaluation/mount-evaluation-page.ts";
import { parseSmokeManifestV2 } from "../src/evaluation/smoke-manifest-v2.ts";

test("image seam returns a unique canonical set and explicit diagnostics", async () => {
  const adapter: BrowserInferenceAdapter = {
    id: "synthetic-test",
    label: "Synthetic test",
    async infer() {
      return { cardIds: ["AS", "7C", "AS", "XX", "10D"] };
    },
  };

  const result = await recognizeImage(new Blob(["synthetic"]), adapter, {
    now: sequenceClock(10, 23),
  });

  assert.deepEqual(result.cards, ["7C", "10D", "AS"]);
  assert.deepEqual(result.diagnostics.duplicateCards, ["AS"]);
  assert.deepEqual(result.diagnostics.invalidCardIds, ["XX"]);
  assert.equal(result.diagnostics.latencyMs, 13);

  const placeholder = await recognizeImage(
    new Blob(["synthetic"]),
    placeholderAdapter,
  );
  assert.deepEqual(placeholder.cards, []);
  assert.equal(
    placeholder.diagnostics.messages[0]?.code,
    "PLACEHOLDER_NO_INFERENCE",
  );
});

test("batch seam scores exact sets, duplicate extras and integrity failures", async () => {
  const adapter: BrowserInferenceAdapter = {
    id: "synthetic-test",
    label: "Synthetic test",
    async infer(image) {
      return {
        cardIds:
          (await image.text()) === "first"
            ? ["7C", "7C", "AS"]
            : ["AS"],
      };
    },
  };
  const dataset: EvaluationDataset = {
    corpusId: "synthetic-corpus",
    samples: [
      sample("first.jpg", "first", ["7C", "8D"], "verified"),
      sample("second.jpg", "second", ["AS"], "verified"),
    ],
  };

  const run = await evaluateBatch(dataset, adapter, {
    hashBlob: async () => "verified",
    createRunId: () => "run-1",
    now: sequenceDates(),
  });

  assert.equal(run.metrics.exactSetAccuracy, 0.5);
  assert.equal(run.metrics.cardAccuracy, 0.4);
  assert.equal(run.metrics.extraPredictions, 2);
  assert.equal(run.metrics.latency.measuredSamples, 2);

  let inferenceCalls = 0;
  const blocked = await evaluateBatch(
    {
      corpusId: "synthetic-integrity",
      samples: [sample("changed.jpg", "changed", ["AS"], "expected")],
    },
    {
      id: "must-not-run",
      label: "Must not run",
      async infer() {
        inferenceCalls += 1;
        return { cardIds: ["AS"] };
      },
    },
    {
      hashBlob: async () => "different",
      now: sequenceDates(),
    },
  );

  assert.equal(inferenceCalls, 0);
  assert.equal(blocked.valid, false);
  assert.equal(blocked.metrics.integrityFailures, 1);
});

test("progress view exposes predictions, failures, metrics, latency and history", () => {
  const run = syntheticRun();
  const root = { innerHTML: "" } as HTMLElement;

  renderEvaluationDashboard(root, [run, { ...run, id: "run-2" }]);

  assert.match(root.innerHTML, /Photos exactes/);
  assert.match(root.innerHTML, /Précision carte/);
  assert.match(root.innerHTML, /Latence P50/);
  assert.match(root.innerHTML, /Échecs \(1\)/);
  assert.match(root.innerHTML, /Toutes les prédictions/);
  assert.match(root.innerHTML, /Évolution des runs/);
  assert.match(root.innerHTML, /2 run\(s\)/);
});

test("truth loader accepts v2 direct cards, rejects v1 and stays out of recognition", async () => {
  const hash = "a".repeat(64);
  const parsed = parseSmokeManifestV2({
    schemaVersion: 2,
    corpusId: "synthetic-corpus",
    images: [{ file: "synthetic.jpg", cards: ["7C"], sha256: hash }],
  });
  assert.deepEqual(parsed.images[0]?.cards, ["7C"]);
  assert.throws(
    () =>
      parseSmokeManifestV2({
        schemaVersion: 1,
        corpusId: "old",
        cardSets: { hand: ["7C"] },
        images: [],
      }),
    /Schéma v1 refusé/,
  );

  const recognitionDirectory = new URL("../src/recognition/", import.meta.url);
  for (const filename of await readdir(recognitionDirectory)) {
    if (!filename.endsWith(".ts")) continue;
    const source = await readFile(new URL(filename, recognitionDirectory), "utf8");
    assert.doesNotMatch(source, /from\s+["'][^"']*evaluation|import\(["'][^"']*evaluation/);
  }
});

function sample(
  filename: string,
  content: string,
  expectedCards: EvaluationDataset["samples"][number]["expectedCards"],
  expectedSha256: string,
): EvaluationDataset["samples"][number] {
  return {
    id: filename,
    filename,
    image: new Blob([content]),
    expectedSha256,
    expectedCards,
  };
}

function sequenceClock(...values: number[]): () => number {
  let index = 0;
  return () => values[index++] ?? values.at(-1) ?? 0;
}

function sequenceDates(): () => Date {
  let index = 0;
  return () => new Date(`2026-07-30T08:00:0${index++}.000Z`);
}

function syntheticRun(): EvaluationRun {
  return {
    id: "run-1",
    corpusId: "synthetic-corpus",
    adapterId: "synthetic-test",
    startedAt: "2026-07-30T08:00:00.000Z",
    finishedAt: "2026-07-30T08:00:01.000Z",
    valid: true,
    metrics: {
      totalSamples: 1,
      exactSets: 0,
      exactSetAccuracy: 0,
      matchedCards: 0,
      missingCards: 1,
      extraPredictions: 0,
      cardAccuracy: 0,
      integrityFailures: 0,
      recognitionFailures: 0,
      latency: {
        measuredSamples: 1,
        meanMs: 12,
        p50Ms: 12,
        p95Ms: 12,
        maxMs: 12,
      },
    },
    samples: [
      {
        id: "synthetic.jpg",
        filename: "synthetic.jpg",
        expectedCards: ["AS"],
        predictedCards: [],
        expectedSha256: "synthetic",
        actualSha256: "synthetic",
        status: "evaluated",
        diff: {
          matchedCards: [],
          missingCards: ["AS"],
          extraCards: [],
          duplicateCards: [],
          invalidCardIds: [],
          exact: false,
        },
        latencyMs: 12,
        diagnostics: [],
      },
    ],
  };
}
