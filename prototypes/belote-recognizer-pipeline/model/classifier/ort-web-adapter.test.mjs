import assert from "node:assert/strict";
import test from "node:test";
import { OrtWebCardClassifier } from "./ort-web-adapter.mjs";

test("batches normalized NCHW crops and maps factorized logits", async () => {
  let receivedTensor;
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  }
  const session = {
    async run({ input }) {
      receivedTensor = input;
      const cardLogits = new Float32Array(32);
      cardLogits[18] = 5;
      return {
        rank_logits: {
          data: new Float32Array([0, 0, 0, 0, 5, 0, 0, 0]),
        },
        suit_logits: { data: new Float32Array([0, 0, 5, 0]) },
        index_logit: { data: new Float32Array([2]) },
        card_logits: { data: cardLogits },
      };
    },
  };
  const classifier = new OrtWebCardClassifier({ Tensor }, session, 1);
  const result = await classifier.classify([
    { width: 1, height: 1, data: new Uint8ClampedArray([255, 0, 128, 255]) },
  ]);

  assert.deepEqual(receivedTensor.dims, [1, 3, 1, 1]);
  assert.ok(Math.abs(receivedTensor.data[0] - 1) < 1e-6);
  assert.ok(Math.abs(receivedTensor.data[1] + 1) < 1e-6);
  assert.equal(result[0].rank, "J");
  assert.equal(result[0].suit, "hearts");
  assert.equal(result[0].isIndex, true);
  assert.ok(result[0].rankConfidence > 0.9);
  assert.ok(result[0].suitConfidence > 0.9);
  assert.ok(result[0].indexConfidence > 0.8);
});
