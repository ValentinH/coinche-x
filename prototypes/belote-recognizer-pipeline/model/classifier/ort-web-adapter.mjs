const RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"];
const SUITS = ["clubs", "diamonds", "hearts", "spades"];
const CARDS = RANKS.flatMap((rank) =>
  SUITS.map((suit) => `${rank}-${suit}`),
);

function imageBatch(images, inputSize) {
  const plane = inputSize * inputSize;
  const values = new Float32Array(images.length * 3 * plane);
  for (const [batchIndex, image] of images.entries()) {
    if (image.width !== inputSize || image.height !== inputSize) {
      throw new Error(`Expected ${inputSize}x${inputSize} classifier crop`);
    }
    for (let pixelIndex = 0; pixelIndex < plane; pixelIndex += 1) {
      const rgbaOffset = pixelIndex * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const tensorOffset =
          batchIndex * 3 * plane + channel * plane + pixelIndex;
        values[tensorOffset] = image.data[rgbaOffset + channel] / 127.5 - 1;
      }
    }
  }
  return values;
}

function winningClass(logits, row, classes) {
  const offset = row * classes.length;
  let winner = 0;
  let maximum = -Infinity;
  let normalizer = 0;
  for (let index = 0; index < classes.length; index += 1) {
    maximum = Math.max(maximum, logits[offset + index]);
  }
  for (let index = 0; index < classes.length; index += 1) {
    normalizer += Math.exp(logits[offset + index] - maximum);
    if (logits[offset + index] > logits[offset + winner]) winner = index;
  }
  return {
    label: classes[winner],
    confidence: Math.exp(logits[offset + winner] - maximum) / normalizer,
  };
}

export class OrtWebCardClassifier {
  constructor(ort, session, inputSize = 96, indexThreshold = 0.5) {
    this.ort = ort;
    this.session = session;
    this.inputSize = inputSize;
    this.indexThreshold = indexThreshold;
  }

  async classify(images) {
    if (images.length === 0) return [];
    const tensor = new this.ort.Tensor(
      "float32",
      imageBatch(images, this.inputSize),
      [images.length, 3, this.inputSize, this.inputSize],
    );
    const outputs = await this.session.run({ input: tensor });
    const rankLogits = outputs.rank_logits?.data;
    const suitLogits = outputs.suit_logits?.data;
    const indexLogits = outputs.index_logit?.data;
    const cardLogits = outputs.card_logits?.data;
    if (!rankLogits || !suitLogits || !indexLogits || !cardLogits) {
      throw new Error("Classifier ONNX outputs are missing");
    }
    return images.map((_, index) => {
      const rank = winningClass(rankLogits, index, RANKS);
      const suit = winningClass(suitLogits, index, SUITS);
      const card = winningClass(cardLogits, index, CARDS);
      const [selectedRank, selectedSuit] = card.label.split("-");
      const indexConfidence = 1 / (1 + Math.exp(-indexLogits[index]));
      return {
        rank: selectedRank,
        suit: selectedSuit,
        rankConfidence: rank.confidence,
        suitConfidence: suit.confidence,
        cardConfidence: card.confidence,
        isIndex: indexConfidence >= this.indexThreshold,
        indexConfidence,
      };
    });
  }
}

export async function createOrtWebCardClassifier({
  ort,
  modelUrl,
  inputSize = 96,
  indexThreshold = 0.5,
  sessionOptions = {},
}) {
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    ...sessionOptions,
  });
  return new OrtWebCardClassifier(
    ort,
    session,
    inputSize,
    indexThreshold,
  );
}
