import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const corpusDir = resolve(
  import.meta.dirname,
  "../../../test-data/card-recognition/smoke",
);
const manifest = JSON.parse(
  await readFile(resolve(corpusDir, "manifest.json"), "utf8"),
);
const validCards = new Set(
  manifest.cardNotation.suits
    ? Object.keys(manifest.cardNotation.suits).flatMap((suit) =>
        manifest.cardNotation.ranks.map((rank) => `${rank}${suit}`),
      )
    : [],
);

if (manifest.schemaVersion !== 2 || "cardSets" in manifest) {
  throw new Error("Le manifest doit porter les cartes directement par image.");
}

for (const image of manifest.images) {
  if (!Array.isArray(image.cards) || image.cards.length !== image.cardCount) {
    throw new Error(`${image.file}: cardCount ne correspond pas à cards.`);
  }

  if (new Set(image.cards).size !== image.cards.length) {
    throw new Error(`${image.file}: carte dupliquée.`);
  }

  for (const card of image.cards) {
    if (!validCards.has(card)) {
      throw new Error(`${image.file}: carte invalide ${card}.`);
    }
  }

  if (image.cardCount === 32 && image.cards.length !== validCards.size) {
    throw new Error(`${image.file}: le jeu complet est incomplet.`);
  }
}

console.log(`${manifest.images.length} images valides, labels propres à chaque image.`);
