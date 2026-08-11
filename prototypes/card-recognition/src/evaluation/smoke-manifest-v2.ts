import {
  canonicalCardSet,
  type CanonicalCard,
} from "../recognition/index.ts";
import type { EvaluationDataset, EvaluationSample } from "./contracts.ts";
import { normalizeSha256 } from "./hash.ts";

interface SmokeImageV2 {
  file: string;
  cards: readonly unknown[];
  cardCount?: number;
  sha256: string;
}

interface SmokeManifestV2 {
  schemaVersion: 2;
  corpusId: string;
  images: readonly SmokeImageV2[];
}

export function parseSmokeManifestV2(value: unknown): SmokeManifestV2 {
  if (!isRecord(value)) {
    throw new TypeError("Manifeste invalide");
  }

  if ("cardSets" in value) {
    throw new TypeError(
      "Schéma v1 refusé : utiliser le schéma v2 avec images[].cards.",
    );
  }

  if (value.schemaVersion !== 2) {
    throw new TypeError("schemaVersion doit valoir 2");
  }

  if (typeof value.corpusId !== "string" || value.corpusId.length === 0) {
    throw new TypeError("corpusId manquant");
  }

  if (!Array.isArray(value.images) || value.images.length === 0) {
    throw new TypeError("images doit être une liste non vide");
  }

  const images = value.images.map((image, index) =>
    parseImage(image, index),
  );
  const filenames = new Set<string>();

  for (const image of images) {
    const filename = basename(image.file);
    if (filenames.has(filename)) {
      throw new TypeError(`Nom d’image ambigu : ${filename}`);
    }
    filenames.add(filename);
  }

  return {
    schemaVersion: 2,
    corpusId: value.corpusId,
    images,
  };
}

export async function loadSmokeDatasetV2(
  manifestFile: File,
  selectedImages: readonly File[],
): Promise<EvaluationDataset> {
  const raw = JSON.parse(await manifestFile.text()) as unknown;
  const manifest = parseSmokeManifestV2(raw);
  const filesByBasename = indexFilesByBasename(selectedImages);
  const samples: EvaluationSample[] = manifest.images.map((image) => {
    const filename = basename(image.file);
    const file = filesByBasename.get(filename);

    if (!file) {
      throw new TypeError(`Image manquante : ${filename}`);
    }

    return {
      id: image.file,
      filename,
      image: file,
      expectedSha256: normalizeSha256(image.sha256),
      expectedCards: image.cards as readonly CanonicalCard[],
    };
  });

  return {
    corpusId: manifest.corpusId,
    samples,
  };
}

function parseImage(value: unknown, index: number): SmokeImageV2 {
  if (!isRecord(value)) {
    throw new TypeError(`images[${index}] invalide`);
  }

  if (typeof value.file !== "string" || value.file.length === 0) {
    throw new TypeError(`images[${index}].file manquant`);
  }

  if (!Array.isArray(value.cards)) {
    throw new TypeError(`images[${index}].cards invalide`);
  }

  const canonical = canonicalCardSet(value.cards);
  if (
    canonical.invalidCardIds.length > 0 ||
    canonical.duplicateCards.length > 0
  ) {
    throw new TypeError(
      `images[${index}].cards doit être un ensemble canonique exact`,
    );
  }

  if (
    value.cardCount !== undefined &&
    value.cardCount !== canonical.cards.length
  ) {
    throw new TypeError(`images[${index}].cardCount incohérent`);
  }

  if (typeof value.sha256 !== "string") {
    throw new TypeError(`images[${index}].sha256 manquant`);
  }

  return {
    file: value.file,
    cards: canonical.cards,
    cardCount:
      typeof value.cardCount === "number" ? value.cardCount : undefined,
    sha256: normalizeSha256(value.sha256),
  };
}

function indexFilesByBasename(files: readonly File[]): Map<string, File> {
  const indexed = new Map<string, File>();

  for (const file of files) {
    const filename = basename(file.webkitRelativePath || file.name);
    if (indexed.has(filename)) {
      throw new TypeError(`Fichier sélectionné ambigu : ${filename}`);
    }
    indexed.set(filename, file);
  }

  return indexed;
}

function basename(path: string): string {
  return path.replaceAll("\\", "/").split("/").at(-1) ?? path;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
