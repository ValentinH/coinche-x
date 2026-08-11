import { canonicalCardSet } from "./cards.ts";
import type {
  BrowserInferenceAdapter,
  RecognitionDiagnostic,
  RecognitionResult,
} from "./contracts.ts";

interface RecognitionOptions {
  signal?: AbortSignal;
  now?: () => number;
}

export async function recognizeImage(
  image: Blob,
  adapter: BrowserInferenceAdapter,
  options: RecognitionOptions = {},
): Promise<RecognitionResult> {
  const now = options.now ?? (() => performance.now());
  const startedAt = now();
  const output = await adapter.infer(image, options.signal);
  const latencyMs = Math.max(0, now() - startedAt);

  if (!Array.isArray(output.cardIds)) {
    throw new TypeError("Inference output cardIds must be an array");
  }

  const canonical = canonicalCardSet(output.cardIds);
  const messages: RecognitionDiagnostic[] = [...(output.diagnostics ?? [])];

  if (canonical.duplicateCards.length > 0) {
    messages.push({
      code: "DUPLICATE_CARD_IDS",
      level: "error",
      message: `${canonical.duplicateCards.length} carte(s) dupliquée(s) ignorée(s).`,
    });
  }

  if (canonical.invalidCardIds.length > 0) {
    messages.push({
      code: "INVALID_CARD_IDS",
      level: "error",
      message: `${canonical.invalidCardIds.length} identifiant(s) non canonique(s) rejeté(s).`,
    });
  }

  return {
    cards: canonical.cards,
    diagnostics: {
      adapterId: adapter.id,
      latencyMs,
      duplicateCards: canonical.duplicateCards,
      invalidCardIds: canonical.invalidCardIds,
      messages,
    },
  };
}
