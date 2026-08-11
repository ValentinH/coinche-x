import type { CanonicalCard } from "./cards.ts";

export type DiagnosticLevel = "info" | "warning" | "error";

export interface RecognitionDiagnostic {
  code: string;
  level: DiagnosticLevel;
  message: string;
}

export interface BrowserInferenceOutput {
  cardIds: readonly unknown[];
  diagnostics?: readonly RecognitionDiagnostic[];
}

export interface BrowserInferenceAdapter {
  readonly id: string;
  readonly label: string;
  infer(image: Blob, signal?: AbortSignal): Promise<BrowserInferenceOutput>;
}

export interface RecognitionResult {
  cards: readonly CanonicalCard[];
  diagnostics: {
    adapterId: string;
    latencyMs: number;
    duplicateCards: readonly CanonicalCard[];
    invalidCardIds: readonly string[];
    messages: readonly RecognitionDiagnostic[];
  };
}
