import type {
  BrowserInferenceAdapter,
  BrowserInferenceOutput,
} from "./contracts.ts";

export const placeholderAdapter: BrowserInferenceAdapter = {
  id: "placeholder-empty-v1",
  label: "Placeholder déterministe — aucune inférence",
  async infer(_image: Blob): Promise<BrowserInferenceOutput> {
    return {
      cardIds: [],
      diagnostics: [
        {
          code: "PLACEHOLDER_NO_INFERENCE",
          level: "warning",
          message:
            "Moteur factice explicite : aucune carte prédite, aucune précision simulée.",
        },
      ],
    };
  },
};
