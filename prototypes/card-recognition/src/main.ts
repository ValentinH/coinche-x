import "./styles.css";
import { mountRecognitionPage } from "./recognition/mount-recognition-page.ts";

const root = document.querySelector<HTMLElement>("#app");
if (!root) throw new Error("#app introuvable");

if (window.location.pathname === "/evaluate") {
  if (import.meta.env.DEV) {
    void import("./evaluation/mount-evaluation-page.ts").then(
      ({ mountEvaluationPage }) => mountEvaluationPage(root),
    );
  } else {
    root.innerHTML =
      '<div class="app-shell"><p class="error-callout">Route d’évaluation désactivée en production.</p><a class="text-link" href="/">Retour</a></div>';
  }
} else {
  mountRecognitionPage(root);
}
