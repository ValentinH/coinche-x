import { placeholderAdapter } from "./placeholder-adapter.ts";
import { recognizeImage } from "./recognize-image.ts";

export function mountRecognitionPage(root: HTMLElement): void {
  root.innerHTML = `
    <div class="app-shell narrow-shell">
      <header class="hero">
        <div>
          <p class="eyebrow">Prototype jetable · navigateur uniquement</p>
          <h1>Une photo. Un ensemble exact.</h1>
          <p class="lede">
            Le contrat public renvoie des cartes canoniques uniques et des diagnostics exploitables.
          </p>
        </div>
        <a class="text-link" href="/evaluate">Ouvrir le banc →</a>
      </header>

      <section class="panel capture-panel">
        <div class="panel-heading">
          <div>
            <p class="step">Entrée image</p>
            <h2>Tester le seam de reconnaissance</h2>
          </div>
          <span class="status status-warning">Placeholder explicite</span>
        </div>
        <p class="muted">
          Le moteur actuel est déterministe et ne prédit rien. Aucun score factice.
        </p>
        <label class="drop-zone">
          <span>Choisir une photo</span>
          <small>Traitée localement, jamais envoyée</small>
          <input id="single-image" type="file" accept="image/*" />
        </label>
        <button id="recognize" class="primary-button" type="button" disabled>
          Reconnaître
        </button>
      </section>

      <section id="single-result" class="panel result-panel" aria-live="polite">
        <div class="empty-state">
          <span class="empty-glyph">◇</span>
          <p>La prédiction et ses diagnostics apparaîtront ici.</p>
        </div>
      </section>
    </div>
  `;

  const input = requireElement<HTMLInputElement>(root, "#single-image");
  const button = requireElement<HTMLButtonElement>(root, "#recognize");
  const result = requireElement<HTMLElement>(root, "#single-result");

  input.addEventListener("change", () => {
    button.disabled = !input.files?.[0];
  });

  button.addEventListener("click", async () => {
    const file = input.files?.[0];
    if (!file) return;

    button.disabled = true;
    button.textContent = "Analyse…";

    try {
      const prediction = await recognizeImage(file, placeholderAdapter);
      const messages = prediction.diagnostics.messages
        .map(
          (message) =>
            `<li><code>${message.code}</code><span>${message.message}</span></li>`,
        )
        .join("");
      result.innerHTML = `
        <div class="panel-heading">
          <div>
            <p class="step">Sortie canonique</p>
            <h2>${prediction.cards.length} carte(s)</h2>
          </div>
          <span class="metric-value">${Math.round(prediction.diagnostics.latencyMs)} ms</span>
        </div>
        <div class="card-list">
          ${
            prediction.cards.length > 0
              ? prediction.cards
                  .map((card) => `<span class="card-chip">${card}</span>`)
                  .join("")
              : `<span class="muted">Ensemble vide</span>`
          }
        </div>
        <h3>Diagnostics</h3>
        <ul class="diagnostic-list">${messages}</ul>
      `;
    } catch (error) {
      result.innerHTML = `<p class="error-callout">${error instanceof Error ? error.message : String(error)}</p>`;
    } finally {
      button.disabled = false;
      button.textContent = "Reconnaître";
    }
  });
}

function requireElement<T extends Element>(
  root: ParentNode,
  selector: string,
): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Élément introuvable : ${selector}`);
  return element;
}
