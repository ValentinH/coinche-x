import { placeholderAdapter } from "../recognition/index.ts";
import { escapeHtml, milliseconds, percent } from "../ui/html.ts";
import type {
  EvaluationDataset,
  EvaluationProgress,
  EvaluationRun,
  SampleEvaluation,
} from "./contracts.ts";
import { evaluateBatch } from "./evaluate-batch.ts";
import { loadSmokeDatasetV2 } from "./smoke-manifest-v2.ts";

export function mountEvaluationPage(root: HTMLElement): void {
  let dataset: EvaluationDataset | undefined;
  const runs: EvaluationRun[] = [];

  root.innerHTML = `
    <div class="app-shell">
      <header class="hero">
        <div>
          <p class="eyebrow">Route développement · données locales</p>
          <h1>Banc de reconnaissance</h1>
          <p class="lede">
            Intégrité, prédictions, échecs, précision exacte, précision carte et latence.
          </p>
        </div>
        <a class="text-link" href="/">← Test unitaire</a>
      </header>

      <section class="panel setup-panel">
        <div class="panel-heading">
          <div>
            <p class="step">01 · Charger</p>
            <h2>Corpus smoke v2</h2>
          </div>
          <span class="status">Local uniquement</span>
        </div>
        <div class="picker-grid">
          <label class="file-picker">
            <span>Manifeste JSON</span>
            <input id="manifest-file" type="file" accept="application/json,.json" />
          </label>
          <label class="file-picker">
            <span>Images du corpus</span>
            <input id="image-files" type="file" accept="image/*" multiple webkitdirectory />
          </label>
        </div>
        <div class="action-row">
          <button id="load-corpus" class="secondary-button" type="button">
            Vérifier la structure
          </button>
          <button id="run-evaluation" class="primary-button" type="button" disabled>
            Lancer l’évaluation
          </button>
          <p id="load-status" class="muted" aria-live="polite">
            Sélectionnez le manifeste et les images. SHA-256 sera vérifié avant chaque inférence.
          </p>
        </div>
        <div id="run-progress"></div>
      </section>

      <div id="evaluation-dashboard">
        <section class="panel empty-state">
          <span class="empty-glyph">↗</span>
          <p>Aucun run. Les résultats restent en mémoire.</p>
        </section>
      </div>
    </div>
  `;

  const manifestInput = requireElement<HTMLInputElement>(
    root,
    "#manifest-file",
  );
  const imageInput = requireElement<HTMLInputElement>(root, "#image-files");
  const loadButton = requireElement<HTMLButtonElement>(root, "#load-corpus");
  const runButton = requireElement<HTMLButtonElement>(root, "#run-evaluation");
  const loadStatus = requireElement<HTMLElement>(root, "#load-status");
  const progress = requireElement<HTMLElement>(root, "#run-progress");
  const dashboard = requireElement<HTMLElement>(
    root,
    "#evaluation-dashboard",
  );

  loadButton.addEventListener("click", async () => {
    const manifest = manifestInput.files?.[0];
    const images = [...(imageInput.files ?? [])];

    if (!manifest || images.length === 0) {
      loadStatus.textContent = "Manifeste et images requis.";
      loadStatus.className = "error-text";
      return;
    }

    try {
      dataset = await loadSmokeDatasetV2(manifest, images);
      loadStatus.textContent = `${dataset.corpusId} · ${dataset.samples.length} image(s) prêtes.`;
      loadStatus.className = "success-text";
      runButton.disabled = false;
    } catch (error) {
      dataset = undefined;
      runButton.disabled = true;
      loadStatus.textContent =
        error instanceof Error ? error.message : String(error);
      loadStatus.className = "error-text";
    }
  });

  runButton.addEventListener("click", async () => {
    if (!dataset) return;
    runButton.disabled = true;
    loadButton.disabled = true;

    try {
      const run = await evaluateBatch(dataset, placeholderAdapter, {
        onProgress: (state) => renderProgress(progress, state),
      });
      runs.push(run);
      renderEvaluationDashboard(dashboard, runs);
    } catch (error) {
      progress.innerHTML = `<p class="error-callout">${escapeHtml(
        error instanceof Error ? error.message : String(error),
      )}</p>`;
    } finally {
      runButton.disabled = false;
      loadButton.disabled = false;
    }
  });
}

export function renderEvaluationDashboard(
  root: HTMLElement,
  runs: readonly EvaluationRun[],
): void {
  const latest = runs.at(-1);
  if (!latest) return;

  const metrics = latest.metrics;
  const failures = latest.samples.filter((sample) => !sample.diff.exact);

  root.innerHTML = `
    ${
      latest.valid
        ? ""
        : `<p class="error-callout">Run invalide : ${metrics.integrityFailures} échec(s) d’intégrité. Les images concernées n’ont pas été inférées.</p>`
    }
    <section class="metric-grid" aria-label="Métriques du dernier run">
      ${metricCard("Photos exactes", `${metrics.exactSets}/${metrics.totalSamples}`, percent(metrics.exactSetAccuracy))}
      ${metricCard("Précision carte", `${metrics.matchedCards} correctes`, percent(metrics.cardAccuracy))}
      ${metricCard("Latence P50", milliseconds(metrics.latency.p50Ms), `P95 ${milliseconds(metrics.latency.p95Ms)}`)}
      ${metricCard("Échecs", String(failures.length), `${metrics.missingCards} manquantes · ${metrics.extraPredictions} extras`)}
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="step">02 · Comprendre</p>
          <h2>Échecs (${failures.length})</h2>
        </div>
        <span class="status ${failures.length === 0 ? "status-success" : "status-warning"}">
          ${failures.length === 0 ? "Tout exact" : "À examiner"}
        </span>
      </div>
      ${
        failures.length === 0
          ? `<p class="success-text">Aucun échec.</p>`
          : `<div class="failure-list">${failures.map(renderFailure).join("")}</div>`
      }
    </section>

    <section class="panel table-panel">
      <div class="panel-heading">
        <div>
          <p class="step">03 · Auditer</p>
          <h2>Toutes les prédictions</h2>
        </div>
        <span class="status">${escapeHtml(latest.adapterId)}</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Image</th>
              <th>Vérité</th>
              <th>Prédiction</th>
              <th>Écart</th>
              <th>Latence</th>
            </tr>
          </thead>
          <tbody>${latest.samples.map(renderPredictionRow).join("")}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="step">04 · Suivre</p>
          <h2>Évolution des runs</h2>
        </div>
        <span class="status">${runs.length} run(s)</span>
      </div>
      <div class="run-evolution">
        ${runs.map((run, index) => renderRunEvolution(run, index)).join("")}
      </div>
    </section>
  `;
}

function renderProgress(
  root: HTMLElement,
  progress: EvaluationProgress,
): void {
  const ratio = progress.total === 0 ? 0 : progress.completed / progress.total;
  root.innerHTML = `
    <div class="progress-copy">
      <span>${progress.completed}/${progress.total}</span>
      <span>${escapeHtml(progress.filename)}</span>
    </div>
    <div class="progress-track"><span style="width:${ratio * 100}%"></span></div>
  `;
}

function metricCard(label: string, value: string, detail: string): string {
  return `
    <article class="metric-card">
      <p>${label}</p>
      <strong>${value}</strong>
      <span>${detail}</span>
    </article>
  `;
}

function renderFailure(sample: SampleEvaluation): string {
  return `
    <article class="failure-item">
      <div>
        <strong>${escapeHtml(sample.filename)}</strong>
        <span>${escapeHtml(failureLabel(sample))}</span>
      </div>
      <div class="diff-line">
        <span class="missing">− ${cardList(sample.diff.missingCards)}</span>
        <span class="extra">+ ${cardList([
          ...sample.diff.extraCards,
          ...sample.diff.duplicateCards,
        ])}</span>
      </div>
    </article>
  `;
}

function renderPredictionRow(sample: SampleEvaluation): string {
  const differences =
    sample.diff.missingCards.length +
    sample.diff.extraCards.length +
    sample.diff.duplicateCards.length +
    sample.diff.invalidCardIds.length;

  return `
    <tr>
      <td><strong>${escapeHtml(sample.filename)}</strong><small>${statusLabel(sample)}</small></td>
      <td>${cardList(sample.expectedCards)}</td>
      <td>${cardList(sample.predictedCards)}</td>
      <td><span class="${differences === 0 ? "success-text" : "error-text"}">${differences === 0 ? "Exact" : `${differences} écart(s)`}</span></td>
      <td>${sample.latencyMs === undefined ? "—" : milliseconds(sample.latencyMs)}</td>
    </tr>
  `;
}

function renderRunEvolution(run: EvaluationRun, index: number): string {
  return `
    <article class="run-row">
      <div><strong>Run ${index + 1}</strong><span>${new Date(run.startedAt).toLocaleTimeString("fr-FR")}</span></div>
      <div class="run-bar"><span style="width:${run.metrics.exactSetAccuracy * 100}%"></span></div>
      <strong>${percent(run.metrics.exactSetAccuracy)}</strong>
      <span>${percent(run.metrics.cardAccuracy)} cartes</span>
      <span>P95 ${milliseconds(run.metrics.latency.p95Ms)}</span>
    </article>
  `;
}

function cardList(cards: readonly string[]): string {
  if (cards.length === 0) return "∅";
  return cards.map((card) => escapeHtml(card)).join(" ");
}

function statusLabel(sample: SampleEvaluation): string {
  if (sample.status === "integrity-failure") return "Intégrité échouée";
  if (sample.status === "recognition-failure") return "Inférence échouée";
  return sample.diff.exact ? "Exact" : "Évaluée";
}

function failureLabel(sample: SampleEvaluation): string {
  if (sample.failure) return sample.failure;
  const parts: string[] = [];
  if (sample.diff.missingCards.length > 0)
    parts.push(`${sample.diff.missingCards.length} manquante(s)`);
  if (sample.diff.extraCards.length > 0)
    parts.push(`${sample.diff.extraCards.length} extra(s)`);
  if (sample.diff.duplicateCards.length > 0)
    parts.push(`${sample.diff.duplicateCards.length} doublon(s)`);
  if (sample.diff.invalidCardIds.length > 0)
    parts.push(`${sample.diff.invalidCardIds.length} invalide(s)`);
  return parts.join(" · ");
}

function requireElement<T extends Element>(
  root: ParentNode,
  selector: string,
): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Élément introuvable : ${selector}`);
  return element;
}
