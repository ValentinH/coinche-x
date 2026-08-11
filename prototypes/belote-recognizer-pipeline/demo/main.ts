interface Annotation {
  bbox: { x: number; y: number; width: number; height: number };
  card: { rank: string; suit: string };
  glyph: string;
  alphabet: string;
  visibleFraction: number;
}

interface Frame {
  image: string;
  width: number;
  height: number;
  cardCount: number;
  mode: string;
  annotations: Annotation[];
}

interface Dataset {
  split: string;
  seed: string;
  source: { id: string; role: string };
  coverage: { canonicalCards: string[]; glyphs: string[]; suits: string[] };
  frames: Frame[];
}

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const sceneSelect = document.querySelector<HTMLSelectElement>("#scene")!;
const sceneMeta = document.querySelector<HTMLElement>("#scene-meta")!;
const metrics = document.querySelector<HTMLElement>("#metrics")!;
const context = canvas.getContext("2d")!;
const suitSymbol: Record<string, string> = {
  clubs: "♣",
  diamonds: "♦",
  hearts: "♥",
  spades: "♠",
};

const dataset = (await fetch("/sample-data/dataset.json").then((response) => {
  if (!response.ok) throw new Error(`Dataset: HTTP ${response.status}`);
  return response.json();
})) as Dataset;

metrics.innerHTML = [
  ["Scenes", dataset.frames.length],
  ["Visible indices", dataset.frames.flatMap((frame) => frame.annotations).length],
  ["Cards covered", `${dataset.coverage.canonicalCards.length}/32`],
  ["Glyphs", dataset.coverage.glyphs.join(" · ")],
  ["Source role", dataset.source.role],
]
  .map(
    ([label, value]) =>
      `<article><span>${label}</span><strong>${value}</strong></article>`,
  )
  .join("");

for (const [index, frame] of dataset.frames.entries()) {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = `${String(index + 1).padStart(2, "0")} · ${frame.cardCount} card${frame.cardCount === 1 ? "" : "s"}`;
  sceneSelect.append(option);
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

async function render(index: number) {
  const frame = dataset.frames[index];
  const image = await loadImage(`/sample-data/${frame.image}`);
  canvas.width = frame.width;
  canvas.height = frame.height;
  context.drawImage(image, 0, 0);
  context.lineWidth = 5;
  context.font = "600 24px system-ui";
  context.textBaseline = "bottom";
  for (const annotation of frame.annotations) {
    const { x, y, width, height } = annotation.bbox;
    const label = `${annotation.glyph}→${annotation.card.rank}${suitSymbol[annotation.card.suit]}`;
    const textWidth = context.measureText(label).width;
    context.fillStyle = "rgba(7, 14, 19, 0.78)";
    context.fillRect(x - 2, Math.max(0, y - 32), textWidth + 16, 30);
    context.fillStyle = "#8ef4ed";
    context.fillText(label, x + 6, Math.max(28, y - 5));
    context.strokeStyle = "#22e0d3";
    context.strokeRect(x, y, width, height);
  }
  sceneMeta.textContent = `${frame.mode} · ${frame.annotations.length} visible indices · ${frame.width}×${frame.height}`;
}

sceneSelect.addEventListener("change", () => {
  void render(Number(sceneSelect.value));
});

await render(0);
