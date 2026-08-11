const manifest = await fetch("./manifest.json").then((response) => {
  if (!response.ok) throw new Error(`manifest: HTTP ${response.status}`);
  return response.json();
});

const gallery = document.querySelector("#gallery");
const template = document.querySelector("#card");
const suit = document.querySelector("#suit");
const split = document.querySelector("#split");
const failed = document.querySelector("#failed");

document.querySelector("#summary").textContent =
  `${manifest.usableCount}/${manifest.assetCount} usable · ` +
  `${manifest.failedCount} flagged · ${manifest.sourceBytes.toLocaleString()} source bytes · ` +
  `${manifest.dataset.commit.slice(0, 12)} pinned`;

function render() {
  gallery.replaceChildren();
  const entries = manifest.entries.filter(
    (entry) =>
      (!suit.value || entry.suit === suit.value) &&
      (!split.value || entry.split === split.value) &&
      (!failed.checked || !entry.usableForClassifier),
  );
  for (const entry of entries) {
    const node = template.content.cloneNode(true);
    node.querySelector(".label").textContent = entry.label;
    const status = node.querySelector(".status");
    status.textContent = entry.usableForClassifier ? "usable" : "flagged";
    status.dataset.ok = entry.usableForClassifier;
    node.querySelector(".photo").src = entry.gallery.photo;
    node.querySelector(".photo").alt = `${entry.label} source photo`;
    node.querySelector(".rectified").src = entry.gallery.card;
    node.querySelector(".rectified").alt = `${entry.label} rectified card`;
    node.querySelector(".top-left").src = entry.gallery.topLeft;
    node.querySelector(".top-left").alt = `${entry.label} top-left corner crop`;
    node.querySelector(".bottom-right").src = entry.gallery.bottomRight;
    node.querySelector(".bottom-right").alt =
      `${entry.label} bottom-right corner crop`;
    node.querySelector(".capture").textContent = entry.captureFamily;
    node.querySelector(".split").textContent = entry.split;
    node.querySelector(".score").textContent =
      `${entry.score.toFixed(3)} · ${entry.method}`;
    node.querySelector(".hash").textContent = entry.sha256.slice(0, 12);
    gallery.append(node);
  }
}

for (const control of [suit, split, failed]) {
  control.addEventListener("change", render);
}
render();
