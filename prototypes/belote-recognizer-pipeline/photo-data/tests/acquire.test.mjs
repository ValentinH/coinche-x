import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  canonicalLabels,
  readSourceConfig,
  remoteDirectory,
  selectEntries,
} from "../acquire.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

test("French labels map exactly to all 32 canonical Belote cards", async () => {
  const config = await readSourceConfig(join(root, "source.json"));
  assert.equal(config.labels.ranks.A, "1");
  assert.equal(config.labels.ranks.J, "Valet");
  assert.equal(config.labels.ranks.Q, "Dame");
  assert.equal(config.labels.ranks.K, "Roi");
  assert.equal(config.labels.suits.clubs, "Trefle");
  assert.equal(config.labels.suits.diamonds, "carreau");
  assert.equal(config.labels.suits.hearts, "Coeur");
  assert.equal(config.labels.suits.spades, "Pique");
  assert.equal(new Set(canonicalLabels()).size, 32);
  assert.equal(remoteDirectory(config, "A", "hearts"), "data/Coeur 1");
  assert.equal(remoteDirectory(config, "J", "diamonds"), "data/carreau Valet");
});

test("selection is deterministic, bandwidth-first, and filename-family grouped", async () => {
  const config = await readSourceConfig(join(root, "source.json"));
  const remote = [
    fake("data/Coeur 7/20240130_b.jpg", 30, "1"),
    fake("data/Coeur 7/20240130_a.jpg", 20, "2"),
    fake("data/Coeur 7/20240314_a.jpg", 40, "3"),
    fake("data/Coeur 7/IMG_1000.JPG", 50, "4"),
    fake("data/Coeur 7/IMG_20240521_100000.jpg", 55, "5"),
    fake("data/Coeur 7/apple0.jpg", 60, "6"),
    fake("data/Coeur 7/coeur 7.jpg", 70, "7"),
  ];
  const first = selectEntries(config, "7", "hearts", remote);
  const second = selectEntries(config, "7", "hearts", [...remote].reverse());
  assert.deepEqual(first, second);
  assert.equal(first[0].entry.path, "data/Coeur 7/20240130_a.jpg");
  const splitByFamily = new Map(
    first.map(({ family }) => [family.id, family.split]),
  );
  assert.equal(splitByFamily.size, config.selection.samplesPerClass);
  assert.equal(splitByFamily.get("march-20240314"), "validation");
  assert.equal(
    new Set(
      config.selection.captureFamilies
        .filter((family) => family.split === "validation")
        .map((family) => family.id),
    ).size,
    3,
  );
  assert.equal(splitByFamily.get("img-dated"), "validation");
});

test("selection skips hashes already assigned to another class", async () => {
  const config = await readSourceConfig(join(root, "source.json"));
  const oneFamilyConfig = {
    ...config,
    selection: {
      ...config.selection,
      captureFamilies: [
        {
          id: "dated",
          filePattern: "^dated-",
          split: "train",
        },
      ],
    },
  };
  const usedHashes = new Set();
  const first = selectEntries(
    oneFamilyConfig,
    "8",
    "clubs",
    [fake("data/clubs/dated-a.jpg", 10, "a")],
    usedHashes,
  );
  const second = selectEntries(
    oneFamilyConfig,
    "8",
    "spades",
    [
      fake("data/spades/dated-duplicate.jpg", 10, "a"),
      fake("data/spades/dated-next.jpg", 20, "b"),
    ],
    usedHashes,
  );
  assert.equal(first[0].entry.path, "data/clubs/dated-a.jpg");
  assert.equal(second[0].entry.path, "data/spades/dated-next.jpg");
  assert.equal(usedHashes.size, 2);
});

test("source is immutable, licensed, online-only, and isolated", async () => {
  const configText = await readFile(join(root, "source.json"), "utf8");
  const config = JSON.parse(configText);
  assert.match(config.dataset.commit, /^[0-9a-f]{40}$/);
  assert.equal(config.dataset.license, "mit");
  assert.match(config.dataset.fileBaseUrl, /^https:\/\/huggingface\.co\//);
  assert.equal(config.dataset.fileBaseUrl.includes("/resolve/main"), false);
  const excludedRoot = ["test-data", "card-recognition"].join("/");
  assert.equal(configText.includes(excludedRoot), false);
  assert.equal(configText.includes("greywyvern"), false);
});

function fake(path, size, hash) {
  return {
    type: "file",
    path,
    size,
    lfs: { oid: hash.repeat(64).slice(0, 64), size },
  };
}
