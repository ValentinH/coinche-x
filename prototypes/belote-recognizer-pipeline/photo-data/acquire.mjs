import { createHash } from "node:crypto";
import {
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { basename, dirname, extname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const photoDataRoot = dirname(fileURLToPath(import.meta.url));
const pipelineRoot = dirname(photoDataRoot);
const sourcePath = join(photoDataRoot, "source.json");
const defaultOutputRoot = join(pipelineRoot, "generated/photo-source");

export const ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"];
export const suits = ["clubs", "diamonds", "hearts", "spades"];

export function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

export async function readSourceConfig(path = sourcePath) {
  return JSON.parse(await readFile(path, "utf8"));
}

export function canonicalLabels() {
  return suits.flatMap((suit) => ranks.map((rank) => `${rank}-${suit}`));
}

export function remoteDirectory(config, rank, suit) {
  return `data/${config.labels.suits[suit]} ${config.labels.ranks[rank]}`;
}

function encodedPath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

export function captureFamilyMatchers(config) {
  return config.selection.captureFamilies.map((family) => ({
    ...family,
    matches: new RegExp(family.filePattern, family.flags ?? ""),
  }));
}

export function rankedFamilyCandidates(family, remoteEntries) {
  return remoteEntries
    .filter(
      (entry) =>
        entry.type === "file" && family.matches.test(basename(entry.path)),
    )
    .sort(
      (left, right) =>
        left.size - right.size || left.path.localeCompare(right.path),
    );
}

export function selectEntries(
  config,
  rank,
  suit,
  remoteEntries,
  usedHashes = new Set(),
) {
  return captureFamilyMatchers(config).map((family) => {
    const candidates = rankedFamilyCandidates(family, remoteEntries);
    if (candidates.length === 0) {
      throw new Error(`${rank}-${suit}: no ${family.id} source`);
    }
    const entry = candidates.find((candidate) => !usedHashes.has(candidate.lfs?.oid));
    if (!entry) {
      throw new Error(`${rank}-${suit}: no unique ${family.id} source`);
    }
    if (!entry.lfs?.oid || entry.lfs.size !== entry.size) {
      throw new Error(`${entry.path}: missing immutable LFS hash`);
    }
    usedHashes.add(entry.lfs.oid);
    return { entry, family };
  });
}

async function fetchWithRetry(url, init = {}, attempts = 4) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, {
        ...init,
        signal: AbortSignal.timeout(45_000),
      });
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status}: ${url}`);
    } catch (error) {
      lastError = error;
    }
    if (attempt < attempts) {
      await new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt));
    }
  }
  throw lastError;
}

async function listRemoteDirectory(config, directory) {
  const url =
    `https://huggingface.co/api/datasets/${config.dataset.id}/tree/` +
    `${config.dataset.commit}/${encodedPath(directory)}?limit=1000`;
  return fetchWithRetry(url).then((response) => response.json());
}

async function verifyLicense(config) {
  const metadata = await fetchWithRetry(config.dataset.metadataUrl).then((response) =>
    response.json(),
  );
  if (metadata.sha !== config.dataset.commit) {
    throw new Error(`metadata commit drift: ${metadata.sha}`);
  }
  if (
    metadata.cardData?.license !== config.dataset.license ||
    !metadata.tags?.includes(`license:${config.dataset.license}`)
  ) {
    throw new Error(`license metadata drift for ${config.dataset.id}`);
  }
}

async function validExistingFile(path, expectedSize, expectedHash) {
  try {
    if ((await stat(path)).size !== expectedSize) return false;
    return sha256(await readFile(path)) === expectedHash;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

async function downloadAsset(url, destination, expectedSize, expectedHash) {
  if (await validExistingFile(destination, expectedSize, expectedHash)) {
    return "reused";
  }
  await mkdir(dirname(destination), { recursive: true });
  const partial = `${destination}.partial`;
  await rm(partial, { force: true });
  const response = await fetchWithRetry(`${url}?download=true`);
  const data = Buffer.from(await response.arrayBuffer());
  const actualHash = sha256(data);
  if (data.length !== expectedSize || actualHash !== expectedHash) {
    throw new Error(
      `${url}: expected ${expectedSize}/${expectedHash}, got ${data.length}/${actualHash}`,
    );
  }
  await writeFile(partial, data);
  await rename(partial, destination);
  return "downloaded";
}

async function mapConcurrent(items, concurrency, action) {
  const results = new Array(items.length);
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await action(items[index], index);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, () => worker()),
  );
  return results;
}

export async function planAcquisition(
  config,
  {
    existingEntries = [],
    failedIds = new Set(),
  } = {},
) {
  const classes = [];
  const existingById = new Map(existingEntries.map((entry) => [entry.id, entry]));
  const usedHashes = new Set(existingEntries.map((entry) => entry.sha256));
  for (const suit of suits) {
    for (const rank of ranks) {
      const label = `${rank}-${suit}`;
      const directory = remoteDirectory(config, rank, suit);
      const entries = await listRemoteDirectory(config, directory);
      const selected = captureFamilyMatchers(config).map((family) => {
        const id = `${label}--${family.id}`;
        const existing = existingById.get(id);
        if (existing && !failedIds.has(id)) return existing;
        const candidates = rankedFamilyCandidates(family, entries);
        const previousIndex = existing
          ? candidates.findIndex((candidate) => candidate.path === existing.sourcePath)
          : -1;
        const entry = candidates
          .slice(previousIndex + 1)
          .find(
            (candidate) =>
              candidate.lfs?.oid &&
              candidate.lfs.size === candidate.size &&
              !usedHashes.has(candidate.lfs.oid),
          );
        if (!entry) {
          if (existing) {
            return {
              ...existing,
              fallbackExhausted: true,
            };
          }
          throw new Error(`${id}: no immutable candidate`);
        }
        usedHashes.add(entry.lfs.oid);
        return {
          id: `${label}--${family.id}`,
          label,
          rank,
          suit,
          captureFamily: family.id,
          split: family.split,
          sourcePath: entry.path,
          sourceUrl: `${config.dataset.fileBaseUrl}/${encodedPath(entry.path)}`,
          bytes: entry.size,
          sha256: entry.lfs.oid,
          extension: extname(entry.path).toLowerCase(),
          fallbackAttempt: existing ? (existing.fallbackAttempt ?? 0) + 1 : 0,
        };
      });
      classes.push({ label, rank, suit, selected });
    }
  }
  return classes.flatMap((cardClass) => cardClass.selected);
}

export async function acquire({
  outputRoot = process.env.PHOTO_DATA_OUT || defaultOutputRoot,
  concurrency = Number(process.env.PHOTO_DATA_CONCURRENCY ?? 4),
  replaceFailed = false,
} = {}) {
  const config = await readSourceConfig();
  await verifyLicense(config);
  let existingEntries = [];
  let failedIds = new Set();
  let existingManifestEntries = [];
  let rejectedLabelEntries = [];
  if (replaceFailed) {
    const existing = JSON.parse(
      await readFile(join(outputRoot, "provenance.json"), "utf8"),
    );
    const manifest = JSON.parse(
      await readFile(join(outputRoot, "gallery/manifest.json"), "utf8"),
    );
    existingEntries = existing.entries;
    existingManifestEntries = manifest.entries;
    rejectedLabelEntries = existing.rejectedLabelEntries ?? [];
    failedIds = new Set(
      manifest.entries
        .filter((entry) => !entry.usableForClassifier)
        .map((entry) => entry.id),
    );
    if (failedIds.size === 0) {
      throw new Error("No failed photo sources to replace");
    }
  }
  const entries = await planAcquisition(config, {
    existingEntries,
    failedIds,
  });
  const manifestById = new Map(
    existingManifestEntries.map((entry) => [entry.id, entry]),
  );
  for (const existingEntry of existingEntries) {
    const rejected = manifestById.get(existingEntry.id);
    if (rejected?.method !== "flagged-label-mismatch") {
      continue;
    }
    rejectedLabelEntries.push({
      ...existingEntry,
      rejection: {
        method: rejected.method,
        expectedLabel: rejected.label,
        inferredLabel:
          rejected.diagnostics?.labelConsensus?.inferredLabel ?? null,
        leaveCaptureFamilyOut:
          rejected.diagnostics?.labelConsensus?.leaveCaptureFamilyOut ?? null,
      },
    });
  }
  rejectedLabelEntries = [
    ...new Map(
      rejectedLabelEntries.map((entry) => [entry.sha256, entry]),
    ).values(),
  ].sort(
    (left, right) =>
      left.sha256.localeCompare(right.sha256) ||
      left.id.localeCompare(right.id),
  );
  const expectedCount =
    canonicalLabels().length * config.selection.samplesPerClass;
  if (entries.length !== expectedCount) {
    throw new Error(`expected ${expectedCount} sources, selected ${entries.length}`);
  }

  const statuses = await mapConcurrent(entries, concurrency, async (entry) => {
    const extension = entry.extension === ".jpeg" ? ".jpg" : entry.extension;
    const relativePath = `raw/${entry.split}/${entry.label}/${entry.captureFamily}${extension}`;
    const destination = join(outputRoot, relativePath);
    const status = await downloadAsset(
      entry.sourceUrl,
      destination,
      entry.bytes,
      entry.sha256,
    );
    return { ...entry, localPath: relativePath, status };
  });

  const provenance = {
    schemaVersion: 1,
    createdBy: "photo-data/acquire.mjs",
    dataset: config.dataset,
    selection: config.selection,
    assetCount: statuses.length,
    sourceBytes: statuses.reduce((sum, entry) => sum + entry.bytes, 0),
    entries: statuses.map(({ status: _status, ...entry }) => entry),
    rejectedLabelEntries,
  };
  await mkdir(outputRoot, { recursive: true });
  await writeFile(
    join(outputRoot, "provenance.json"),
    `${JSON.stringify(provenance, null, 2)}\n`,
  );
  await writeFile(
    join(outputRoot, "assets.sha256"),
    `${provenance.entries
      .map((entry) => `${entry.sha256}  ${entry.localPath}`)
      .sort()
      .join("\n")}\n`,
  );

  const downloaded = statuses.filter((entry) => entry.status === "downloaded").length;
  const reused = statuses.length - downloaded;
  const exhausted = statuses.filter((entry) => entry.fallbackExhausted).length;
  console.log(
    `Photo source ready: ${statuses.length} assets, ` +
      `${provenance.sourceBytes} bytes; downloaded ${downloaded}, reused ${reused}` +
      `${replaceFailed ? `, attempted ${failedIds.size}, exhausted ${exhausted}` : ""}.`,
  );
  return provenance;
}

const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  await acquire({ replaceFailed: process.argv.includes("--replace-failed") });
}
