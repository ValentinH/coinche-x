import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdtemp,
  mkdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const provenance = JSON.parse(
  await readFile(join(root, "provenance/sources.json"), "utf8"),
);
const ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"];
const suits = ["clubs", "diamonds", "hearts", "spades"];

function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

function trainingEntry(rank, suit) {
  const rankNumber = { J: "11", Q: "12", K: "13", A: "1" }[rank] ?? rank;
  const suitName = {
    clubs: "Clubs",
    diamonds: "Diamond",
    hearts: "Hearts",
    spades: "Spades",
  }[suit];
  return `PNG/Large/${suitName} ${rankNumber}.png`;
}

function holdoutEntry(rank, suit) {
  const rankNumber = { J: "11", Q: "12", K: "13", A: "01" }[rank] ?? rank;
  const suitLetter = { clubs: "C", diamonds: "D", hearts: "H", spades: "S" }[
    suit
  ];
  return `greywyvern-cardset/${suitLetter}${rankNumber.padStart(2, "0")}.png`;
}

async function download(source, temporaryDirectory) {
  const response = await fetch(source.archiveUrl, {
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) {
    throw new Error(`${source.id}: HTTP ${response.status}`);
  }
  const archive = Buffer.from(await response.arrayBuffer());
  const actualHash = sha256(archive);
  if (actualHash !== source.archiveSha256) {
    throw new Error(
      `${source.id}: expected ${source.archiveSha256}, got ${actualHash}`,
    );
  }
  const archivePath = join(temporaryDirectory, basename(new URL(source.archiveUrl).pathname));
  await writeFile(archivePath, archive);
  return archivePath;
}

function readArchiveEntry(archivePath, entry) {
  return execFileSync("unzip", ["-p", archivePath, entry], {
    maxBuffer: 5 * 1024 * 1024,
  });
}

const temporaryDirectory = await mkdtemp(join(tmpdir(), "belote-assets-"));
const assetHashes = [];

try {
  for (const source of provenance.sources) {
    const archivePath = await download(source, temporaryDirectory);
    const targetDirectory = join(root, source.assetPath);
    await mkdir(targetDirectory, { recursive: true });

    for (const suit of suits) {
      for (const rank of ranks) {
        const entry =
          source.role === "training"
            ? trainingEntry(rank, suit)
            : holdoutEntry(rank, suit);
        const data = readArchiveEntry(archivePath, entry);
        if (data.length === 0) throw new Error(`${source.id}: empty ${entry}`);
        const destination = join(targetDirectory, `${rank}-${suit}.png`);
        await writeFile(destination, data);
        assetHashes.push(
          `${sha256(data)}  ${relative(root, destination).replaceAll("\\", "/")}`,
        );
      }
    }

    const licenseEntry =
      source.role === "training" ? "License.txt" : "greywyvern-cardset/licence.txt";
    const licenseDestination = join(
      root,
      `provenance/LICENSE.${source.id}.txt`,
    );
    await mkdir(dirname(licenseDestination), { recursive: true });
    await writeFile(licenseDestination, readArchiveEntry(archivePath, licenseEntry));
  }

  await writeFile(
    join(root, "provenance/assets.sha256"),
    `${assetHashes.sort().join("\n")}\n`,
  );
  console.log(`Acquired ${assetHashes.length} verified card assets.`);
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}
