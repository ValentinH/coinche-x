import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const temporaryDirectory = await mkdtemp(join(tmpdir(), "belote-pipeline-test-"));
const trainA = join(temporaryDirectory, "train-a");
const trainB = join(temporaryDirectory, "train-b");
const holdout = join(temporaryDirectory, "holdout");

function run(command, arguments_) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, arguments_, {
      cwd: root,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk;
    });
    child.stderr.on("data", (chunk) => {
      output += chunk;
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve(output);
      else reject(new Error(output));
    });
  });
}

try {
  await Promise.all([
    run("node", [
      "scripts/generate-dataset.mjs",
      "--split",
      "train",
      "--count",
      "4",
      "--out",
      trainA,
    ]),
    run("node", [
      "scripts/generate-dataset.mjs",
      "--split",
      "train",
      "--count",
      "4",
      "--out",
      trainB,
    ]),
    run("node", [
      "scripts/generate-dataset.mjs",
      "--split",
      "holdout",
      "--count",
      "4",
      "--out",
      holdout,
    ]),
  ]);
  const firstChecksums = await readFile(join(trainA, "dataset.sha256"), "utf8");
  const secondChecksums = await readFile(join(trainB, "dataset.sha256"), "utf8");
  if (firstChecksums !== secondChecksums) {
    throw new Error("Same seed produced different bytes");
  }
  const verification = await run("node", [
    "scripts/verify-dataset.mjs",
    trainA,
    holdout,
  ]);
  process.stdout.write(`Determinism: identical\n${verification}`);
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}
