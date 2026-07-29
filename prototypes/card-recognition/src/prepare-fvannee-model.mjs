import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const prototypeDirectory = resolve(import.meta.dirname, "..");
const cacheDirectory = resolve(prototypeDirectory, ".cache");
const cloneDirectory = resolve(cacheDirectory, "fvannee-android");
const modelDirectory = resolve(cloneDirectory, "app/src/main/res/raw");
const sources = [
  ["cartamundi_samples_25.data", "cartamundi_responses_25.data"],
  ["samples_25.data", "responses_25.data"],
];

await mkdir(cacheDirectory, { recursive: true });
try {
  await readFile(resolve(cloneDirectory, "LICENSE"));
} catch {
  execFileSync(
    "gh",
    [
      "repo",
      "clone",
      "fvannee/android-cards-image-recognition",
      cloneDirectory,
      "--",
      "--depth=1",
    ],
    { stdio: "inherit" },
  );
}

const rows = [];
for (const [sampleFile, responseFile] of sources) {
  const samples = parseRows(
    await readFile(resolve(modelDirectory, sampleFile), "utf8"),
  );
  const responses = parseRows(
    await readFile(resolve(modelDirectory, responseFile), "utf8"),
  );
  if (samples.length !== responses.length) {
    throw new Error(`Modèle invalide : ${sampleFile}`);
  }
  for (let index = 0; index < samples.length; index += 1) {
    rows.push({
      label: responses[index][0],
      sample: samples[index],
    });
  }
}

const sampleSize = rows[0].sample.length;
const output = Buffer.alloc(8 + rows.length * (sampleSize + 1));
output.write("CFK1", 0);
output.writeUInt16LE(rows.length, 4);
output.writeUInt16LE(sampleSize, 6);
let offset = 8;
for (const { label, sample } of rows) {
  output.writeUInt8(Math.round(label), offset);
  offset += 1;
  for (const value of sample) {
    output.writeUInt8(Math.max(0, Math.min(255, Math.round(value))), offset);
    offset += 1;
  }
}

const outputPath = resolve(cacheDirectory, "fvannee-model.bin");
await writeFile(outputPath, output);
console.log(`${rows.length} gabarits, ${output.length} octets : ${outputPath}`);

function parseRows(source) {
  return source
    .trim()
    .split("\n")
    .map((line) => line.trim().split(/\s+/).map(Number));
}
