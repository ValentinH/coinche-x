import { resolve } from "node:path";

import { defineConfig } from "vite";

export default defineConfig({
  root: resolve(import.meta.dirname, "../.."),
  cacheDir: resolve(import.meta.dirname, ".cache/vite"),
});
