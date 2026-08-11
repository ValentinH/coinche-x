import { defineConfig } from "vite";

export default defineConfig({
  server: {
    fs: {
      strict: true,
    },
  },
  build: {
    sourcemap: true,
  },
});
