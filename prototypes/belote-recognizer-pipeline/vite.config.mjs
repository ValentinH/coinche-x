import { fileURLToPath, URL } from "node:url";

export default {
  root: fileURLToPath(new URL("./demo", import.meta.url)),
  build: {
    outDir: fileURLToPath(new URL("./dist", import.meta.url)),
    emptyOutDir: true,
  },
};
