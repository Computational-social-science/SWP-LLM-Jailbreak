import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// the server owns /api and /ws; the vite dev server just proxies so there is
// no cors dance during development. LS_API lets the e2e point a throwaway
// vite at a replay server on another port without touching this file.
const api = process.env.LS_API ?? "127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": `http://${api}`,
      "/ws": { target: `ws://${api}`, ws: true },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
