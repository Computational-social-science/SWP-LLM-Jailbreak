import { defineConfig } from "@playwright/test";

// e2e runs against the replay server, so it's deterministic and needs no
// model weights: playwright boots the recorded-session server and a
// throwaway vite on side ports, runs the flows, tears both down.
export default defineConfig({
  testDir: "e2e",
  fullyParallel: false, // one recorded session, tests share it politely
  use: {
    baseURL: "http://127.0.0.1:5199",
    viewport: { width: 1440, height: 900 },
  },
  webServer: [
    {
      command:
        "cd ../server && .venv/bin/python -m logitscope --replay tests/fixtures/replay.jsonl --port 8020",
      url: "http://127.0.0.1:8020/api/health",
      reuseExistingServer: !process.env.CI,
    },
    {
      // --host 127.0.0.1 matters: vite 6 binds plain `localhost`, which on
      // macos resolves to ::1 only, and playwright polls the ipv4 url forever
      command: "npm run dev -- --port 5199 --strictPort --host 127.0.0.1",
      url: "http://127.0.0.1:5199",
      reuseExistingServer: !process.env.CI,
      env: { LS_API: "127.0.0.1:8020" },
    },
  ],
});
