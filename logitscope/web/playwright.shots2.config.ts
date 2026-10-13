// throwaway playwright config: the real-model server is already running on 8000.
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:5198",
    viewport: { width: 1600, height: 1000 },
    launchOptions: {
      executablePath:
        process.env.LS_CHROME ??
        `${process.env.USERPROFILE}\\AppData\\Local\\ms-playwright\\chromium-1148\\chrome-win\\chrome.exe`,
    },
  },
  timeout: 300_000,
  webServer: [
    {
      command: "npm run dev -- --port 5198 --strictPort --host 127.0.0.1",
      url: "http://127.0.0.1:5198",
      reuseExistingServer: true,
      env: { LS_API: "127.0.0.1:8000" },
    },
  ],
});
