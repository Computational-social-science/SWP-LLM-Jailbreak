// screenshot rig: boots the real server + a throwaway vite, runs one real
// generation on smollm2, and captures the four readme shots into docs/.
// deterministic-ish: the sampler is seeded, so the same machine reproduces
// the same text. needs smollm2 weights in the hub cache already.
//
//   node scripts/shots.mjs
//
// (not a playwright *test* on purpose — e2e stays weight-free; this rig is
// only for regenerating the pictures.)

import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const docs = resolve(repo, "docs");
// 8021 looks free but launchd squats on it (socket-activated something) and
// the bind fails with EADDRINUSE. 8022 is genuinely free.
const SERVER = 8022;
const WEB = 5198;

const PROMPT = "The oscilloscope on my desk has started showing me things that are not signals.";
const STREAM_PICK = 5; // chip selected for the stream + bars shot
const FORK_PICK = 4; // the high-entropy chip — forking it diverges nicely
const FORK_ROW = 1; // onto its second-best candidate
const ATTN_LAYER = "15";

const procs = [];
let browser = null;
const boot = (cmd, args, opts) => {
  const p = spawn(cmd, args, { stdio: "ignore", detached: true, ...opts });
  procs.push(p);
  return p;
};

async function up(url, timeoutMs = 60_000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {
      /* not yet */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`${url} never came up`);
}

const idle = (page) =>
  page.waitForFunction(
    () => !document.querySelector(".stream-wrap .panel-title")?.textContent?.includes("generating"),
    undefined,
    { timeout: 120_000 },
  );

async function main() {
  mkdirSync(docs, { recursive: true });

  boot(resolve(repo, "server/.venv/bin/python"), ["-m", "logitscope", "--port", String(SERVER)], {
    cwd: resolve(repo, "server"),
  });
  await up(`http://127.0.0.1:${SERVER}/api/health`);
  console.log("server up, loading smollm2…");
  const load = await fetch(`http://127.0.0.1:${SERVER}/api/models/load`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name: "smollm2-135m" }),
  });
  if (!load.ok) throw new Error(`model load failed: ${await load.text()}`);

  boot(resolve(web, "node_modules/.bin/vite"), ["--port", String(WEB), "--strictPort", "--host", "127.0.0.1"], {
    cwd: web,
    env: { ...process.env, LS_API: `127.0.0.1:${SERVER}` },
  });
  await up(`http://127.0.0.1:${WEB}/`);

  browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  await page.goto(`http://127.0.0.1:${WEB}/`);
  await page.waitForSelector(".conn-open");

  console.log("generating…");
  await page.fill(".prompt-bar textarea", PROMPT);
  await page.keyboard.press("Enter");
  await page.waitForSelector(".stream .chip:not(.chip-prompt)");
  await idle(page);
  const chips = page.locator(".stream .chip:not(.chip-prompt)");

  // 1. stream + what-it-almost-said bars (everything above the bottom panel)
  await chips.nth(STREAM_PICK).click();
  await page.waitForSelector(".detail-canvas");
  await page.waitForTimeout(400); // token labels for the bars arrive async
  const bottom = await page.locator(".bottom.panel").boundingBox();
  await page.screenshot({
    path: resolve(docs, "stream.png"),
    clip: { x: 0, y: 0, width: 1440, height: Math.round(bottom.y) },
  });
  console.log("stream.png");

  // 2. the waterfall strip
  await page.locator(".bottom.panel").screenshot({ path: resolve(docs, "waterfall.png") });
  console.log("waterfall.png");

  // 3. attention: heatmap below, ribbons over the text
  await page.getByRole("button", { name: "attention" }).click();
  await page.waitForFunction(() => /×/.test(document.querySelector(".attn-status")?.textContent ?? ""));
  await page.locator(".attn-side select").first().selectOption(ATTN_LAYER);
  await page.waitForFunction(() => !(document.querySelector(".attn-status")?.textContent ?? "").includes("computing"));
  // the model stops when it stops — hover a late chip, wherever "late" is today
  await chips.nth(Math.min(30, (await chips.count()) - 2)).hover();
  await page.waitForTimeout(350);
  await page.screenshot({ path: resolve(docs, "attention.png") });
  console.log("attention.png");

  // 4. fork the timeline, then the side-by-side diff
  await chips.nth(FORK_PICK).click();
  await page.waitForSelector(".detail-canvas");
  await page.locator(".detail-canvas").click({ position: { x: 30, y: 6 + FORK_ROW * 19 + 9 } });
  await idle(page);
  await page.waitForSelector(".branch-on");
  await page.getByRole("button", { name: "fork diff" }).click();
  await page.waitForSelector(".chip-ins");
  await page.waitForTimeout(400);
  await page.screenshot({ path: resolve(docs, "forkdiff.png") });
  console.log("forkdiff.png");
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(async () => {
    // close the browser even on the error path or node never exits
    await browser?.close().catch(() => {});
    for (const p of procs) {
      try {
        process.kill(-p.pid, "SIGTERM"); // the whole group: vite spawns children
      } catch {
        /* already gone */
      }
    }
  });
