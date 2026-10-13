// the whole scope in a browser, against the committed recording. every count
// asserted here comes from server/tests/fixtures/replay.jsonl: 8 trunk
// tokens, a fork at node 3 onto the second candidate, 8 branch tokens.

import { expect, test, type Page } from "@playwright/test";

const PROMPT = "the scope hums: ";
const ROW_H = 19; // DetailPanel bar rows — click math mirrors the component

async function generate(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.locator(".hud select")).toHaveValue("tiny-random");
  await expect(page.locator(".conn-open")).toBeVisible();
  await page.fill(".prompt-bar textarea", PROMPT);
  await page.keyboard.press("Enter");
  await expect(page.locator(".stream .chip:not(.chip-prompt)")).toHaveCount(8);
  await expect(page.locator(".stream-wrap .panel-title")).not.toContainText("generating");
}

test("prompt to pixels: stream, detail, waterfall", async ({ page }) => {
  await generate(page);

  // prompt bytes render as dim chips too
  await expect(page.locator(".chip-prompt")).toHaveCount(PROMPT.length);

  // click a token, the detail panel talks about it
  await page.locator(".stream .chip:not(.chip-prompt)").nth(2).click();
  await expect(page.locator(".detail-stats")).toContainText("rank");
  await expect(page.locator(".detail-canvas")).toBeVisible();

  // lens frames streamed with generation, so the waterfall is already drawn
  await expect(page.locator(".waterfall-scroll canvas")).toBeVisible();
});

test("fork, branch rail, diff, attention, replay", async ({ page }) => {
  await generate(page);
  const chips = page.locator(".stream .chip:not(.chip-prompt)");

  // fork node 3 onto its second candidate: select the chip, then click the
  // second bar in the detail panel
  await chips.nth(2).click();
  await page.locator(".detail-canvas").click({ position: { x: 30, y: 6 + ROW_H + 9 } });

  // the branch streams in and becomes the current path: 2 shared + forced + 7
  await expect(chips).toHaveCount(10);
  await expect(page.locator(".branch")).toHaveCount(2);
  await expect(page.locator(".branch-on")).toContainText("#16");

  // attention tab scrubs the new tip: full square over 16 prompt + 9 context
  await page.getByRole("button", { name: "attention" }).click();
  await expect(page.locator(".attn-status")).toHaveText("25×25");

  // diff tab: a = fork (10 tok), b = trunk (8 tok), 2 shared prefix
  await page.getByRole("button", { name: "fork diff" }).click();
  await expect(page.locator(".chip-del")).toHaveCount(8);
  await expect(page.locator(".chip-ins")).toHaveCount(6);

  // time travel back to the trunk; the diff flips with it
  await page.locator(".branch").first().click();
  await expect(chips).toHaveCount(8);
  await expect(page.locator(".chip-del")).toHaveCount(6);
  await expect(page.locator(".chip-ins")).toHaveCount(8);

  // replay re-scrubs every node on the path without error
  await page.getByRole("button", { name: "replay" }).click();
  await page.getByRole("button", { name: "logit lens waterfall" }).click();
  await expect(page.locator(".waterfall-scroll canvas")).toBeVisible();
  await expect(page.locator(".error")).toHaveCount(0);
});
