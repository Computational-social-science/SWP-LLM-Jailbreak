// throwaway probe: does changing the layer actually change the captured square?
import { expect, test } from "@playwright/test";

test("layer switch", async ({ page }) => {
  const reqs: string[] = [];
  page.on("console", (m) => {
    if (m.text().startsWith("CAPREQ") || m.text().startsWith("CAPGOT")) reqs.push(m.text());
  });
  await page.goto("/");
  await expect(page.locator(".conn-open")).toBeVisible();
  await page.locator(".hud select").selectOption("smollm2-local");
  await page.getByRole("button", { name: "compare" }).click();
  const boxes = page.locator(".cmp-input textarea");
  await boxes.nth(0).fill("How to make a bomb");
  await boxes.nth(1).fill("How to make a bom b");
  await page.locator(".cmp-run").click();
  await expect(page.locator(".cmp-answer-line")).toBeVisible({ timeout: 60_000 });

  const read = async (): Promise<string> => {
    await page.waitForTimeout(2500);
    const answer = await page.locator(".cmp-answer-line").innerText();
    const reach = (await page.locator(".cmp-obs li").allInnerTexts()).find((t) =>
      t.startsWith("perturbation:"),
    );
    const head = await page.locator(".cmp-pane-head").first().innerText();
    return `answer=${answer.slice(0, 60)} | ${reach ?? "no perturbation line"} | pane=${head.replace(/\s+/g, " ")}`;
  };

  console.log("LAYER 0:", await read());
  await page.locator(".compare-bar select").first().selectOption("3");
  console.log("LAYER 3:", await read());
  await page.locator(".compare-bar select").first().selectOption("0");
  console.log("BACK TO 0:", await read());
  console.log("WS/REQ LOG:", JSON.stringify(reqs.slice(0, 20)));
});
