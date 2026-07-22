// SPDX-License-Identifier: MIT
// Puppeteer driver for the music_sound_generator recorder page.
//
// This project-authored integration launches system Chrome headless, opens the
// recorder over the local Vite server, waits for RESULT_JSON, and writes the
// returned WebM data. It calls the external Strudel API but contains no Strudel
// implementation code.
//
// Usage: node record_cue.mjs <cue> <seconds> <outDir> [outName]
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const cue = process.argv[2] || '';
const seconds = Number(process.argv[3] || 8);
const outDir = process.argv[4] || '.';
const outName = process.argv[5] || cue;
const port = Number(process.env.STRUDEL_PORT || 5398);
const chrome = process.env.CHROME_BIN || '/usr/bin/google-chrome';
const url =
  `http://127.0.0.1:${port}/music_sound_generator_assets/record.html` +
  `?cue=${encodeURIComponent(cue)}&seconds=${seconds}`;

if (!cue) {
  console.error('usage: node record_cue.mjs <cue> <seconds> <outDir> [outName]');
  process.exit(2);
}

fs.mkdirSync(outDir, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: 'new',
  args: [
    '--no-sandbox',
    '--disable-gpu',
    '--autoplay-policy=no-user-gesture-required',
    '--use-fake-ui-for-media-stream',
  ],
});

try {
  const page = await browser.newPage();
  page.on('pageerror', (err) => console.error(`[pageerror] ${err.stack || err.message || err}`));
  page.on('console', (msg) => {
    console.error(`[browser:${msg.type()}] ${msg.text()}`);
  });
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.mouse.click(12, 12);
  try {
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent?.startsWith('RESULT_JSON:') ||
            document.getElementById('status')?.textContent?.startsWith('ERROR_JSON:'),
      { timeout: Math.max(45000, (seconds + 25) * 1000) },
    );
  } catch (err) {
    const current = await page.$eval('#status', (node) => node.textContent).catch(() => '<missing status>');
    throw new Error(`Timed out waiting for recorder result; current status: ${current}`);
  }
  const statusText = await page.$eval('#status', (node) => node.textContent);
  if (statusText.startsWith('ERROR_JSON:')) {
    throw new Error(statusText.slice('ERROR_JSON:'.length));
  }
  const result = JSON.parse(statusText.slice('RESULT_JSON:'.length));
  const [, meta, b64] = result.dataUrl.match(/^data:([^,]+);base64,(.*)$/) || [];
  if (!b64) {
    throw new Error('Recorder did not return a base64 data URL');
  }
  const ext = meta.includes('webm') ? 'webm' : 'bin';
  const outPath = path.join(outDir, `${outName}.${ext}`);
  fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
  const { dataUrl, ...summary } = result;
  console.log(JSON.stringify({ ...summary, path: outPath }, null, 2));
} finally {
  await browser.close();
}
