/**
 * Packs the libraries (`pnpm pack`), installs the tarballs into a fresh temporary project and
 * checks that every runtime export resolves and imports in Node, the worker entry is shipped,
 * all type declarations resolve for both `Bundler` and `NodeNext` module resolution, and — in
 * a small Vite app consuming the installed package (dev server with dependency pre-bundling,
 * and production build) — the default detector worker URL loads and detects a document in
 * headless Chrome (`channel: 'chrome'`, override with CHROME_PATH).
 * Run after `pnpm build`: `pnpm check:pack`.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright-core';

const root = fileURLToPath(new URL('..', import.meta.url));
const tmp = mkdtempSync(join(tmpdir(), 'scanoficus-pack-'));
const run = (cmd, args, cwd) => {
  try {
    return execFileSync(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'inherit'] }).toString();
  } catch (error) {
    throw new Error(`${cmd} ${args.join(' ')} failed:\n${error.stdout ?? ''}`);
  }
};

try {
  const tarballs = {};
  for (const name of ['core', 'elements', 'react', 'detector-nn']) {
    run('pnpm', ['pack', '--pack-destination', tmp], join(root, 'packages', name));
    const file = readdirSync(tmp).find(
      (f) => f.startsWith(`scanoficus-${name}-`) && f.endsWith('.tgz'),
    );
    tarballs[`@scanoficus/${name}`] = `file:${join(tmp, file)}`;
  }
  const reactDev = JSON.parse(
    readFileSync(join(root, 'packages/react/package.json'), 'utf8'),
  ).devDependencies;
  writeFileSync(
    join(tmp, 'package.json'),
    JSON.stringify({
      name: 'scanoficus-pack-smoke',
      private: true,
      type: 'module',
      dependencies: {
        ...tarballs,
        react: reactDev.react,
        '@types/react': reactDev['@types/react'],
        vite: JSON.parse(readFileSync(join(root, 'apps/demo/package.json'), 'utf8')).devDependencies
          .vite,
      },
    }),
  );
  // Packed workspace dependencies point at version 0.0.0 (unpublished): use the tarballs.
  writeFileSync(
    join(tmp, 'pnpm-workspace.yaml'),
    `overrides:\n${Object.entries(tarballs)
      .map(([name, spec]) => `  '${name}': '${spec}'`)
      .join('\n')}\n`,
  );
  run('pnpm', ['install', '--prefer-offline'], tmp);

  const core = JSON.parse(
    readFileSync(join(tmp, 'node_modules/@scanoficus/core/package.json'), 'utf8'),
  );
  const specifiers = [
    ...Object.keys(core.exports)
      .filter((k) => k !== './package.json' && k !== './worker-entry')
      .map((k) => `@scanoficus/core${k.slice(1)}`),
    '@scanoficus/elements',
    '@scanoficus/elements/define',
    '@scanoficus/react',
    '@scanoficus/detector-nn',
  ];
  // Runtime: every entry imports in Node (SSR safety) and exposes exports.
  writeFileSync(
    join(tmp, 'runtime.mjs'),
    `${specifiers.map((s, i) => `import * as m${i} from '${s}';`).join('\n')}
const modules = [${specifiers.map((_, i) => `m${i}`).join(', ')}];
const empty = ${JSON.stringify(specifiers)}.filter((s, i) => s !== '@scanoficus/elements/define' && Object.keys(modules[i]).length === 0);
if (empty.length) throw new Error('No exports: ' + empty.join(', '));
console.log('runtime ok:', ${specifiers.length}, 'entries');`,
  );
  console.log(run('node', ['runtime.mjs'], tmp).trim());
  const workerEntry = join(
    tmp,
    'node_modules/@scanoficus/core',
    core.exports['./worker-entry'].import,
  );
  if (!existsSync(workerEntry)) throw new Error(`worker entry missing: ${workerEntry}`);
  console.log('worker entry shipped:', core.exports['./worker-entry'].import);

  // Types: all entries resolve with both module resolution modes.
  writeFileSync(
    join(tmp, 'types.ts'),
    `${specifiers.map((s, i) => `import * as t${i} from '${s}';`).join('\n')}
import type { ScannerController } from '@scanoficus/core/controller';
import type { ScanDocumentElement } from '@scanoficus/elements';
import type { DocumentScannerProps } from '@scanoficus/react';
export type Check = [ScannerController, ScanDocumentElement, DocumentScannerProps];
export const all = [${specifiers.map((_, i) => `t${i}`).join(', ')}];
`,
  );
  const tsc = join(root, 'node_modules/.bin/tsc');
  for (const [module, resolution] of [
    ['ESNext', 'Bundler'],
    ['NodeNext', 'NodeNext'],
  ]) {
    run(
      tsc,
      [
        '--noEmit',
        '--strict',
        '--skipLibCheck',
        'false',
        '--lib',
        'ES2023,DOM,DOM.Iterable',
        '--types',
        '',
        '--module',
        module,
        '--moduleResolution',
        resolution,
        'types.ts',
      ],
      tmp,
    );
    console.log(`types ok: ${resolution}`);
  }

  // Installed-package Vite app: the default worker URL must work in dev and build.
  const app = join(tmp, 'vite-app');
  run('mkdir', ['-p', app], tmp);
  writeFileSync(
    join(app, 'index.html'),
    '<!doctype html><script type="module" src="./main.js"></script>',
  );
  writeFileSync(
    join(app, 'main.js'),
    `import { createDetectorWorker } from '@scanoficus/core/worker';
const canvas = new OffscreenCanvas(384, 288);
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#222';
ctx.fillRect(0, 0, 384, 288);
ctx.fillStyle = '#eee';
ctx.beginPath();
ctx.moveTo(90, 40); ctx.lineTo(300, 60); ctx.lineTo(280, 250); ctx.lineTo(70, 230);
ctx.closePath();
ctx.fill();
const detector = createDetectorWorker();
detector.detect(ctx.getImageData(0, 0, 384, 288)).then(
  (detection) => { window.result = { detection }; },
  (error) => { window.result = { error: String(error) }; },
);
`,
  );
  const vite = await import(pathToFileURL(join(tmp, 'node_modules/vite/dist/node/index.js')).href);
  const browser = await chromium.launch({
    ...(process.env.CHROME_PATH
      ? { executablePath: process.env.CHROME_PATH }
      : { channel: 'chrome' }),
    headless: true,
  });
  const probe = async (label, url) => {
    const page = await browser.newPage();
    await page.goto(url);
    await page.waitForFunction(() => window.result, null, { timeout: 20000 });
    const result = await page.evaluate(() => window.result);
    await page.close();
    if (!result.detection) throw new Error(`${label}: worker detection failed: ${result.error}`);
    console.log(`vite ${label} (installed package): default worker detects a document`);
  };
  try {
    const dev = await vite.createServer({ root: app, logLevel: 'error', server: { port: 0 } });
    await dev.listen();
    await probe('dev', dev.resolvedUrls.local[0]);
    await dev.close();
    await vite.build({ root: app, logLevel: 'error' });
    const preview = await vite.preview({ root: app, logLevel: 'error', preview: { port: 0 } });
    await probe('build', preview.resolvedUrls.local[0]);
    await new Promise((resolve) => preview.httpServer.close(resolve));
  } finally {
    await browser.close();
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
