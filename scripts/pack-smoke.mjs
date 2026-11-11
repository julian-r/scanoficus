/**
 * Packs the libraries (`pnpm pack`), installs the tarballs into a fresh temporary project and
 * checks that every runtime export resolves and imports in Node, the worker entry is shipped,
 * and all type declarations resolve for both `Bundler` and `NodeNext` module resolution.
 * Run after `pnpm build`: `pnpm check:pack`.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

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
  for (const name of ['core', 'elements', 'react']) {
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
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
