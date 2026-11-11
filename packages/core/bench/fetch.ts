/**
 * Downloads and unpacks the public detector benchmark datasets outside the repository.
 *
 * Usage: `pnpm bench:fetch [--datasets smartdoc,midv500,midv2019] [--midv-types 01,06,...]
 *   [--keep-archives]`
 *
 * Datasets go to `$SCANOFICUS_DATASETS` (default `~/datasets/scanoficus`), never into the repo.
 * Downloads resume (`curl -C -`) and are verified against the published checksums; finished
 * steps are skipped on re-runs. Requires `curl`, `tar`, `unzip` and `shasum`/`md5`.
 *
 * - SmartDoc 2015 Challenge 1 (CC BY 4.0): `frames.tar.gz` (1.0 GB) from the GitHub release of
 *   jchazalon/smartdoc15-ch1-dataset.
 * - MIDV-500 and MIDV-2019 (CC BY-SA 2.5): one ~0.65–0.8 GB zip per document type from the
 *   Smart Engines FTP server. Only `images/` and `ground_truth/` are extracted (videos are
 *   skipped). By default a fixed subset of 10 of the 50 types is fetched (every 5th type).
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { DEFAULT_MIDV_TYPES, datasetsRoot, MIDV_TYPES } from './datasets.ts';

const SMARTDOC_RELEASE =
  'https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0';
const MIDV_FTP = {
  midv500: 'ftp://smartengines.com/midv-500',
  midv2019: 'ftp://smartengines.com/midv-500/extra/midv-2019',
} as const;

const { values } = parseArgs({
  options: {
    datasets: { type: 'string', default: 'smartdoc,midv500,midv2019' },
    'midv-types': { type: 'string', default: DEFAULT_MIDV_TYPES.join(',') },
    'keep-archives': { type: 'boolean', default: false },
  },
});
const root = datasetsRoot();
const datasets = values.datasets.split(',');
const midvTypes = values['midv-types'].split(',').map((prefix) => {
  const name = MIDV_TYPES.find((t) => t.startsWith(`${prefix}_`) || t === prefix);
  if (!name) throw new Error(`Unknown MIDV document type: ${prefix}`);
  return name;
});

function run(command: string, args: string[], cwd?: string): string {
  return execFileSync(command, args, { cwd, encoding: 'utf8', maxBuffer: 1 << 26 });
}

function download(url: string, file: string): void {
  console.log(`download ${url}`);
  execFileSync('curl', ['-fSL', '-#', '--retry', '5', '-C', '-', '-o', file, url], {
    stdio: ['ignore', 'inherit', 'inherit'],
  });
}

function checksum(file: string, algorithm: 'sha256' | 'md5'): string {
  const out =
    algorithm === 'sha256' ? run('shasum', ['-a', '256', file]) : run('openssl', ['md5', file]);
  const match = out.match(/[0-9a-f]{32,64}/);
  if (!match) throw new Error(`Cannot parse checksum output: ${out}`);
  return match[0];
}

function expectedChecksum(list: string, name: string): string {
  for (const line of list.split('\n')) {
    const [sum, file] = line.trim().split(/\s+/);
    if (file === name) return sum;
  }
  throw new Error(`No checksum listed for ${name}`);
}

function fetchSmartDoc(): void {
  const dir = join(root, 'smartdoc15-ch1');
  mkdirSync(dir, { recursive: true });
  if (existsSync(join(dir, 'metadata.csv.gz'))) {
    console.log(`smartdoc: already unpacked in ${dir}`);
    return;
  }
  const archive = join(dir, 'frames.tar.gz');
  download(`${SMARTDOC_RELEASE}/sha256.chksum`, join(dir, 'sha256.chksum'));
  download(`${SMARTDOC_RELEASE}/frames.tar.gz`, archive);
  const expected = expectedChecksum(
    readFileSync(join(dir, 'sha256.chksum'), 'utf8'),
    'frames.tar.gz',
  );
  if (checksum(archive, 'sha256') !== expected) throw new Error(`${archive}: checksum mismatch`);
  console.log('smartdoc: unpacking');
  run('tar', ['xzf', archive], dir);
  if (!values['keep-archives']) rmSync(archive);
}

function fetchMidv(dataset: keyof typeof MIDV_FTP): void {
  const dir = join(root, dataset === 'midv500' ? 'midv-500' : 'midv-2019');
  const zips = join(dir, 'zips');
  mkdirSync(zips, { recursive: true });
  const base = MIDV_FTP[dataset];
  for (const name of ['license.txt', 'readme.txt', 'md5.txt']) {
    if (!existsSync(join(dir, name))) download(`${base}/${name}`, join(dir, name));
  }
  const md5List = readFileSync(join(dir, 'md5.txt'), 'utf8');
  for (const type of midvTypes) {
    const done = join(dir, type, '.complete');
    if (existsSync(done)) continue;
    const zip = join(zips, `${type}.zip`);
    download(`${base}/dataset/${type}.zip`, zip);
    if (checksum(zip, 'md5') !== expectedChecksum(md5List, `${type}.zip`)) {
      throw new Error(`${zip}: checksum mismatch (delete it to re-download)`);
    }
    console.log(`${dataset}: unpacking ${type} (without videos)`);
    // MIDV-500 archives contain a `<type>/` folder, MIDV-2019 archives do not.
    const prefixed = run('unzip', ['-Z1', zip])
      .split('\n')
      .some((entry) => entry.startsWith(`${type}/`));
    run('unzip', ['-q', '-o', zip, '-x', '*videos/*', '-d', prefixed ? dir : join(dir, type)]);
    writeFileSync(done, '');
    if (!values['keep-archives']) rmSync(zip);
  }
}

console.log(`datasets root: ${root}`);
for (const dataset of datasets) {
  if (dataset === 'smartdoc') fetchSmartDoc();
  else if (dataset === 'midv500' || dataset === 'midv2019') fetchMidv(dataset);
  else throw new Error(`Unknown dataset: ${dataset}`);
}
