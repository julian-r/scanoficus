/**
 * Downloads and unpacks the public detector benchmark datasets outside the repository.
 *
 * Usage: `pnpm bench:fetch [--datasets smartdoc,midv500,midv2019,negatives]
 *   [--midv-types 01,06,...] [--keep-archives]`
 *
 * Datasets go to `$SCANOFICUS_DATASETS` (default `~/datasets/scanoficus`), never into the repo.
 * Downloads resume (`curl -C -`) and are verified against the published checksums (SmartDoc,
 * MIDV); finished steps are skipped on re-runs. Requires `curl`, `tar`, `unzip` and
 * `shasum`/`md5`.
 *
 * - SmartDoc 2015 Challenge 1 (CC BY 4.0): `frames.tar.gz` (1.0 GB) from the GitHub release of
 *   jchazalon/smartdoc15-ch1-dataset.
 * - MIDV-500 and MIDV-2019 (CC BY-SA 2.5): one ~0.65–0.8 GB zip per document type from the
 *   Smart Engines FTP server. Only `images/` and `ground_truth/` are extracted (videos are
 *   skipped). By default a fixed subset of 10 of the 50 types is fetched (every 5th type).
 * - No-document negatives: the COCO 2017 panoptic and caption annotations (0.86 + 0.25 GB zips,
 *   CC BY 4.0) are downloaded once to select 800 images (see `negatives.ts`), which are then
 *   fetched from images.cocodataset.org (≈ 130 MB). The selection is written to
 *   `manifest.json`; delete it to select again. COCO publishes no checksums and its host serves
 *   no valid HTTPS certificate, so downloads use plain HTTP. Instead, the SHA-256 of the
 *   resulting set (file names and contents, {@link NEGATIVES_DIGEST}) is checked after the
 *   download, which also pins the selection made from the annotations.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { DEFAULT_MIDV_TYPES, datasetsRoot, MIDV_TYPES } from './datasets.ts';
import { decodeFrames } from './frames.ts';
import {
  hasUniformBorder,
  type NegativesManifest,
  negativesDir,
  selectNegatives,
} from './negatives.ts';

const SMARTDOC_RELEASE =
  'https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0';
const MIDV_FTP = {
  midv500: 'ftp://smartengines.com/midv-500',
  midv2019: 'ftp://smartengines.com/midv-500/extra/midv-2019',
} as const;

const { values } = parseArgs({
  options: {
    datasets: { type: 'string', default: 'smartdoc,midv500,midv2019,negatives' },
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

const COCO_ANNOTATIONS = 'http://images.cocodataset.org/annotations';
/**
 * SHA-256 over the lines `<file> <sha256 of the file>\n` of all 800 selected images, sorted by
 * file name (the set used since round 1, 2026-11-11).
 */
const NEGATIVES_DIGEST = '4eb3369ed2d80e2cc322b3fefd572e2345926481997712fea9fa7a22342bd38b';

async function fetchNegatives(): Promise<void> {
  const dir = negativesDir(root);
  const images = join(dir, 'images');
  mkdirSync(images, { recursive: true });
  const manifestFile = join(dir, 'manifest.json');
  if (!existsSync(manifestFile)) {
    const panoptic = join(dir, 'panoptic_annotations_trainval2017.zip');
    const captions = join(dir, 'annotations_trainval2017.zip');
    download(`${COCO_ANNOTATIONS}/panoptic_annotations_trainval2017.zip`, panoptic);
    download(`${COCO_ANNOTATIONS}/annotations_trainval2017.zip`, captions);
    const read = (zip: string, name: string) =>
      JSON.parse(
        execFileSync('unzip', ['-p', zip, `annotations/${name}`], {
          maxBuffer: 1 << 30,
        }).toString('utf8'),
      );
    console.log('negatives: selecting images');
    const manifest = selectNegatives(
      read(panoptic, 'panoptic_val2017.json'),
      read(captions, 'captions_val2017.json'),
      read(panoptic, 'panoptic_train2017.json'),
      read(captions, 'captions_train2017.json'),
    );
    writeFileSync(manifestFile, `${JSON.stringify(manifest, null, 1)}\n`);
    if (!values['keep-archives']) {
      rmSync(panoptic);
      rmSync(captions);
    }
  }
  const manifest = JSON.parse(readFileSync(manifestFile, 'utf8')) as NegativesManifest;
  const selected = new Set(manifest.images.map((image) => image.file));
  for (const file of readdirSync(images)) {
    if (!selected.has(file)) rmSync(join(images, file));
  }
  const missing = manifest.images.filter((image) => !existsSync(join(images, image.file)));
  console.log(`negatives: ${missing.length} of ${manifest.images.length} images to download`);
  // One curl process per batch reuses the connection.
  for (let start = 0; start < missing.length; start += 50) {
    const batch = missing.slice(start, start + 50);
    const args = ['-fsSL', '--retry', '5'];
    for (const image of batch) args.push('-o', join(images, `${image.file}.part`), image.url);
    execFileSync('curl', args, { stdio: ['ignore', 'inherit', 'inherit'] });
    for (const image of batch) {
      renameSync(join(images, `${image.file}.part`), join(images, image.file));
    }
    console.log(`negatives: ${Math.min(start + 50, missing.length)} / ${missing.length}`);
  }
  const digest = createHash('sha256');
  for (const image of [...manifest.images].sort((a, b) => (a.file < b.file ? -1 : 1))) {
    const file = createHash('sha256').update(readFileSync(join(images, image.file)));
    digest.update(`${image.file} ${file.digest('hex')}\n`);
  }
  if (digest.digest('hex') !== NEGATIVES_DIGEST) {
    throw new Error(
      `negatives: the images differ from the benchmarked set (SHA-256 mismatch); delete ${dir} to fetch again`,
    );
  }
  if (manifest.images.some((image) => image.uniformBorder === undefined)) {
    for (const image of manifest.images) {
      await decodeFrames([join(images, image.file)], 'gray', image, (gray) => {
        image.uniformBorder = hasUniformBorder(gray);
      });
    }
    writeFileSync(manifestFile, `${JSON.stringify(manifest, null, 1)}\n`);
    const bordered = manifest.images.filter((image) => image.uniformBorder).length;
    console.log(`negatives: ${bordered} images with a uniform border are skipped`);
  }
}

console.log(`datasets root: ${root}`);
for (const dataset of datasets) {
  if (dataset === 'smartdoc') fetchSmartDoc();
  else if (dataset === 'midv500' || dataset === 'midv2019') fetchMidv(dataset);
  else if (dataset === 'negatives') await fetchNegatives();
  else throw new Error(`Unknown dataset: ${dataset}`);
}
