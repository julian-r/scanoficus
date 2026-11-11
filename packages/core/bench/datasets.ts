/**
 * Dataset adapters for the real-data detector benchmark. Each adapter lists the clips of a
 * dataset as {@link BenchClip}s: an ordered sequence of frame files with their ground-truth quad
 * (TL, TR, BR, BL in frame pixels) and the size of the document's reference frame used by the
 * SmartDoc Jaccard index. The `negatives` dataset has no document in any frame (quad `null`).
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import type { Quad } from '../src/geometry/index.ts';
import { loadNegatives } from './negatives.ts';

export type DatasetName = 'smartdoc' | 'midv500' | 'midv2019' | 'dizaj' | 'rwmd' | 'negatives';
export const DATASETS: readonly DatasetName[] = [
  'smartdoc',
  'midv500',
  'midv2019',
  'dizaj',
  'rwmd',
  'negatives',
];
/**
 * Kind of a dataset: `video` clips (consecutive frames of one document), `photo` sets (single
 * photos in a `gt.jsonl`, see {@link PHOTO_DATASETS}) or the no-document `negatives`.
 */
export type DatasetKind = 'video' | 'photo' | 'negatives';
export function datasetKind(dataset: DatasetName): DatasetKind {
  if (dataset === 'negatives') return 'negatives';
  return dataset in PHOTO_DATASETS ? 'photo' : 'video';
}
/**
 * Single-photo test sets in the `gt.jsonl` format (`$SCANOFICUS_DATASETS/FORMAT.md`), keyed by
 * dataset name: the directory below the datasets root. Frozen, evaluation only.
 */
export const PHOTO_DATASETS: Partial<Record<DatasetName, string>> = {
  dizaj: 'dizaj-mendeley-v3',
  rwmd: 'rwmd',
};
/** Images per negatives clip (images of one pixel size are grouped into clips). */
const NEGATIVES_PER_CLIP = 100;
/** Photos per photo-set clip (photos of one pixel size, in `gt.jsonl` order). */
const PHOTOS_PER_CLIP = 50;

/** One frame with its ground truth. */
export interface BenchFrame {
  /** 1-based frame number within the clip (the COCO image id for the negatives). */
  index: number;
  /** Absolute path of the frame image. */
  path: string;
  /**
   * Ground-truth document quad (TL, TR, BR, BL), corners may lie outside the frame; `null` for
   * the no-document negatives.
   */
  quad: Quad | null;
  /** Photo sets: the image id in `gt.jsonl`. */
  id?: string;
  /** Photo sets: the `gt.jsonl` flags (e.g. `partial`, `multi-document`). */
  flags?: string[];
  /** Photo sets: the image category, if the set has one (RWMD). */
  category?: string;
}

/** A video clip (consecutive frames of one document on one background / condition). */
export interface BenchClip {
  dataset: DatasetName;
  /** Unique within the dataset, safe as a file name (e.g. `background01-datasheet001`). */
  id: string;
  /** Background (SmartDoc), capture condition (MIDV) or image source (negatives). */
  background: string;
  /** Document type (SmartDoc), ID document type (MIDV) or `none` (negatives). */
  docType: string;
  /** Size of the document's reference frame (SmartDoc: model size in 0.1 mm). */
  reference: { width: number; height: number };
  frames: BenchFrame[];
}

/** All 50 MIDV-500 / MIDV-2019 document types (archive names). */
export const MIDV_TYPES: readonly string[] = [
  '01_alb_id',
  '02_aut_drvlic_new',
  '03_aut_id_old',
  '04_aut_id',
  '05_aze_passport',
  '06_bra_passport',
  '07_chl_id',
  '08_chn_homereturn',
  '09_chn_id',
  '10_cze_id',
  '11_cze_passport',
  '12_deu_drvlic_new',
  '13_deu_drvlic_old',
  '14_deu_id_new',
  '15_deu_id_old',
  '16_deu_passport_new',
  '17_deu_passport_old',
  '18_dza_passport',
  '19_esp_drvlic',
  '20_esp_id_new',
  '21_esp_id_old',
  '22_est_id',
  '23_fin_drvlic',
  '24_fin_id',
  '25_grc_passport',
  '26_hrv_drvlic',
  '27_hrv_passport',
  '28_hun_passport',
  '29_irn_drvlic',
  '30_ita_drvlic',
  '31_jpn_drvlic',
  '32_lva_passport',
  '33_mac_id',
  '34_mda_passport',
  '35_nor_drvlic',
  '36_pol_drvlic',
  '37_prt_id',
  '38_rou_drvlic',
  '39_rus_internalpassport',
  '40_srb_id',
  '41_srb_passport',
  '42_svk_id',
  '43_tur_id',
  '44_ukr_id',
  '45_ukr_passport',
  '46_ury_passport',
  '47_usa_bordercrossing',
  '48_usa_passportcard',
  '49_usa_ssn82',
  '50_xpo_id',
];

/** Default MIDV subset: every 5th document type (01, 06, …, 46), 10 of 50. */
export const DEFAULT_MIDV_TYPES: readonly string[] = MIDV_TYPES.filter((_, i) => i % 5 === 0).map(
  (t) => t.slice(0, 2),
);

/** MIDV capture conditions, keyed by the first letter of the clip name. */
const MIDV_CONDITIONS: Record<string, string> = {
  T: 'table',
  K: 'keyboard',
  H: 'hand',
  P: 'partial',
  C: 'clutter',
  D: 'distorted',
  L: 'low-light',
};

/** Root directory of the downloaded datasets (`$SCANOFICUS_DATASETS` or `~/datasets/scanoficus`). */
export function datasetsRoot(): string {
  return process.env.SCANOFICUS_DATASETS ?? join(homedir(), 'datasets', 'scanoficus');
}

export function loadClips(dataset: DatasetName, root: string = datasetsRoot()): BenchClip[] {
  if (dataset === 'smartdoc') return loadSmartDoc(join(root, 'smartdoc15-ch1'));
  if (dataset === 'negatives') return loadNegativeClips(root);
  const photoDir = PHOTO_DATASETS[dataset];
  if (photoDir) return loadPhotoClips(dataset, join(root, photoDir));
  return loadMidv(dataset as 'midv500' | 'midv2019', root);
}

/**
 * Single-photo sets (`gt.jsonl`, see `$SCANOFICUS_DATASETS/FORMAT.md`): photos of the same size
 * (after EXIF orientation) are grouped into clips of at most {@link PHOTOS_PER_CLIP} in file
 * order, with ids such as `3000x4000-01`. The frame index is the 1-based line number in
 * `gt.jsonl`. The quad keeps the file's corner order (clockwise from the corner with the smallest
 * x + y). Photos with a `null` quad are excluded from accuracy (the format allows them; neither
 * current set has any). There is no document template, so the reference frame of the JI is
 * derived per photo from its ground truth (see `photoReference` in `report-data.ts`).
 */
function loadPhotoClips(dataset: DatasetName, dir: string): BenchClip[] {
  const file = join(dir, 'gt.jsonl');
  if (!existsSync(file)) throw new Error(`${dataset} not found in ${dir} (no gt.jsonl)`);
  const lines = readFileSync(file, 'utf8').split('\n');
  const bySize = new Map<string, BenchFrame[]>();
  lines.forEach((line, i) => {
    if (!line.trim()) return;
    const row = JSON.parse(line) as {
      id: string;
      image: string;
      width: number;
      height: number;
      quad: [number, number][] | null;
      flags?: string[];
      category?: string;
    };
    if (!row.quad) return;
    const key = `${row.width}x${row.height}`;
    const [a, b, c, d] = row.quad.map(([x, y]) => ({ x, y }));
    const frame: BenchFrame = {
      index: i + 1,
      path: join(dir, row.image),
      quad: [a, b, c, d],
      id: row.id,
      flags: row.flags ?? [],
    };
    if (row.category) frame.category = row.category;
    bySize.set(key, [...(bySize.get(key) ?? []), frame]);
  });
  const clips: BenchClip[] = [];
  for (const [size, frames] of [...bySize].sort(([a], [b]) => a.localeCompare(b))) {
    const [width, height] = size.split('x').map(Number);
    for (let start = 0; start < frames.length; start += PHOTOS_PER_CLIP) {
      clips.push({
        dataset,
        id: `${size}-${String(start / PHOTOS_PER_CLIP + 1).padStart(2, '0')}`,
        background: 'photo',
        docType: 'mixed',
        reference: { width, height },
        frames: frames.slice(start, start + PHOTOS_PER_CLIP),
      });
    }
  }
  return clips;
}

/**
 * No-document negatives (see `negatives.ts`): images of the same pixel size are grouped into
 * clips of at most {@link NEGATIVES_PER_CLIP}, ordered by COCO image id, with ids such as
 * `coco-640x480-01`. The reference frame is the image size (unused).
 */
function loadNegativeClips(root: string): BenchClip[] {
  const { dir, images } = loadNegatives(root);
  const bySize = new Map<string, typeof images>();
  for (const image of [...images].sort((a, b) => a.id - b.id)) {
    const key = `${image.width}x${image.height}`;
    bySize.set(key, [...(bySize.get(key) ?? []), image]);
  }
  const clips: BenchClip[] = [];
  for (const [size, group] of [...bySize].sort(([a], [b]) => a.localeCompare(b))) {
    for (let start = 0; start < group.length; start += NEGATIVES_PER_CLIP) {
      const chunk = group.slice(start, start + NEGATIVES_PER_CLIP);
      clips.push({
        dataset: 'negatives',
        id: `coco-${size}-${String(start / NEGATIVES_PER_CLIP + 1).padStart(2, '0')}`,
        background: 'coco',
        docType: 'none',
        reference: { width: chunk[0].width, height: chunk[0].height },
        frames: chunk.map((image) => ({
          index: image.id,
          path: join(dir, 'images', image.file),
          quad: null,
        })),
      });
    }
  }
  return clips;
}

/**
 * SmartDoc 2015 Challenge 1 (frames release v2.0.0): `metadata.csv.gz` lists every frame with
 * its model size and the GT corners in the order TL, BL, BR, TR.
 */
function loadSmartDoc(dir: string): BenchClip[] {
  const file = join(dir, 'metadata.csv.gz');
  if (!existsSync(file)) throw new Error(`SmartDoc not found in ${dir}; run pnpm bench:fetch`);
  const lines = gunzipSync(readFileSync(file)).toString('utf8').trim().split('\n');
  const header = lines[0].split(',');
  const col = (name: string) => {
    const i = header.indexOf(name);
    if (i < 0) throw new Error(`metadata.csv: missing column ${name}`);
    return i;
  };
  const c = {
    bg: col('bg_name'),
    model: col('model_name'),
    type: col('modeltype_name'),
    path: col('image_path'),
    index: col('frame_index'),
    width: col('model_width'),
    height: col('model_height'),
    tlx: col('tl_x'),
    tly: col('tl_y'),
    trx: col('tr_x'),
    try: col('tr_y'),
    brx: col('br_x'),
    bry: col('br_y'),
    blx: col('bl_x'),
    bly: col('bl_y'),
  };
  const clips = new Map<string, BenchClip>();
  for (const line of lines.slice(1)) {
    const f = line.split(',');
    const id = `${f[c.bg]}-${f[c.model]}`;
    let clip = clips.get(id);
    if (!clip) {
      clip = {
        dataset: 'smartdoc',
        id,
        background: f[c.bg],
        docType: f[c.type],
        reference: { width: Number(f[c.width]), height: Number(f[c.height]) },
        frames: [],
      };
      clips.set(id, clip);
    }
    const p = (x: number, y: number) => ({ x: Number(f[x]), y: Number(f[y]) });
    clip.frames.push({
      index: Number(f[c.index]),
      path: join(dir, f[c.path]),
      quad: [p(c.tlx, c.tly), p(c.trx, c.try), p(c.brx, c.bry), p(c.blx, c.bly)],
    });
  }
  const result = [...clips.values()];
  for (const clip of result) clip.frames.sort((a, b) => a.index - b.index);
  return result.sort((a, b) => a.id.localeCompare(b.id));
}

/**
 * MIDV-500 / MIDV-2019: `<type>/ground_truth/<clip>/<clip>_NN.json` holds `{ quad }` (TL, TR,
 * BR, BL) per frame. The pixel size of the printed template `<type>/images/<type>.tif` is the
 * document's reference frame. Only types unpacked completely by `pnpm bench:fetch` are listed.
 */
function loadMidv(dataset: 'midv500' | 'midv2019', root: string): BenchClip[] {
  const dir = join(root, dataset === 'midv500' ? 'midv-500' : 'midv-2019');
  if (!existsSync(dir)) throw new Error(`${dataset} not found in ${dir}; run pnpm bench:fetch`);
  const clips: BenchClip[] = [];
  for (const type of MIDV_TYPES) {
    const gtDir = join(dir, type, 'ground_truth');
    if (!existsSync(join(dir, type, '.complete'))) continue;
    const reference = tiffSize(join(dir, type, 'images', `${type}.tif`));
    for (const clipName of readdirSync(gtDir).sort()) {
      if (clipName.endsWith('.json')) continue;
      const frames: BenchFrame[] = [];
      for (const file of readdirSync(join(gtDir, clipName)).sort()) {
        const match = file.match(/_(\d+)\.json$/);
        if (!match) continue;
        const gt = JSON.parse(readFileSync(join(gtDir, clipName, file), 'utf8')) as {
          quad: [number, number][];
        };
        const [tl, tr, br, bl] = gt.quad.map(([x, y]) => ({ x, y }));
        frames.push({
          index: Number(match[1]),
          path: join(dir, type, 'images', clipName, file.replace(/\.json$/, '.tif')),
          quad: [tl, tr, br, bl],
        });
      }
      clips.push({
        dataset,
        id: `${type}-${clipName}`,
        background: MIDV_CONDITIONS[clipName[0]] ?? clipName[0],
        docType: type,
        reference,
        frames,
      });
    }
  }
  return clips;
}

/** Width and height from a baseline TIFF header (tags 256/257). */
function tiffSize(file: string): { width: number; height: number } {
  const buf = readFileSync(file);
  const little = buf.toString('latin1', 0, 2) === 'II';
  const u16 = (o: number) => (little ? buf.readUInt16LE(o) : buf.readUInt16BE(o));
  const u32 = (o: number) => (little ? buf.readUInt32LE(o) : buf.readUInt32BE(o));
  const ifd = u32(4);
  const count = u16(ifd);
  let width = 0;
  let height = 0;
  for (let i = 0; i < count; i++) {
    const entry = ifd + 2 + 12 * i;
    const tag = u16(entry);
    const type = u16(entry + 2);
    const value = type === 3 ? u16(entry + 8) : u32(entry + 8);
    if (tag === 256) width = value;
    if (tag === 257) height = value;
  }
  if (!width || !height) throw new Error(`${file}: cannot read TIFF size`);
  return { width, height };
}
