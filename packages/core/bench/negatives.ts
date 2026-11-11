/**
 * No-document negatives for the false-positive rate: COCO 2017 photos of desks, tables, screens,
 * keyboards and phones without any paper, book or other printed matter.
 *
 * Source: the COCO 2017 panoptic annotations, which label every pixel including the stuff class
 * `paper-merged`, and the COCO 2017 captions (five human captions per image). Both are CC BY 4.0
 * (cocodataset.org). Selection (deterministic, see {@link selectNegatives}): train2017 and
 * val2017 images that
 *
 * - contain at least one of `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`,
 *   `dining table` or `table-merged`,
 * - contain none of `book`, `paper-merged`, `cardboard` or `banner`,
 * - have no caption that mentions paper, documents, books, letters, cards, menus, signs, boards,
 *   text etc. ({@link DOCUMENT_WORDS}; the annotations miss small or partly hidden papers, the
 *   captions catch many of them),
 * - have a Flickr license that permits commercial use (CC BY 2.0, CC BY-SA 2.0, CC BY-ND 2.0,
 *   "no known copyright restrictions", US government work).
 *
 * All 2017 val images that qualify are taken, then evenly spaced train images (by image id) up
 * to {@link NEGATIVE_COUNT}. After the download, images with a uniform synthetic border (a frame
 * added in photo editing, see {@link hasUniformBorder}) are marked and skipped, as a camera frame
 * never has one. The images are only used locally for evaluation and are never committed.
 * Neither filter is perfect: a few images may still show a sheet of paper or an edited border.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { GrayImage } from '../src/image.ts';

/** Number of selected images. */
export const NEGATIVE_COUNT = 800;

/** Categories of which an image must contain at least one. */
const INCLUDE = [
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'dining table',
  'table-merged',
];
/** Categories an image must not contain (documents and document-like printed matter). */
const EXCLUDE = ['book', 'paper-merged', 'cardboard', 'banner'];
/** COCO license ids that permit commercial use (CC BY, BY-SA, BY-ND, no known restrictions, US gov). */
const LICENSES = [4, 5, 6, 7, 8];
/** Caption words (whole words, case-insensitive) that suggest printed matter in the image. */
export const DOCUMENT_WORDS =
  /\b(papers?|paperwork|documents?|books?|booklets?|notebooks?|notepads?|notes?|magazines?|newspapers?|news ?paper|letters?|envelopes?|mail|menus?|cards?|receipts?|pages?|sheets?|binders?|folders?|files?|forms?|flyers?|brochures?|pamphlets?|calendars?|posters?|maps?|signs?|tickets?|manuals?|textbooks?|comics?|novels?|homework|reading|reads|writing|writes|printer|printouts?|clipboards?|whiteboards?|boards?|charts?|diagrams?|text|words)\b/i;

export interface NegativeImage {
  /** COCO image id. */
  id: number;
  split: 'train2017' | 'val2017';
  file: string;
  url: string;
  width: number;
  height: number;
  /** COCO license id (see `licenses` in the manifest). */
  license: number;
  flickrUrl: string;
  /** Category names present in the image. */
  categories: string[];
  /** Set after the download: the image has a uniform synthetic border and is skipped. */
  uniformBorder?: boolean;
}

export interface NegativesManifest {
  source: string;
  licenses: { id: number; name: string; url: string }[];
  images: NegativeImage[];
}

export interface PanopticJson {
  licenses: { id: number; name: string; url: string }[];
  images: {
    id: number;
    file_name: string;
    coco_url: string;
    width: number;
    height: number;
    license: number;
    flickr_url: string;
  }[];
  annotations: { image_id: number; segments_info: { category_id: number }[] }[];
  categories: { id: number; name: string }[];
}

export interface CaptionsJson {
  annotations: { image_id: number; caption: string }[];
}

/** Images of one annotation split that pass the category, caption and license rules, by id. */
function candidates(
  json: PanopticJson,
  captions: CaptionsJson,
  split: NegativeImage['split'],
): NegativeImage[] {
  const names = new Map(json.categories.map((c) => [c.id, c.name]));
  const present = new Map(
    json.annotations.map((a) => [
      a.image_id,
      [...new Set(a.segments_info.map((s) => names.get(s.category_id) ?? ''))].sort(),
    ]),
  );
  const mentionsDocument = new Set(
    captions.annotations.filter((a) => DOCUMENT_WORDS.test(a.caption)).map((a) => a.image_id),
  );
  const out: NegativeImage[] = [];
  for (const image of json.images) {
    const categories = present.get(image.id) ?? [];
    if (!categories.some((c) => INCLUDE.includes(c))) continue;
    if (categories.some((c) => EXCLUDE.includes(c))) continue;
    if (mentionsDocument.has(image.id)) continue;
    if (!LICENSES.includes(image.license)) continue;
    out.push({
      id: image.id,
      split,
      file: image.file_name,
      url: image.coco_url,
      width: image.width,
      height: image.height,
      license: image.license,
      flickrUrl: image.flickr_url,
      categories,
    });
  }
  return out.sort((a, b) => a.id - b.id);
}

/** Deterministic selection from the val2017 and train2017 panoptic and caption annotations. */
export function selectNegatives(
  val: PanopticJson,
  valCaptions: CaptionsJson,
  train: PanopticJson,
  trainCaptions: CaptionsJson,
): NegativesManifest {
  const fromVal = candidates(val, valCaptions, 'val2017').slice(0, NEGATIVE_COUNT);
  const fromTrain = candidates(train, trainCaptions, 'train2017');
  const needed = Math.min(NEGATIVE_COUNT - fromVal.length, fromTrain.length);
  const picked = Array.from(
    { length: needed },
    (_, i) => fromTrain[Math.floor((i * fromTrain.length) / needed)],
  );
  return {
    source:
      'COCO 2017 panoptic and caption annotations (cocodataset.org, CC BY 4.0), images from Flickr',
    licenses: val.licenses,
    images: [...fromVal, ...picked],
  };
}

/** Directory of the negatives (`<root>/negatives/coco`). */
export function negativesDir(root: string): string {
  return join(root, 'negatives', 'coco');
}

/** Rim width (px) and maximum gray standard deviation of {@link hasUniformBorder}. */
const BORDER_RIM = 3;
const BORDER_MAX_STD = 3;

/**
 * True if each of the four outermost 3-px rims of `gray` is uniform (standard deviation ≤ 3
 * gray levels), as with frames, film borders or white margins added in photo editing.
 */
export function hasUniformBorder(gray: GrayImage): boolean {
  const { width, height, data } = gray;
  const uniform = (count: number, at: (k: number, d: number) => number) => {
    let sum = 0;
    let squares = 0;
    for (let d = 0; d < BORDER_RIM; d++) {
      for (let k = 0; k < count; k++) {
        const v = data[at(k, d)];
        sum += v;
        squares += v * v;
      }
    }
    const n = count * BORDER_RIM;
    const mean = sum / n;
    return squares / n - mean * mean <= BORDER_MAX_STD ** 2;
  };
  return (
    uniform(width, (k, d) => d * width + k) &&
    uniform(width, (k, d) => (height - 1 - d) * width + k) &&
    uniform(height, (k, d) => k * width + d) &&
    uniform(height, (k, d) => k * width + width - 1 - d)
  );
}

/** The selected images (see `manifest.json`) without the ones with a uniform border. */
export function loadNegatives(root: string): { dir: string; images: NegativeImage[] } {
  const dir = negativesDir(root);
  const manifest = join(dir, 'manifest.json');
  if (!existsSync(manifest)) throw new Error(`negatives not found in ${dir}; run pnpm bench:fetch`);
  const { images } = JSON.parse(readFileSync(manifest, 'utf8')) as NegativesManifest;
  if (images.some((image) => image.uniformBorder === undefined)) {
    throw new Error(`negatives in ${dir} are incomplete; run pnpm bench:fetch`);
  }
  return { dir, images: images.filter((image) => !image.uniformBorder) };
}
