/**
 * Tuning / held-out split of the real-data benchmark, so that detector changes are not fitted to
 * the frames they are judged on.
 *
 * - SmartDoc 2015 Challenge 1 is split by document type. Each of the 6 types has 5 documents,
 *   and every document was filmed on all 5 backgrounds. The tuning split holds the 1st, 3rd and
 *   5th type in alphabetical order (datasheet, magazine, patent: 15 documents, 75 clips), the
 *   held-out split the others (letter, paper, tax: 15 documents, 75 clips). No document appears
 *   in both, and both contain every background.
 * - MIDV-500 (the 10 benchmarked types 01, 06, …, 46) is split by document type with the same
 *   rule: the 1st, 3rd, 5th, 7th and 9th type in numeric order (01 alb_id, 11 cze_passport,
 *   21 esp_id_old, 31 jpn_drvlic, 41 srb_passport) are tuning, the others (06 bra_passport,
 *   16 deu_passport_new, 26 hrv_drvlic, 36 pol_drvlic, 46 ury_passport) are held out. Every type
 *   has all 5 conditions × 2 phones, so both halves hold every condition. This split was added
 *   in round 2; aggregate numbers of all 10 types (including the held-out half) were looked at in
 *   round 1, so the MIDV-500 held-out half is not fully clean (see docs/benchmarks.md).
 * - MIDV-2019 is held out entirely. It films the same document types as MIDV-500 (other
 *   captures, devices and conditions), so the report also gives MIDV-2019 restricted to the
 *   MIDV-500 held-out types.
 * - The single-photo sets (Dizaj, RWMD) are frozen test sets: evaluation only, never tuned on.
 * - The no-document negatives are a separate check of the false-positive rate. They are split
 *   by COCO image id: even ids are the calibration half (score thresholds are set on it), odd
 *   ids the report half (false-positive rates are reported on it). Neither half is used to rank
 *   candidates on positives. Before round 2, all negatives were used for both.
 *
 * Detector changes are developed and tuned on the tuning split only (including failure montages);
 * held-out numbers are reported, not optimized.
 */
import type { DatasetName } from './datasets.ts';

export type Split = 'tune' | 'heldout';
export const SPLITS: readonly Split[] = ['tune', 'heldout'];

/** SmartDoc document types (`modeltype_name`) in the tuning split. */
export const SMARTDOC_TUNING_TYPES: readonly string[] = ['datasheet', 'magazine', 'patent'];

/** MIDV-500 document types (number prefix of the archive name) in the tuning split. */
export const MIDV500_TUNING_TYPES: readonly string[] = ['01', '11', '21', '31', '41'];

/** True if a MIDV document type (archive name such as `01_alb_id`) is a MIDV-500 tuning type. */
export function isMidvTuningType(docType: string): boolean {
  return MIDV500_TUNING_TYPES.includes(docType.slice(0, 2));
}

/** Split of a clip, or `null` for datasets outside the split (the negatives). */
export function splitOf(clip: { dataset: DatasetName; docType: string }): Split | null {
  switch (clip.dataset) {
    case 'negatives':
      return null;
    case 'smartdoc':
      return SMARTDOC_TUNING_TYPES.includes(clip.docType) ? 'tune' : 'heldout';
    case 'midv500':
      return isMidvTuningType(clip.docType) ? 'tune' : 'heldout';
    default:
      return 'heldout';
  }
}

export type NegativeSplit = 'calibration' | 'report';

/** Half of the no-document negatives a COCO image id belongs to (see the module docs). */
export function negativeSplitOf(imageId: number): NegativeSplit {
  return imageId % 2 === 0 ? 'calibration' : 'report';
}
