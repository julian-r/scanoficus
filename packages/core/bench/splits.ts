/**
 * Tuning / held-out split of the real-data benchmark, so that detector changes are not fitted to
 * the frames they are judged on.
 *
 * - SmartDoc 2015 Challenge 1 is split by document type. Each of the 6 types has 5 documents,
 *   and every document was filmed on all 5 backgrounds. The tuning split holds the 1st, 3rd and
 *   5th type in alphabetical order (datasheet, magazine, patent: 15 documents, 75 clips), the
 *   held-out split the others (letter, paper, tax: 15 documents, 75 clips). No document appears
 *   in both, and both contain every background.
 * - MIDV-500 and MIDV-2019 are held out entirely (other documents, devices and conditions).
 * - The no-document negatives are a separate check of the false-positive rate. They are used to
 *   calibrate score thresholds, never to rank candidates on positives.
 *
 * Detector changes are developed and tuned on the tuning split only (including failure montages);
 * held-out numbers are reported, not optimized.
 */
import type { DatasetName } from './datasets.ts';

export type Split = 'tune' | 'heldout';
export const SPLITS: readonly Split[] = ['tune', 'heldout'];

/** SmartDoc document types (`modeltype_name`) in the tuning split. */
export const SMARTDOC_TUNING_TYPES: readonly string[] = ['datasheet', 'magazine', 'patent'];

/** Split of a clip, or `null` for datasets outside the split (the negatives). */
export function splitOf(clip: { dataset: DatasetName; docType: string }): Split | null {
  if (clip.dataset === 'negatives') return null;
  if (clip.dataset !== 'smartdoc') return 'heldout';
  return SMARTDOC_TUNING_TYPES.includes(clip.docType) ? 'tune' : 'heldout';
}
