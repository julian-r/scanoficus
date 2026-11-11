import { describe, expect, it } from 'vitest';
import { applyOrientation, jpegInfo } from '../bench/frames.ts';
import { DOCUMENT_WORDS, hasUniformBorder } from '../bench/negatives.ts';
import { splitOf } from '../bench/splits.ts';

describe('splitOf', () => {
  it('splits SmartDoc and MIDV-500 by document type and holds out the rest', () => {
    const smartdoc = (docType: string) => splitOf({ dataset: 'smartdoc', docType });
    expect(['datasheet', 'magazine', 'patent'].map(smartdoc)).toEqual(['tune', 'tune', 'tune']);
    expect(['letter', 'paper', 'tax'].map(smartdoc)).toEqual(['heldout', 'heldout', 'heldout']);
    const midv = (docType: string) => splitOf({ dataset: 'midv500', docType });
    expect(['01_alb_id', '11_cze_passport', '41_srb_passport'].map(midv)).toEqual([
      'tune',
      'tune',
      'tune',
    ]);
    expect(['06_bra_passport', '46_ury_passport'].map(midv)).toEqual(['heldout', 'heldout']);
    expect(splitOf({ dataset: 'midv2019', docType: '01_alb_id' })).toBe('heldout');
    expect(splitOf({ dataset: 'dizaj', docType: 'mixed' })).toBe('heldout');
    expect(splitOf({ dataset: 'rwmd', docType: 'mixed' })).toBe('heldout');
    expect(splitOf({ dataset: 'negatives', docType: 'none' })).toBeNull();
  });
});

describe('applyOrientation', () => {
  // 3 × 2 image, one byte per pixel: a b c / d e f.
  const data = Uint8Array.from([1, 2, 3, 4, 5, 6]);
  const oriented = (o: number) => {
    const r = applyOrientation(data, 3, 2, 1, o);
    return [r.width, r.height, ...r.data];
  };

  it('rotates and mirrors like EXIF viewers', () => {
    expect(oriented(1)).toEqual([3, 2, 1, 2, 3, 4, 5, 6]);
    expect(oriented(2)).toEqual([3, 2, 3, 2, 1, 6, 5, 4]);
    expect(oriented(3)).toEqual([3, 2, 6, 5, 4, 3, 2, 1]);
    expect(oriented(4)).toEqual([3, 2, 4, 5, 6, 1, 2, 3]);
    expect(oriented(5)).toEqual([2, 3, 1, 4, 2, 5, 3, 6]);
    // 90° clockwise: the left column (a, d) becomes the top row, read bottom-up (d, a).
    expect(oriented(6)).toEqual([2, 3, 4, 1, 5, 2, 6, 3]);
    expect(oriented(7)).toEqual([2, 3, 6, 3, 5, 2, 4, 1]);
    expect(oriented(8)).toEqual([2, 3, 3, 6, 2, 5, 1, 4]);
  });
});

describe('jpegInfo', () => {
  it('reads the size and the EXIF orientation', () => {
    // SOI, APP1 Exif (big-endian TIFF, IFD0 with orientation 6), SOF0 64 × 32, SOS.
    const tiff = [0x4d, 0x4d, 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0];
    const app1 = [0x45, 0x78, 0x69, 0x66, 0, 0, ...tiff, 0, 0, 0, 0];
    const bytes = Uint8Array.from([
      ...[0xff, 0xd8],
      ...[0xff, 0xe1, 0, app1.length + 2, ...app1],
      ...[0xff, 0xc0, 0, 11, 8, 0, 32, 0, 64, 1, 1, 0x11, 0],
      ...[0xff, 0xda, 0, 2],
    ]);
    expect(jpegInfo(bytes)).toEqual({ width: 64, height: 32, orientation: 6 });
  });
});

describe('DOCUMENT_WORDS', () => {
  it('flags captions that mention printed matter', () => {
    for (const caption of [
      'A desk with a sheet of paper and a laptop.',
      'Man reading the newspaper at the table',
      'A laptop next to some Books.',
      'A menu on a restaurant table',
    ]) {
      expect(DOCUMENT_WORDS.test(caption)).toBe(true);
    }
  });

  it('keeps captions of plain desks, screens and keyboards', () => {
    for (const caption of [
      'A photo of a computer desk with two monitors and a keyboard.',
      'A dining table set with plates and cups',
      'A remote control lying on a couch next to a cell phone.',
    ]) {
      expect(DOCUMENT_WORDS.test(caption)).toBe(false);
    }
  });
});

describe('hasUniformBorder', () => {
  const image = (frame: boolean) => {
    const width = 40;
    const height = 30;
    const data = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const rim = x < 4 || y < 4 || x >= width - 4 || y >= height - 4;
        data[y * width + x] = frame && rim ? 250 : (x * 7 + y * 13) % 256;
      }
    }
    return { width, height, data };
  };

  it('detects a frame added around a photo', () => {
    expect(hasUniformBorder(image(true))).toBe(true);
  });

  it('keeps photos whose content reaches the edge', () => {
    expect(hasUniformBorder(image(false))).toBe(false);
  });
});
