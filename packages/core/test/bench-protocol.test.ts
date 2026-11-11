import { describe, expect, it } from 'vitest';
import { DOCUMENT_WORDS, hasUniformBorder } from '../bench/negatives.ts';
import { splitOf } from '../bench/splits.ts';

describe('splitOf', () => {
  it('splits SmartDoc by document type and holds out MIDV entirely', () => {
    const smartdoc = (docType: string) => splitOf({ dataset: 'smartdoc', docType });
    expect(['datasheet', 'magazine', 'patent'].map(smartdoc)).toEqual(['tune', 'tune', 'tune']);
    expect(['letter', 'paper', 'tax'].map(smartdoc)).toEqual(['heldout', 'heldout', 'heldout']);
    expect(splitOf({ dataset: 'midv500', docType: '01_alb_id' })).toBe('heldout');
    expect(splitOf({ dataset: 'midv2019', docType: '01_alb_id' })).toBe('heldout');
    expect(splitOf({ dataset: 'negatives', docType: 'none' })).toBeNull();
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
