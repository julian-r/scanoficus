import { readFileSync } from 'node:fs';
import { PDFDocument } from 'pdf-lib';
import { describe, expect, it } from 'vitest';
import { createPdf, PAGE_SIZES, readJpegInfo } from '../src/pdf/index.ts';

const fixture = (name: string) =>
  new Uint8Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
const rgb = fixture('rgb.jpg'); // 8×6, 3 components
const gray = fixture('gray.jpg'); // 4×10, 1 component
const progressive = fixture('prog.jpg'); // 8×6 progressive

const text = (bytes: Uint8Array) => new TextDecoder('latin1').decode(bytes);

describe('readJpegInfo', () => {
  it('reads size and components of baseline and progressive JPEGs', () => {
    expect(readJpegInfo(rgb)).toEqual({ width: 8, height: 6, components: 3 });
    expect(readJpegInfo(gray)).toEqual({ width: 4, height: 10, components: 1 });
    expect(readJpegInfo(progressive)).toEqual({ width: 8, height: 6, components: 3 });
  });

  it('rejects non-JPEG data', () => {
    expect(() => readJpegInfo(new Uint8Array([0x89, 0x50, 0x4e, 0x47]))).toThrow('Not a JPEG');
  });
});

describe('createPdf', () => {
  it('writes a cross-reference table whose offsets point at the objects', () => {
    const pdf = text(createPdf([rgb, gray], { title: 'Scan' }));
    expect(pdf.startsWith('%PDF-1.4\n')).toBe(true);
    expect(pdf.endsWith('%%EOF\n')).toBe(true);
    const startxref = Number(/startxref\n(\d+)\n%%EOF\n$/.exec(pdf)?.[1]);
    expect(pdf.slice(startxref, startxref + 4)).toBe('xref');
    const [, first, count] = /^xref\n(\d+) (\d+)\n/.exec(pdf.slice(startxref)) ?? [];
    expect(Number(first)).toBe(0);
    const entries = pdf
      .slice(startxref)
      .split('\n')
      .slice(2, 2 + Number(count));
    expect(entries[0]).toBe('0000000000 65535 f ');
    for (let id = 1; id < entries.length; id++) {
      expect(entries[id]).toMatch(/^\d{10} 00000 n $/);
      const offset = Number(entries[id].slice(0, 10));
      expect(pdf.slice(offset, offset + `${id} 0 obj`.length)).toBe(`${id} 0 obj`);
    }
    expect(pdf).toContain(`/Size ${count}`);
  });

  it('embeds the JPEG bytes unchanged with stream lengths that match', () => {
    const bytes = createPdf([rgb]);
    const pdf = text(bytes);
    const start = pdf.indexOf(text(rgb));
    expect(start).toBeGreaterThan(0);
    expect(pdf).toContain(`/Filter /DCTDecode /Length ${rgb.length} >>`);
    expect(pdf).toContain('/ColorSpace /DeviceRGB');
  });

  it('parses with pdf-lib and sizes pages from the image at the given dpi', async () => {
    const doc = await PDFDocument.load(
      createPdf([rgb, gray, progressive], { dpi: 72, title: 'Bücher' }),
    );
    expect(doc.getPageCount()).toBe(3);
    expect(doc.getPage(0).getSize()).toEqual({ width: 8, height: 6 });
    expect(doc.getPage(1).getSize()).toEqual({ width: 4, height: 10 });
    expect(doc.getTitle()).toBe('Bücher');
  });

  it('fits images into A4 / Letter, landscape for landscape images', async () => {
    const doc = await PDFDocument.load(createPdf([rgb, gray], { pageSize: 'a4', margin: 36 }));
    const [a4w, a4h] = PAGE_SIZES.a4;
    expect(doc.getPage(0).getSize()).toEqual({ width: a4h, height: a4w });
    expect(doc.getPage(1).getSize()).toEqual({ width: a4w, height: a4h });
    const letter = await PDFDocument.load(createPdf([gray], { pageSize: 'letter' }));
    expect(letter.getPage(0).getSize()).toEqual({ width: 612, height: 792 });
    // Gray page fitted by height: 10 px → 792 pt tall, 4 px → 316.8 pt wide, centered.
    expect(text(createPdf([gray], { pageSize: 'letter' }))).toContain('q 316.8 0 0 792 147.6 0 cm');
  });

  it('rejects empty documents', () => {
    expect(() => createPdf([])).toThrow(RangeError);
  });
});
