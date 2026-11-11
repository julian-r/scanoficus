/** Basic facts about a baseline or progressive JPEG, read from its SOF segment. */
export interface JpegInfo {
  width: number;
  height: number;
  /** 1 = gray, 3 = YCbCr/RGB, 4 = CMYK. */
  components: number;
}

/** Reads width, height and component count from JPEG bytes. Throws for non-JPEG data. */
export function readJpegInfo(bytes: Uint8Array): JpegInfo {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) throw new Error('Not a JPEG (missing SOI marker)');
  let i = 2;
  while (i + 4 <= bytes.length) {
    if (bytes[i] !== 0xff) throw new Error('Corrupt JPEG: expected a marker');
    const marker = bytes[i + 1];
    if (marker === 0xff) {
      i++; // fill byte
      continue;
    }
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) {
      i += 2; // stand-alone marker without length
      continue;
    }
    const length = (bytes[i + 2] << 8) | bytes[i + 3];
    // SOF0–SOF15 except DHT (C4), JPG (C8) and DAC (CC).
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return {
        height: (bytes[i + 5] << 8) | bytes[i + 6],
        width: (bytes[i + 7] << 8) | bytes[i + 8],
        components: bytes[i + 9],
      };
    }
    i += 2 + length;
  }
  throw new Error('Corrupt JPEG: no SOF segment found');
}

/** Page size presets in PDF points (1/72 inch), portrait. */
export const PAGE_SIZES: Record<'a4' | 'letter', readonly [number, number]> = {
  a4: [595.28, 841.89],
  letter: [612, 792],
};

/** Options for {@link createPdf}. */
export interface PdfOptions {
  /**
   * `'image'` (default): every page has its image's aspect ratio, sized from `dpi`.
   * `'a4'` / `'letter'`: the image is fitted (aspect preserved, centered) inside the page minus
   * `margin`; the page is rotated to landscape for landscape images.
   */
  pageSize?: 'image' | 'a4' | 'letter';
  /** Resolution used to size `'image'` pages. Default 150. */
  dpi?: number;
  /** Margin in points for `'a4'` / `'letter'`. Default 0. */
  margin?: number;
  /** Document title (Info dictionary). */
  title?: string;
}

/**
 * Writes a PDF 1.4 file with one JPEG image per page, embedded as-is with `/DCTDecode` (no
 * re-encoding). Gray and RGB JPEGs are supported.
 */
export function createPdf(jpegs: readonly Uint8Array[], options: PdfOptions = {}): Uint8Array {
  if (jpegs.length === 0) throw new RangeError('A PDF needs at least one page');
  const writer = new PdfWriter();
  const pageSize = options.pageSize ?? 'image';
  const dpi = options.dpi ?? 150;
  const margin = options.margin ?? 0;

  // Object numbers: 1 catalog, 2 page tree, 3 info, then (page, content, image) per page.
  const pageIds = jpegs.map((_, i) => 4 + i * 3);
  writer.object(1, '<< /Type /Catalog /Pages 2 0 R >>');
  writer.object(
    2,
    `<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(' ')}] /Count ${jpegs.length} >>`,
  );
  const title = options.title ? ` /Title ${pdfString(options.title)}` : '';
  writer.object(3, `<< /Producer (scanoficus)${title} >>`);

  jpegs.forEach((jpeg, i) => {
    const info = readJpegInfo(jpeg);
    if (info.components !== 1 && info.components !== 3) {
      throw new Error(`Unsupported JPEG with ${info.components} components`);
    }
    const [pageWidth, pageHeight, x, y, w, h] = layout(info, pageSize, dpi, margin);
    const id = pageIds[i];
    writer.object(
      id,
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(pageWidth)} ${num(pageHeight)}] ` +
        `/Resources << /XObject << /Im0 ${id + 2} 0 R >> >> /Contents ${id + 1} 0 R >>`,
    );
    writer.stream(id + 1, '', ascii(`q ${num(w)} 0 0 ${num(h)} ${num(x)} ${num(y)} cm /Im0 Do Q`));
    writer.stream(
      id + 2,
      `/Type /XObject /Subtype /Image /Width ${info.width} /Height ${info.height} ` +
        `/ColorSpace /${info.components === 1 ? 'DeviceGray' : 'DeviceRGB'} ` +
        '/BitsPerComponent 8 /Filter /DCTDecode',
      jpeg,
    );
  });
  return writer.finish(3 + jpegs.length * 3, 1, 3);
}

/** Page size and image placement `[pageW, pageH, x, y, w, h]` in points. */
function layout(
  info: JpegInfo,
  pageSize: 'image' | 'a4' | 'letter',
  dpi: number,
  margin: number,
): [number, number, number, number, number, number] {
  if (pageSize === 'image') {
    const w = (info.width * 72) / dpi;
    const h = (info.height * 72) / dpi;
    return [w, h, 0, 0, w, h];
  }
  const [short, long] = PAGE_SIZES[pageSize];
  const landscape = info.width > info.height;
  const pageW = landscape ? long : short;
  const pageH = landscape ? short : long;
  const scale = Math.min((pageW - 2 * margin) / info.width, (pageH - 2 * margin) / info.height);
  const w = info.width * scale;
  const h = info.height * scale;
  return [pageW, pageH, (pageW - w) / 2, (pageH - h) / 2, w, h];
}

/** Number formatting for PDF: at most 3 decimals, no exponent. */
function num(v: number): string {
  return String(Math.round(v * 1000) / 1000);
}

/** PDF text string: literal for printable ASCII, UTF-16BE hex otherwise. */
function pdfString(text: string): string {
  if (/^[\x20-\x7e]*$/.test(text)) return `(${text.replace(/[\\()]/g, '\\$&')})`;
  let hex = 'FEFF';
  for (let i = 0; i < text.length; i++) hex += text.charCodeAt(i).toString(16).padStart(4, '0');
  return `<${hex}>`;
}

function ascii(text: string): Uint8Array {
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xff;
  return out;
}

/** Accumulates objects and records their byte offsets for the cross-reference table. */
class PdfWriter {
  private readonly chunks: Uint8Array[] = [];
  private readonly offsets: number[] = [];
  private length = 0;

  constructor() {
    // Header plus a comment with high-bit bytes so tools treat the file as binary.
    this.push(ascii('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'));
  }

  object(id: number, body: string): void {
    this.offsets[id] = this.length;
    this.push(ascii(`${id} 0 obj\n${body}\nendobj\n`));
  }

  stream(id: number, dictionary: string, data: Uint8Array): void {
    this.offsets[id] = this.length;
    this.push(ascii(`${id} 0 obj\n<< ${dictionary} /Length ${data.length} >>\nstream\n`));
    this.push(data);
    this.push(ascii('\nendstream\nendobj\n'));
  }

  finish(lastId: number, rootId: number, infoId: number): Uint8Array {
    const xref = this.length;
    let table = `xref\n0 ${lastId + 1}\n0000000000 65535 f \n`;
    for (let id = 1; id <= lastId; id++) {
      const offset = this.offsets[id];
      if (offset === undefined) throw new Error(`Object ${id} was not written`);
      table += `${String(offset).padStart(10, '0')} 00000 n \n`;
    }
    table += `trailer\n<< /Size ${lastId + 1} /Root ${rootId} 0 R /Info ${infoId} 0 R >>\n`;
    table += `startxref\n${xref}\n%%EOF\n`;
    this.push(ascii(table));
    const out = new Uint8Array(this.length);
    let at = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, at);
      at += chunk.length;
    }
    return out;
  }

  private push(chunk: Uint8Array): void {
    this.chunks.push(chunk);
    this.length += chunk.length;
  }
}
