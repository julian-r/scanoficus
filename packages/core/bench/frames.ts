/**
 * Frame decoding for the real-data benchmark: ffmpeg decodes each frame at full resolution to
 * 8-bit gray (the luma of the decoded image, within ±1 of `rgbaToGray`), streamed as raw video.
 */
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fitLongSide, type GrayImage, type RgbaImage } from '../src/image.ts';
import { resampleArea } from '../src/internal/resample.ts';

/** Pixel size of an image file, via ffprobe. */
export function probeSize(path: string): { width: number; height: number } {
  const out = execFileSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-select_streams',
      'v:0',
      '-show_entries',
      'stream=width,height',
      '-of',
      'csv=p=0',
      path,
    ],
    { encoding: 'utf8' },
  );
  const [width, height] = out.trim().split(',').map(Number);
  if (!width || !height) throw new Error(`ffprobe: cannot read the size of ${path}`);
  return { width, height };
}

/**
 * Pixel size and EXIF orientation (1–8, 1 if absent) of a JPEG file, read from its SOF and APP1
 * (Exif) segments.
 */
export function jpegInfo(bytes: Uint8Array): {
  width: number;
  height: number;
  orientation: number;
} {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint16(0) !== 0xffd8) throw new Error('not a JPEG file');
  let orientation = 1;
  let offset = 2;
  while (offset + 4 <= bytes.length) {
    if (bytes[offset] !== 0xff) throw new Error(`JPEG: no marker at ${offset}`);
    const marker = bytes[offset + 1];
    if (marker === 0xff) {
      offset++;
      continue;
    }
    const length = view.getUint16(offset + 2);
    const start = offset + 4;
    if (
      marker === 0xe1 &&
      view.getUint32(start) === 0x45786966 &&
      view.getUint16(start + 4) === 0
    ) {
      orientation = exifOrientation(view, start + 6) ?? orientation;
    }
    const sof = marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker);
    if (sof) {
      return { height: view.getUint16(start + 1), width: view.getUint16(start + 3), orientation };
    }
    if (marker === 0xda) break;
    offset += 2 + length;
  }
  throw new Error('JPEG: no SOF segment');
}

/** Orientation tag (0x0112) of IFD0 of the TIFF structure at `tiff`, if present. */
function exifOrientation(view: DataView, tiff: number): number | undefined {
  const little = view.getUint16(tiff) === 0x4949;
  const ifd = tiff + view.getUint32(tiff + 4, little);
  const count = view.getUint16(ifd, little);
  for (let i = 0; i < count; i++) {
    const entry = ifd + 2 + 12 * i;
    if (view.getUint16(entry, little) === 0x0112) {
      const value = view.getUint16(entry + 8, little);
      return value >= 1 && value <= 8 ? value : undefined;
    }
  }
  return undefined;
}

/**
 * Applies EXIF orientation `orientation` (1–8) to an interleaved image with `channels` bytes per
 * pixel, returning the image as a viewer shows it (browsers, `createImageBitmap`, PIL
 * `exif_transpose`, OpenCV `imread`). Orientations 5–8 swap width and height.
 */
export function applyOrientation(
  data: Uint8Array,
  width: number,
  height: number,
  channels: number,
  orientation: number,
): { width: number; height: number; data: Uint8Array } {
  if (orientation === 1) return { width, height, data };
  const swap = orientation >= 5;
  const outW = swap ? height : width;
  const outH = swap ? width : height;
  const out = new Uint8Array(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let dx: number;
      let dy: number;
      switch (orientation) {
        case 2:
          dx = width - 1 - x;
          dy = y;
          break;
        case 3:
          dx = width - 1 - x;
          dy = height - 1 - y;
          break;
        case 4:
          dx = x;
          dy = height - 1 - y;
          break;
        case 5:
          dx = y;
          dy = x;
          break;
        case 6:
          dx = height - 1 - y;
          dy = x;
          break;
        case 7:
          dx = height - 1 - y;
          dy = width - 1 - x;
          break;
        default: // 8
          dx = y;
          dy = width - 1 - x;
      }
      const s = (y * width + x) * channels;
      const d = (dy * outW + dx) * channels;
      for (let c = 0; c < channels; c++) out[d + c] = data[s + c];
    }
  }
  return { width: outW, height: outH, data: out };
}

/**
 * Decodes one JPEG photo at full resolution with ffmpeg and applies its EXIF orientation here
 * (ffmpeg's own autorotation is disabled, so the result does not depend on the ffmpeg version).
 * The result is the image as viewers show it, the frame of the `gt.jsonl` quads.
 */
export function decodePhoto<F extends 'gray' | 'rgba'>(
  path: string,
  format: F,
): F extends 'gray' ? GrayImage : RgbaImage {
  const info = jpegInfo(readFileSync(path));
  const channels = format === 'gray' ? 1 : 4;
  const raw = execFileSync(
    'ffmpeg',
    [
      ...['-v', 'error', '-noautorotate', '-i', path, '-frames:v', '1'],
      ...['-f', 'rawvideo', '-pix_fmt', format, '-'],
    ],
    { maxBuffer: 1 << 30 },
  );
  if (raw.length !== info.width * info.height * channels) {
    throw new Error(
      `${path}: ffmpeg returned ${raw.length} bytes for ${info.width}x${info.height}`,
    );
  }
  const bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.length);
  const oriented = applyOrientation(bytes, info.width, info.height, channels, info.orientation);
  const data =
    format === 'gray'
      ? oriented.data
      : new Uint8ClampedArray(oriented.data.buffer, oriented.data.byteOffset, oriented.data.length);
  return { width: oriented.width, height: oriented.height, data } as never;
}

/**
 * Decodes `paths` (all of the same size) in order with one ffmpeg process and calls `onFrame`
 * synchronously for each frame. Gray frames are full resolution; RGBA frames can be scaled to
 * `width × height` by ffmpeg (for visualizations only).
 */
export async function decodeFrames<F extends 'gray' | 'rgba'>(
  paths: readonly string[],
  format: F,
  size: { width: number; height: number },
  onFrame: (frame: F extends 'gray' ? GrayImage : RgbaImage, index: number) => void,
  scaleTo?: { width: number; height: number },
): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'scanoficus-bench-'));
  const list = join(dir, 'frames.txt');
  writeFileSync(list, paths.map((p) => `file '${p.replaceAll("'", "'\\''")}'\n`).join(''));
  const out = scaleTo ?? size;
  const channels = format === 'gray' ? 1 : 4;
  const frameBytes = out.width * out.height * channels;
  const args = [
    '-v',
    'error',
    '-f',
    'concat',
    '-safe',
    '0',
    '-i',
    list,
    '-fps_mode',
    'passthrough',
  ];
  if (scaleTo) args.push('-vf', `scale=${scaleTo.width}:${scaleTo.height}:flags=area`);
  args.push('-f', 'rawvideo', '-pix_fmt', format, '-');
  try {
    await new Promise<void>((resolve, reject) => {
      const ffmpeg = spawn('ffmpeg', args, { stdio: ['ignore', 'pipe', 'pipe'] });
      let buffer = Buffer.alloc(frameBytes);
      let filled = 0;
      let index = 0;
      let stderr = '';
      ffmpeg.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString();
      });
      ffmpeg.stdout.on('data', (chunk: Buffer) => {
        let offset = 0;
        while (offset < chunk.length) {
          const n = Math.min(frameBytes - filled, chunk.length - offset);
          chunk.copy(buffer, filled, offset, offset + n);
          filled += n;
          offset += n;
          if (filled === frameBytes) {
            const data =
              format === 'gray'
                ? new Uint8Array(buffer.buffer, buffer.byteOffset, frameBytes)
                : new Uint8ClampedArray(buffer.buffer, buffer.byteOffset, frameBytes);
            onFrame({ width: out.width, height: out.height, data } as never, index++);
            buffer = Buffer.alloc(frameBytes);
            filled = 0;
          }
        }
      });
      ffmpeg.on('error', reject);
      ffmpeg.on('close', (code) => {
        if (code !== 0) reject(new Error(`ffmpeg exited with ${code}: ${stderr}`));
        else if (index !== paths.length || filled !== 0) {
          reject(new Error(`ffmpeg decoded ${index} of ${paths.length} frames: ${stderr}`));
        } else resolve();
      });
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * Area-averaging downscale of an RGBA image so that its longer side is at most `longSide`: the
 * library's `resampleArea` per channel (as `downscaleGray` does for gray). Returns the input if
 * it is already small enough.
 */
export function downscaleRgba(rgba: RgbaImage, longSide: number): RgbaImage {
  const size = fitLongSide(rgba.width, rgba.height, longSide);
  if (size.width === rgba.width && size.height === rgba.height) return rgba;
  const n = rgba.width * rgba.height;
  const plane = new Uint8Array(n);
  const small = new Uint8Array(size.width * size.height);
  const out = new Uint8ClampedArray(size.width * size.height * 4);
  for (let c = 0; c < 4; c++) {
    for (let i = 0; i < n; i++) plane[i] = rgba.data[4 * i + c];
    resampleArea(plane, rgba.width, rgba.height, small, size.width, size.height);
    for (let i = 0; i < small.length; i++) out[4 * i + c] = small[i];
  }
  return { width: size.width, height: size.height, data: out };
}
