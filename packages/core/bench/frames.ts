/**
 * Frame decoding for the real-data benchmark: ffmpeg decodes each frame at full resolution to
 * 8-bit gray (the luma of the decoded image, within ±1 of `rgbaToGray`), streamed as raw video.
 */
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { GrayImage, RgbaImage } from '../src/image.ts';

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
