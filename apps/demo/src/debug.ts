import type { LiveDetection, ScannerController } from '@scanoficus/core/controller';
import { DetectorContext, detectDocument } from '@scanoficus/core/detect';
import { type DetectorStatus, onDetectorStatus } from './detector.ts';

/**
 * Debug overlay for a live scanner: the classic detector's edge map at processing resolution
 * with its quad (blue) and the active detector's quad (orange), plus the active detector and
 * its backend, cold start and median detect time, detection rate, latency, stability and
 * sharpness. The edge map comes from re-running the classic detector on the main thread with an
 * inspectable `DetectorContext` (debug only; the scanner itself detects with the chosen
 * detector).
 */
export function attachDebugOverlay(controller: ScannerController, host: HTMLElement): () => void {
  const panel = document.createElement('div');
  panel.className = 'debug-panel';
  panel.innerHTML = '<canvas></canvas><pre></pre>';
  host.append(panel);
  const canvas = panel.querySelector('canvas') as HTMLCanvasElement;
  const stats = panel.querySelector('pre') as HTMLPreElement;
  const context = new DetectorContext();

  let detectorNote = '';
  let status: DetectorStatus | null = null;
  const offStatus = onDetectorStatus((s) => {
    status = s;
  });
  const offError = controller.on('error', ({ error }) => {
    if (error.message.startsWith('Falling back')) detectorNote = `controller: ${error.message}`;
  });
  const offFrame = controller.on('frame', ({ image, detection: active }) => {
    const detection = detectDocument(image, {}, context);
    const { width, height, edges } = context;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const pixels = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const v = edges[i] ? 255 : 0;
      pixels.data[i * 4] = v;
      pixels.data[i * 4 + 1] = v;
      pixels.data[i * 4 + 2] = v;
      pixels.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(pixels, 0, 0);
    const sx = width / image.width;
    const sy = height / image.height;
    for (const [d, color] of [
      [detection, '#0a84ff'],
      [active, '#ff9f0a'],
    ] as const) {
      if (!d) continue;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const p of d.quad) ctx.lineTo(p.x * sx, p.y * sy);
      ctx.closePath();
      ctx.stroke();
    }
  });
  const offDetection = controller.on('detection', (d: LiveDetection) => {
    stats.textContent = [
      `detections/s ${d.fps}`,
      `detect ms    ${d.detectMs.toFixed(1)}`,
      `stable       ${d.stable ? 'yes' : 'no'}  progress ${(d.progress * 100).toFixed(0)}%`,
      `sharpness    ${d.sharpness.toFixed(0)} (${(d.relativeSharpness * 100).toFixed(0)}% of peak)`,
      `frame        ${d.frameWidth}×${d.frameHeight}`,
      ...detectorLines(status),
      detectorNote,
    ]
      .filter(Boolean)
      .join('\n');
  });
  const offState = controller.on('state', ({ state }) => {
    panel.hidden = state !== 'scanning' && state !== 'capturing';
  });
  return () => {
    offFrame();
    offDetection();
    offState();
    offError();
    offStatus();
    panel.remove();
  };
}

function detectorLines(status: DetectorStatus | null): string[] {
  if (!status) return [];
  return [
    `detector     ${status.running}`,
    `cold start   ${status.coldStartMs === null ? '–' : `${status.coldStartMs.toFixed(0)} ms`}`,
    `detect p50   ${status.p50Ms === null ? '–' : `${status.p50Ms.toFixed(1)} ms`} (last 60)`,
    status.stillMs === null
      ? ''
      : `still        ${status.stillMs.toFixed(0)} ms (${status.stillSize}, refiner)`,
    status.error ? `fallback     ${status.error}` : '',
    status.notice ? `notice       ${status.notice}` : '',
  ];
}
