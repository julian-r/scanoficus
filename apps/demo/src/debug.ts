import type { LiveDetection, ScannerController } from '@scanoficus/core/controller';
import { DetectorContext, detectDocument } from '@scanoficus/core/detect';
import { rgbaToGray } from '@scanoficus/core/image';

/**
 * Debug overlay for a live scanner: the detector's edge map at processing resolution with the
 * detected quad, plus detection rate, latency, stability and sharpness. The edge map comes from
 * re-running the detector on the main thread with an inspectable `DetectorContext` (debug
 * only; the scanner itself detects in a worker).
 */
export function attachDebugOverlay(controller: ScannerController, host: HTMLElement): () => void {
  const panel = document.createElement('div');
  panel.className = 'debug-panel';
  panel.innerHTML = '<canvas></canvas><pre></pre>';
  host.append(panel);
  const canvas = panel.querySelector('canvas') as HTMLCanvasElement;
  const stats = panel.querySelector('pre') as HTMLPreElement;
  const context = new DetectorContext();

  let detectorNote = 'detector: worker';
  const offError = controller.on('error', ({ error }) => {
    if (error.message.startsWith('Falling back'))
      detectorNote = `detector: main thread (${error.message})`;
  });
  const offFrame = controller.on('frame', ({ image }) => {
    const detection = detectDocument(rgbaToGray(image), {}, context);
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
    if (detection) {
      ctx.strokeStyle = '#0a84ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const p of detection.quad) ctx.lineTo(p.x, p.y);
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
      detectorNote,
    ].join('\n');
  });
  const offState = controller.on('state', ({ state }) => {
    panel.hidden = state !== 'scanning' && state !== 'capturing';
  });
  return () => {
    offFrame();
    offDetection();
    offState();
    offError();
    panel.remove();
  };
}
