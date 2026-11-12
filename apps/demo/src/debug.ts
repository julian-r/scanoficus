import type {
  AutoCaptureInfo,
  LiveDetection,
  ScannerController,
} from '@scanoficus/core/controller';
import { DetectorContext, detectDocument } from '@scanoficus/core/detect';
import { type DetectorStatus, onDetectorStatus } from './detector.ts';

/**
 * Debug overlay for a live scanner: the classic detector's edge map at processing resolution
 * with its quad (blue) and the active detector's quad (orange), plus the active detector and
 * its backend, cold start and median detect time, detection rate, latency, and why auto-capture
 * does (not) fire: every criterion with its value, limit and ✓/✗, and the first failing one. The
 * edge map comes from re-running the classic detector on the main thread with an
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
      ...autoCaptureLines(d),
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

/** Auto-capture criteria: value vs limit with ✓/✗, the blocker, and the progress. */
function autoCaptureLines(d: LiveDetection): string[] {
  const a: AutoCaptureInfo | undefined = d.autoCapture;
  if (!a) return [`progress     ${(d.progress * 100).toFixed(0)}%`];
  const mark = (ok: boolean | null) => (ok === null ? '·' : ok ? '✓' : '✗');
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const relative = d.relativeSharpness;
  const rate = a.detectionRate;
  const waiting =
    a.blocker === 'slow-detection' && rate !== null
      ? `detection rate too low for auto-capture: ${rate.toFixed(1)}/s (needs ≥ ${a.minDetectionRate}/s)`
      : BLOCKER_TEXT[a.blocker ?? 'settling'];
  return [
    `auto-capture ${a.ready ? 'READY' : `waiting: ${waiting}`}`,
    `${mark(rate === null ? null : rate >= a.minDetectionRate)} rate       ${
      rate === null ? '–' : rate.toFixed(1)
    } ≥ ${a.minDetectionRate} detections/s`,
    `${mark(a.speed === null ? null : a.speed <= a.maxSpeed)} speed      ${
      a.speed === null ? '–' : a.speed.toFixed(3)
    } ≤ ${a.maxSpeed} diag/s`,
    `${mark(a.jitter === null ? null : a.jitter <= a.maxJitter)} jitter     ${
      a.jitter === null ? '–' : pct(a.jitter)
    } ≤ ${pct(a.maxJitter)} of diag`,
    `${mark(a.windowFrames >= a.minFrames && a.windowSpanMs >= a.minWindowSpanMs)} frames     ${
      a.windowFrames
    } ≥ ${a.minFrames} over ${a.windowSpanMs.toFixed(0)} ≥ ${a.minWindowSpanMs} ms`,
    `${mark(relative >= a.minRelativeSharpness)} sharpness  ${(relative * 100).toFixed(0)}% ≥ ${(
      a.minRelativeSharpness * 100
    ).toFixed(0)}% of ref ${a.sharpnessReference.toFixed(0)} (now ${d.sharpness.toFixed(0)})`,
    a.rotationRate === null
      ? `· gyro       off (no motion samples)`
      : `${mark(a.rotationRate <= a.maxRotationRate)} gyro       ${a.rotationRate.toFixed(1)} ≤ ${
          a.maxRotationRate
        } °/s`,
    `${mark(a.readyMs >= a.stableMs)} time       ${a.readyMs.toFixed(0)} / ${a.stableMs} ms ready (${(
      d.progress * 100
    ).toFixed(0)}%)${a.bestFrame ? ', best frame kept' : ''}`,
  ];
}

const BLOCKER_TEXT: Record<NonNullable<AutoCaptureInfo['blocker']>, string> = {
  disabled: 'auto-capture off',
  'no-document': 'no document',
  holding: 'detection missed (holding last quad)',
  captured: 'already captured, show the next page',
  settling: 'collecting frames',
  moving: 'moving too fast',
  jitter: 'detections scatter',
  'device-motion': 'device rotating (gyro)',
  blurry: 'blurry vs recent frames',
  'slow-detection': 'detection rate too low for auto-capture',
};

function detectorLines(status: DetectorStatus | null): string[] {
  if (!status) return [];
  return [
    `detector     ${status.running}`,
    `cold start   ${status.coldStartMs === null ? '–' : `${status.coldStartMs.toFixed(0)} ms`}`,
    `detect p50   ${status.p50Ms === null ? '–' : `${status.p50Ms.toFixed(1)} ms`} (last 60)`,
    status.stillMs === null
      ? ''
      : `still        ${status.stillMs.toFixed(0)} ms (${status.stillSize}, refiner)`,
    status.decision && !status.error ? `decision     ${status.decision}` : '',
    status.error ? `fallback     ${status.error}` : '',
    status.notice ? `notice       ${status.notice}` : '',
  ];
}
