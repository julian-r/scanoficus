import cocoCreditsUrl from '@scanoficus/detector-nn/models/docquadnet3/coco-credits.csv?url&no-inline';
import modelLicenseUrl from '@scanoficus/detector-nn/models/LICENSE.md?url&no-inline';
import { defineScanElements } from '@scanoficus/elements';
import {
  DETECTOR_LABELS,
  type DetectorChoice,
  loadChoice,
  onDetectorStatus,
  saveChoice,
} from './detector.ts';
import { mountLive } from './live.ts';
import { mountPhoto } from './photo.ts';
import './style.css';

defineScanElements();

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch((error) => {
    console.warn('Service worker registration failed', error);
  });
}

const main = document.getElementById('app') as HTMLElement;
const debugToggle = document.getElementById('debug') as HTMLInputElement;
debugToggle.checked = localStorage.getItem('scanoficus-debug') === '1';
debugToggle.addEventListener('change', () => {
  localStorage.setItem('scanoficus-debug', debugToggle.checked ? '1' : '0');
});

// Detector choice: applies to the live scanner and the photo import (the page is remounted).
const detectorSelect = document.getElementById('detector') as HTMLSelectElement;
for (const [value, label] of Object.entries(DETECTOR_LABELS)) {
  detectorSelect.append(new Option(label, value));
}
detectorSelect.value = loadChoice();
detectorSelect.addEventListener('change', () => {
  saveChoice(detectorSelect.value as DetectorChoice);
  void route();
});
// Exposed for the headless browser check.
onDetectorStatus((status) => {
  (window as unknown as { scanoficusDetector: unknown }).scanoficusDetector = status;
});

// Attribution that must travel with the neural detector's weights (SmartDoc; rounds 2 and 3 also
// Dizaj, CC BY 4.0; round 3 also COCO / Flickr, CC BY 2.0, with the image list).
const credits = document.createElement('footer');
credits.className = 'credits';
credits.innerHTML = `Neural detector: DocQuadNet weights (BSD-3-Clause), trained with SmartDoc 2015
frames (CC BY 4.0, © Chazalon et al.); rounds 2 and 3 also with "A New Image Dataset for Document Corner
Localization" by Shima Dizaj, Mendeley Data v3,
<a href="https://doi.org/10.17632/x3nm4cxr83.3">https://doi.org/10.17632/x3nm4cxr83.3</a>, licensed
under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Changes: photos resized,
cropped, rotated and augmented for training; labels converted. Content shown in the photos can
belong to third parties (Mendeley Data). Round 3 also with 6,329 COCO train2017 photos (Flickr, CC
BY 2.0 or no known copyright restrictions; cropped, resized and augmented),
<a href="${cocoCreditsUrl}">list of images with authors, titles and sources</a>. <a href="${modelLicenseUrl}">Model
license and attribution</a>.`;
document.body.append(credits);

let unmount: (() => void) | null = null;

async function route(): Promise<void> {
  unmount?.();
  unmount = null;
  const page = location.hash.replace(/^#\//, '') || 'live';
  for (const link of document.querySelectorAll('nav a')) {
    link.toggleAttribute('aria-current', link.getAttribute('href') === `#/${page}`);
  }
  // A fresh container per page: React owns its root element exclusively.
  const container = document.createElement('div');
  main.replaceChildren(container);
  if (page === 'photo') {
    unmount = mountPhoto(container, loadChoice());
  } else if (page === 'react') {
    const { mountReact } = await import('./react-demo.tsx');
    unmount = mountReact(container);
  } else {
    unmount = mountLive(container, () => debugToggle.checked, loadChoice);
  }
}

window.addEventListener('hashchange', () => void route());
void route();
