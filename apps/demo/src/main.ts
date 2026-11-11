import { defineScanElements } from '@scanoficus/elements';
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
    unmount = mountPhoto(container);
  } else if (page === 'react') {
    const { mountReact } = await import('./react-demo.tsx');
    unmount = mountReact(container);
  } else {
    unmount = mountLive(container, () => debugToggle.checked);
  }
}

window.addEventListener('hashchange', () => void route());
void route();
