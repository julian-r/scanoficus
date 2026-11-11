import type { ScanPage } from '@scanoficus/core/controller';
import { download, scanFileName } from './download.ts';

/** Renders the outcome of a scan: page thumbnails, PDF and image downloads. */
export function renderResults(container: HTMLElement, pages: readonly ScanPage[], pdf: Blob): void {
  container.replaceChildren();
  const heading = document.createElement('h2');
  heading.textContent = `${pages.length} page${pages.length === 1 ? '' : 's'} scanned`;
  const actions = document.createElement('div');
  actions.className = 'actions';
  const pdfButton = button(`Download PDF (${Math.round(pdf.size / 1024)} KB)`, () =>
    download(pdf, scanFileName('pdf')),
  );
  const imagesButton = button('Download images', () => {
    pages.forEach((p, i) => {
      download(p.image, scanFileName(`${i + 1}.jpg`));
    });
  });
  actions.append(pdfButton, imagesButton);
  const grid = document.createElement('div');
  grid.className = 'thumbs';
  for (const page of pages) {
    // Copy the image: the page URLs are revoked when the scanner is disposed.
    const img = document.createElement('img');
    img.src = URL.createObjectURL(page.image);
    img.alt = `Page ${pages.indexOf(page) + 1} (${page.filter})`;
    grid.append(img);
  }
  container.append(heading, actions, grid);
}

export function button(label: string, onClick: () => void, className = ''): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = label;
  b.className = className;
  b.addEventListener('click', onClick);
  return b;
}
