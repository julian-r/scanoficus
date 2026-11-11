/** Triggers a download of `blob` and releases the URL afterwards. */
export function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

/** Timestamped file name, e.g. `scan-2026-11-10-1805.pdf`. */
export function scanFileName(extension: string): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `scan-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.${extension}`;
}
