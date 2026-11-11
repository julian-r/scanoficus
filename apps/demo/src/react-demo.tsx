import { DocumentScanner, useDocumentScanner } from '@scanoficus/react';
import { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { download, scanFileName } from './download.ts';

/** A custom scanner UI built only with the `useDocumentScanner` hook. */
function HookScanner() {
  // The hook's controller owns its detector worker and disposes it on unmount.
  const scanner = useDocumentScanner();
  const video = useRef<HTMLVideoElement>(null);
  const live =
    scanner.state === 'scanning' || scanner.state === 'capturing' || scanner.state === 'starting';
  const quad = scanner.detection?.quad;

  return (
    <div className="hook-scanner">
      <div className="hook-preview">
        <video ref={video} playsInline muted />
        {quad && scanner.detection && (
          <svg
            viewBox={`0 0 ${scanner.detection.frameWidth} ${scanner.detection.frameHeight}`}
            preserveAspectRatio="xMidYMid slice"
            aria-hidden="true"
          >
            <polygon points={quad.map((p) => `${p.x},${p.y}`).join(' ')} />
          </svg>
        )}
      </div>
      <p role="status">
        State: {scanner.state}
        {scanner.cameraStage && ` (${scanner.cameraStage})`}
        {scanner.detection && ` · ${scanner.detection.stable ? 'stable' : 'moving'}`}
        {scanner.error && ` · ${scanner.error.message}`}
      </p>
      <div className="actions">
        {!live && (
          <button
            type="button"
            className="primary"
            onClick={() => video.current && void scanner.start(video.current)}
          >
            Start camera
          </button>
        )}
        {scanner.state === 'error' && (
          <button type="button" onClick={() => void scanner.retry()}>
            Retry camera
          </button>
        )}
        {live && (
          <>
            <button
              type="button"
              onClick={() => void scanner.capture()}
              disabled={scanner.state !== 'scanning'}
            >
              Capture
            </button>
            <button type="button" onClick={() => scanner.setAutoCapture(!scanner.autoCapture)}>
              Auto-capture: {scanner.autoCapture ? 'on' : 'off'}
            </button>
            <button type="button" onClick={() => scanner.finish()}>
              Stop
            </button>
          </>
        )}
        <button
          type="button"
          disabled={scanner.pages.length === 0}
          onClick={async () =>
            download(await scanner.exportPdf({ pageSize: 'a4' }), scanFileName('pdf'))
          }
        >
          Download PDF ({scanner.pages.length})
        </button>
      </div>
      <div className="thumbs">
        {scanner.pages.map((page) => (
          <figure key={page.id}>
            <img src={page.url} alt={`Page ${page.id}`} />
            <figcaption>
              <select
                value={page.filter}
                aria-label="Filter"
                onChange={(e) =>
                  void scanner.updatePage(page.id, { filter: e.target.value as typeof page.filter })
                }
              >
                <option value="color">Color</option>
                <option value="grayscale">Grayscale</option>
                <option value="bw">B&amp;W</option>
                <option value="photo">Photo</option>
              </select>
              <button type="button" onClick={() => scanner.removePage(page.id)}>
                Delete
              </button>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

/** The ready-made flow through the `<DocumentScanner>` component. */
function ElementScanner() {
  const [open, setOpen] = useState(false);
  const [summary, setSummary] = useState('');
  if (!open) {
    return (
      <div className="actions">
        <button type="button" className="primary" onClick={() => setOpen(true)}>
          Open &lt;DocumentScanner&gt;
        </button>
        {summary && <p>{summary}</p>}
      </div>
    );
  }
  return (
    <div className="scanner-overlay">
      <DocumentScanner
        pageSize="a4"
        style={{ height: '100%' }}
        onCancel={() => setOpen(false)}
        onComplete={({ pages, pdf }) => {
          download(pdf, scanFileName('pdf'));
          setSummary(`Saved ${pages.length} page(s).`);
          setOpen(false);
        }}
      />
    </div>
  );
}

function ReactDemo() {
  return (
    <>
      <section>
        <h1>React</h1>
        <p>
          <code>useDocumentScanner()</code> drives a custom UI; <code>&lt;DocumentScanner&gt;</code>{' '}
          wraps the complete custom-element flow.
        </p>
        <ElementScanner />
      </section>
      <section>
        <h2>Custom UI with the hook</h2>
        <HookScanner />
      </section>
    </>
  );
}

/** Mounts the React demo; returns an unmount function. */
export function mountReact(main: HTMLElement): () => void {
  const root = createRoot(main);
  root.render(<ReactDemo />);
  return () => root.unmount();
}
