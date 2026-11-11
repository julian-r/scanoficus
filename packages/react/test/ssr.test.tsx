// @vitest-environment node
import { renderToString } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { DocumentScanner, useDocumentScanner } from '../src/index.ts';

function Probe() {
  const scanner = useDocumentScanner();
  return <p>{scanner.state}</p>;
}

describe('server rendering', () => {
  it('imports and renders without browser globals', () => {
    expect(typeof window).toBe('undefined');
    const html = renderToString(
      <>
        <DocumentScanner className="scanner" />
        <Probe />
      </>,
    );
    expect(html).toBe('<div class="scanner"></div><p>idle</p>');
  });
});
