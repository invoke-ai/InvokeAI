import { describe, expect, it } from 'vitest';

import { DOCUMENT_TITLE_BASE, DOCUMENT_TITLE_PRODUCT, formatDocumentTitle } from './documentTitle';

const labels = { generating: 'Generating', queued: '3 queued' };

describe('formatDocumentTitle', () => {
  it('leaves the title alone when the queue is empty', () => {
    expect(formatDocumentTitle({ current: 0, labels, percent: 42, total: 0 })).toBe(DOCUMENT_TITLE_BASE);
  });

  it('reports the backlog before anything starts running', () => {
    expect(formatDocumentTitle({ current: 0, labels, percent: null, total: 3 })).toBe(
      `3 queued · ${DOCUMENT_TITLE_PRODUCT}`
    );
  });

  it('leads with percent and position for a multi-image batch', () => {
    expect(formatDocumentTitle({ current: 2, labels, percent: 42, total: 8 })).toBe(
      `42% · 2/8 · ${DOCUMENT_TITLE_PRODUCT}`
    );
  });

  it('drops the position for a single image', () => {
    expect(formatDocumentTitle({ current: 1, labels, percent: 42, total: 1 })).toBe(`42% · ${DOCUMENT_TITLE_PRODUCT}`);
  });

  it('falls back to a word when a single image has no percentage yet', () => {
    expect(formatDocumentTitle({ current: 1, labels, percent: null, total: 1 })).toBe(
      `Generating · ${DOCUMENT_TITLE_PRODUCT}`
    );
  });

  it('still shows the position when a batch has no percentage yet', () => {
    expect(formatDocumentTitle({ current: 2, labels, percent: null, total: 8 })).toBe(
      `2/8 · ${DOCUMENT_TITLE_PRODUCT}`
    );
  });

  it('trails a running title with the short label, not the idle one', () => {
    // The whole point of the split: the idle title names the product in full,
    // and a running title spends that width on the numbers instead.
    const running = formatDocumentTitle({ current: 2, labels, percent: 42, total: 8 });

    expect(DOCUMENT_TITLE_PRODUCT.length).toBeLessThan(DOCUMENT_TITLE_BASE.length);
    expect(running).not.toContain(DOCUMENT_TITLE_BASE);
    expect(running.endsWith(DOCUMENT_TITLE_PRODUCT)).toBe(true);
  });
});
