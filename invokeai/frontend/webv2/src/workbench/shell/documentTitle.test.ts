import { describe, expect, it } from 'vitest';

import {
  DOCUMENT_TITLE_BASE,
  DOCUMENT_TITLE_PRODUCT,
  formatDocumentTitle,
  getBatchDocumentTitlePercent,
} from './documentTitle';

const labels = { generating: 'Generating', queued: '3 queued' };

describe('formatDocumentTitle', () => {
  it.each([
    ['leaves the title alone when the queue is empty', { current: 0, percent: 42, total: 0 }, DOCUMENT_TITLE_BASE],
    [
      'reports the backlog before anything starts running',
      { current: 0, percent: null, total: 3 },
      `3 queued · ${DOCUMENT_TITLE_PRODUCT}`,
    ],
    [
      'leads with percent and position for a multi-image batch',
      { current: 2, percent: 42, total: 8 },
      `42% · 2/8 · ${DOCUMENT_TITLE_PRODUCT}`,
    ],
    ['drops the position for a single image', { current: 1, percent: 42, total: 1 }, `42% · ${DOCUMENT_TITLE_PRODUCT}`],
    [
      'falls back to a word when a single image has no percentage yet',
      { current: 1, percent: null, total: 1 },
      `Generating · ${DOCUMENT_TITLE_PRODUCT}`,
    ],
    [
      'still shows the position when a batch has no percentage yet',
      { current: 2, percent: null, total: 8 },
      `2/8 · ${DOCUMENT_TITLE_PRODUCT}`,
    ],
  ])('%s', (_, queue, expected) => {
    expect(formatDocumentTitle({ ...queue, labels })).toBe(expected);
  });

  it('trails a running title with the short label, not the idle one', () => {
    const running = formatDocumentTitle({ current: 2, labels, percent: 42, total: 8 });

    expect(DOCUMENT_TITLE_PRODUCT.length).toBeLessThan(DOCUMENT_TITLE_BASE.length);
    expect(running).not.toContain(DOCUMENT_TITLE_BASE);
    expect(running.endsWith(DOCUMENT_TITLE_PRODUCT)).toBe(true);
  });
});

describe('getBatchDocumentTitlePercent', () => {
  it('retains live progress for a single-iteration Invoke', () => {
    expect(getBatchDocumentTitlePercent({ batchCount: 1, completedCount: 0, singleItemPercent: 87 })).toBe(87);
  });

  it('has no misleading percent before any iteration finishes', () => {
    expect(getBatchDocumentTitlePercent({ batchCount: 8, completedCount: 0, singleItemPercent: 87 })).toBeNull();
  });

  it('uses stable overall batch progress regardless of other GPUs progress events', () => {
    expect(getBatchDocumentTitlePercent({ batchCount: 8, completedCount: 2, singleItemPercent: 10 })).toBe(25);
    expect(getBatchDocumentTitlePercent({ batchCount: 8, completedCount: 2, singleItemPercent: 90 })).toBe(25);
  });

  it('advances through completions and caps at 100 percent', () => {
    expect(getBatchDocumentTitlePercent({ batchCount: 8, completedCount: 3, singleItemPercent: 99 })).toBe(38);
    expect(getBatchDocumentTitlePercent({ batchCount: 8, completedCount: 8, singleItemPercent: 12 })).toBe(100);
    expect(getBatchDocumentTitlePercent({ batchCount: 8, completedCount: 9, singleItemPercent: 12 })).toBe(100);
  });
});
