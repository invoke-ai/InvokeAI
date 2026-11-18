import { describe, expect, it } from 'vitest';

import {
  applyServerOutputRead,
  EMPTY_TRANSCRIPT,
  getTranscriptText,
  iterateRows,
  type ServerOutputRead,
  type ServerOutputRow,
  type Transcript,
} from './transcript';

const row = (id: number, text = `line ${id}`): ServerOutputRow => ({ id, styles: [], text, truncated: false });

const read = (overrides: Partial<ServerOutputRead>): ServerOutputRead => ({
  cursor: 'c',
  floor_id: 1,
  has_more: false,
  reset: null,
  rows: [],
  run_id: 'run-a',
  snapshot_truncated: false,
  ...overrides,
});

const ids = (transcript: Transcript): number[] => [...iterateRows(transcript)].map((held) => held.id);
const texts = (transcript: Transcript): string[] => [...iterateRows(transcript)].map((held) => held.text);

const snapshot = (rows: ServerOutputRow[], overrides: Partial<ServerOutputRead> = {}): Transcript =>
  applyServerOutputRead(EMPTY_TRANSCRIPT, read({ reset: 'initial', rows, ...overrides }));

describe('applyServerOutputRead', () => {
  it('replaces everything with a snapshot and reports only resets the user should know about', () => {
    const first = snapshot([row(1), row(2)], { snapshot_truncated: true });

    expect(ids(first)).toEqual([1, 2]);
    expect(first).toMatchObject({ reset: null, rowCount: 2, runId: 'run-a', truncated: true });

    const restarted = applyServerOutputRead(first, read({ reset: 'restarted', rows: [row(1, 'fresh')], run_id: 'b' }));

    expect(texts(restarted)).toEqual(['fresh']);
    expect(restarted).toMatchObject({ reset: { reason: 'restarted', serial: 1 }, runId: 'b', truncated: false });
    // A later replacement is a new notice, even for the same reason.
    expect(applyServerOutputRead(restarted, read({ reset: 'expired', rows: [row(9)] })).reset).toEqual({
      reason: 'expired',
      serial: 2,
    });
  });

  it('upserts changed rows by ID, keeping unchanged blocks as they were', () => {
    const rows = Array.from({ length: 130 }, (_, index) => row(index + 1));
    const held = snapshot(rows);
    // A progress rewrite of row 129, a new row, and an older row rewritten while another is appended.
    const next = applyServerOutputRead(
      held,
      read({ rows: [row(2, 'rewritten'), row(129, '50%'), row(131, 'appended')] })
    );

    expect(ids(next)).toEqual([...rows.map((held) => held.id), 131]);
    expect(texts(next)[1]).toBe('rewritten');
    expect(texts(next)[128]).toBe('50%');
    // Rows 64-127 share a block no change touched.
    expect(next.blocks[1]).toBe(held.blocks[1]);
    expect(next.blocks[0]).not.toBe(held.blocks[0]);
  });

  it('orders rows by ID whatever order the response lists them in', () => {
    const held = snapshot([row(1), row(5)]);
    const next = applyServerOutputRead(held, read({ rows: [row(4), row(3), row(4, 'latest')] }));

    expect(ids(next)).toEqual([1, 3, 4, 5]);
    expect(texts(next)[2]).toBe('latest');
  });

  it('drops rows below the floor and marks the transcript truncated', () => {
    const held = snapshot(Array.from({ length: 100 }, (_, index) => row(index + 1)));
    const next = applyServerOutputRead(held, read({ floor_id: 70, rows: [row(101)] }));

    expect(ids(next)[0]).toBe(70);
    expect(next.rowCount).toBe(32);
    expect(next.truncated).toBe(true);
    expect(next.reset).toBeNull();
  });

  it('returns the same transcript when a read changes nothing', () => {
    const held = snapshot([row(1), row(2)]);

    expect(applyServerOutputRead(held, read({ floor_id: 1 }))).toBe(held);
  });

  it('drops the oldest rows past the row bound', () => {
    const limits = { maxRows: 100, maxSize: Number.POSITIVE_INFINITY };
    const held = applyServerOutputRead(
      EMPTY_TRANSCRIPT,
      read({ reset: 'initial', rows: Array.from({ length: 90 }, (_, index) => row(index + 1)) }),
      limits
    );

    expect(held.truncated).toBe(false);

    const next = applyServerOutputRead(
      held,
      read({ rows: Array.from({ length: 30 }, (_, index) => row(index + 91)) }),
      limits
    );

    expect(next.rowCount).toBe(100);
    expect(ids(next)[0]).toBe(21);
    expect(ids(next).at(-1)).toBe(120);
    expect(next.truncated).toBe(true);
  });

  it('drops the oldest rows past the size bound, a huge row included', () => {
    const limits = { maxRows: Number.POSITIVE_INFINITY, maxSize: 10_000 };
    const held = applyServerOutputRead(
      EMPTY_TRANSCRIPT,
      read({ reset: 'initial', rows: [row(1), row(2, 'x'.repeat(5_000)), row(3)] }),
      limits
    );
    const next = applyServerOutputRead(held, read({ rows: [row(4, 'y'.repeat(6_000))] }), limits);

    expect(ids(next)).toEqual([3, 4]);
    expect(next.size).toBeLessThanOrEqual(10_000);
    expect(next.truncated).toBe(true);
  });
});

describe('getTranscriptText', () => {
  it('joins rows and cuts a range at its first and last offsets', () => {
    const held = snapshot([row(1, 'alpha'), row(2, 'beta'), row(3, 'gamma')]);

    expect(getTranscriptText(held)).toBe('alpha\nbeta\ngamma');
    expect(getTranscriptText(held, { fromId: 1, fromOffset: 2, toId: 3, toOffset: 3 })).toBe('pha\nbeta\ngam');
    expect(getTranscriptText(held, { fromId: 2, fromOffset: 1, toId: 2, toOffset: 3 })).toBe('et');
  });
});
