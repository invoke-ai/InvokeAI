import { describe, expect, it } from 'vitest';

import {
  describeIndexProgress,
  formatDuration,
  getDisplayPercent,
  getIndexPercent,
  hasProgressed,
  isIndexing,
  STALE_AFTER_MS,
  type ImageIndexCounts,
} from './indexProgress';

const counts = (overrides: Partial<ImageIndexCounts> = {}): ImageIndexCounts => ({
  embedded: 25,
  failed: 0,
  pending: 75,
  total: 100,
  ...overrides,
});

describe('index progress arithmetic', () => {
  it('reports the share of the gallery the index has finished with', () => {
    expect(getIndexPercent(counts())).toBe(25);
  });

  it('reports zero rather than NaN before the backend knows the total', () => {
    expect(getIndexPercent(counts({ embedded: 0, pending: 0, total: 0 }))).toBe(0);
  });

  it('counts images given up on as finished so the bar reaches full', () => {
    // They never drain. Leaving them out parks the bar short of full for a run
    // that has in fact finished.
    expect(getIndexPercent(counts({ embedded: 94, failed: 5, pending: 1, total: 100 }))).toBe(99);
    expect(getIndexPercent(counts({ embedded: 95, failed: 5, pending: 0, total: 100 }))).toBe(100);
  });

  it('agrees with the counts line it is shown beside', () => {
    // "300 of 1,000 images" next to "90%" is not something a user can
    // reconcile; both are read off `pending`, so they cannot diverge.
    const description = describeIndexProgress(counts({ embedded: 300, failed: 600, pending: 100, total: 1000 }));

    expect(description.counts).toBe('900 of 1,000 images');
    expect(description.percent).toBe(90);
    expect(description.skipped).toBe('600 skipped after repeated failures');
  });

  it('holds the announced value below full while anything is outstanding', () => {
    // 99.7% rounds to 100, which tells a screen reader the run is done with
    // 300 images still queued.
    expect(getDisplayPercent(counts({ embedded: 99_700, failed: 0, pending: 300, total: 100_000 }))).toBe(99);
    expect(getDisplayPercent(counts({ embedded: 100_000, failed: 0, pending: 0, total: 100_000 }))).toBe(100);
  });

  it('treats only a non-empty queue as indexing', () => {
    expect(isIndexing(null)).toBe(false);
    expect(isIndexing(counts({ pending: 0 }))).toBe(false);
    expect(isIndexing(counts())).toBe(true);
  });
});

describe('hasProgressed', () => {
  it('is progress when the index has done more work', () => {
    expect(hasProgressed(counts({ embedded: 25 }), counts({ embedded: 26, pending: 74 }))).toBe(true);
    expect(hasProgressed(counts({ failed: 0 }), counts({ failed: 1, pending: 74 }))).toBe(true);
    expect(hasProgressed(null, counts())).toBe(true);
  });

  it('is not progress when only the gallery moved', () => {
    // `total` shifts whenever anyone saves or deletes an image — including the
    // very generation the indexer is waiting out. Counting that as progress
    // would reset the "no progress" clock every time someone generates.
    expect(hasProgressed(counts({ pending: 75, total: 100 }), counts({ pending: 76, total: 101 }))).toBe(false);
    expect(hasProgressed(counts(), counts())).toBe(false);
  });
});

describe('formatDuration', () => {
  it('scales the unit to the magnitude', () => {
    expect(formatDuration(45)).toBe('45s');
    expect(formatDuration(270)).toBe('4m 30s');
    expect(formatDuration(3900)).toBe('1h 05m');
  });

  it('never rounds an elapsed interval down to zero', () => {
    expect(formatDuration(0.2)).toBe('1s');
  });

  it('carries a rounded-up remainder instead of rendering a 60th second', () => {
    // Rounding the parts independently gives "60s", "1m 60s" and "59m 60s".
    expect(formatDuration(59.6)).toBe('1m 00s');
    expect(formatDuration(119.6)).toBe('2m 00s');
    expect(formatDuration(3599.6)).toBe('1h 00m');
  });

  it('renders the unit boundaries themselves', () => {
    expect(formatDuration(59)).toBe('59s');
    expect(formatDuration(60)).toBe('1m 00s');
    expect(formatDuration(3599)).toBe('59m 59s');
    expect(formatDuration(3600)).toBe('1h 00m');
  });

  it('renders nothing for a value that is not a duration', () => {
    expect(formatDuration(Number.NaN)).toBe('');
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('');
    expect(formatDuration(-5)).toBe('');
  });
});

describe('describeIndexProgress', () => {
  it('spells out the counts and the share', () => {
    const description = describeIndexProgress(counts({ embedded: 1204, pending: 3108, total: 4312 }));

    expect(description.counts).toBe(`${(1204).toLocaleString()} of ${(4312).toLocaleString()} images`);
    expect(description.compact).toBe(`${(1204).toLocaleString()}/${(4312).toLocaleString()}`);
    expect(description.percent).toBe(28);
  });

  it('says nothing about staleness while reports are still arriving', () => {
    expect(describeIndexProgress(counts(), STALE_AFTER_MS - 1).stale).toBeNull();
  });

  it('states how long the index has stood still, and claims nothing more', () => {
    const stale = describeIndexProgress(counts(), STALE_AFTER_MS);

    // Not "paused while the queue is busy": the indexer waiting out a
    // generation and the indexer having died look identical from here, and
    // the first is routine, so neither may be asserted.
    expect(stale.stale).toBe('No progress reported for 2m 00s');
  });

  it('keeps the note honest over long waits', () => {
    expect(describeIndexProgress(counts(), 3_600_000).stale).toBe('No progress reported for 1h 00m');
  });

  it('explains images that were given up on', () => {
    expect(describeIndexProgress(counts({ failed: 3 })).skipped).toBe('3 skipped after repeated failures');
  });
});
