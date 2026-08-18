import { describe, expect, it } from 'vitest';

import {
  appendRateSample,
  describeIndexProgress,
  estimateRate,
  formatDuration,
  getEtaSeconds,
  getIndexPercent,
  isIndexing,
  RATE_WINDOW_MS,
  type ImageIndexCounts,
  type IndexRateSample,
} from './indexProgress';

const counts = (overrides: Partial<ImageIndexCounts> = {}): ImageIndexCounts => ({
  embedded: 25,
  failed: 0,
  pending: 75,
  total: 100,
  ...overrides,
});

describe('index progress arithmetic', () => {
  it('reports the embedded share of the eligible images', () => {
    expect(getIndexPercent(counts())).toBe(25);
  });

  it('reports zero rather than NaN before the backend knows the total', () => {
    expect(getIndexPercent(counts({ embedded: 0, pending: 0, total: 0 }))).toBe(0);
  });

  it('clamps a total that lags behind the embedded count', () => {
    // The counts are two separate queries on the backend, so a deletion landing
    // between them can leave embedded > total for one event.
    expect(getIndexPercent(counts({ embedded: 120, total: 100 }))).toBe(100);
  });

  it('divides the outstanding work by the measured rate', () => {
    expect(getEtaSeconds(counts({ pending: 60 }), 2)).toBe(30);
  });

  it('has no estimate without a rate, and none once the queue is empty', () => {
    expect(getEtaSeconds(counts(), null)).toBeNull();
    expect(getEtaSeconds(counts(), 0)).toBeNull();
    expect(getEtaSeconds(counts({ pending: 0 }), 2)).toBeNull();
  });

  it('treats only a non-empty queue as indexing', () => {
    expect(isIndexing(null)).toBe(false);
    expect(isIndexing(counts({ pending: 0 }))).toBe(false);
    expect(isIndexing(counts())).toBe(true);
  });
});

describe('formatDuration', () => {
  it('scales the unit to the magnitude', () => {
    expect(formatDuration(45)).toBe('45s');
    expect(formatDuration(270)).toBe('4m 30s');
    expect(formatDuration(3900)).toBe('1h 05m');
  });

  it('never rounds an outstanding queue down to zero', () => {
    expect(formatDuration(0.2)).toBe('1s');
  });

  it('renders nothing for a value that is not a duration', () => {
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('');
    expect(formatDuration(-5)).toBe('');
  });
});

describe('rate estimation', () => {
  const samples = (...entries: Array<[number, number]>): IndexRateSample[] =>
    entries.map(([at, embedded]) => ({ at, embedded }));

  it('measures images per second across the retained window', () => {
    expect(estimateRate(samples([0, 0], [1000, 4], [2000, 10]))).toBe(5);
  });

  it('has no estimate from a single sample or a stalled one', () => {
    expect(estimateRate([])).toBeNull();
    expect(estimateRate(samples([1000, 10]))).toBeNull();
    expect(estimateRate(samples([0, 10], [5000, 10]))).toBeNull();
  });

  it('drops samples that have aged out of the window', () => {
    const aged = samples([0, 0], [RATE_WINDOW_MS - 1000, 10]);
    const result = appendRateSample(aged, { at: RATE_WINDOW_MS + 1000, embedded: 20 });

    // The 0ms sample is older than the window; the estimate is the recent
    // 10 images in 2s, not the run-long average of 20 in 61s.
    expect(result).toHaveLength(2);
    expect(estimateRate(result)).toBe(5);
  });

  it('keeps an anchor when events arrive further apart than the window', () => {
    // The worker parks for the length of every generation, so a gap wider than
    // the window is routine. Trimming strictly would leave one sample and no
    // estimate at all for the rest of the run.
    const parked = samples([0, 100]);
    const result = appendRateSample(parked, { at: 5 * RATE_WINDOW_MS, embedded: 400 });

    expect(result).toHaveLength(2);
    expect(estimateRate(result)).toBeCloseTo(1);
  });

  it('restarts the history when the counts stop being comparable', () => {
    // A drop in embedded means the index was reset or images were deleted;
    // measuring across that boundary yields a negative rate.
    const result = appendRateSample(samples([0, 100], [1000, 120]), { at: 2000, embedded: 5 });

    expect(result).toEqual(samples([2000, 5]));
    expect(estimateRate(result)).toBeNull();
  });
});

describe('describeIndexProgress', () => {
  it('spells out the counts, the share and the time remaining', () => {
    const description = describeIndexProgress(counts({ embedded: 1204, pending: 3108, total: 4312 }), 12);

    expect(description.percent).toBeCloseTo(27.92, 2);
    expect(description.counts).toBe(`${(1204).toLocaleString()} of ${(4312).toLocaleString()} images`);
    expect(description.eta).toBe('About 4m 19s remaining');
    expect(description.skipped).toBeNull();
  });

  it('says it is still measuring rather than inventing a time', () => {
    expect(describeIndexProgress(counts(), null).eta).toBe('Estimating time remaining…');
  });

  it('explains images that were given up on', () => {
    expect(describeIndexProgress(counts({ failed: 3 }), null).skipped).toBe('3 skipped after repeated failures');
  });
});
