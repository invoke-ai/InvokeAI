import { describe, expect, it } from 'vitest';

import { formatDuration } from './formatDuration';

const after = (seconds: number) => {
  const start = new Date('2026-01-01T00:00:00.000Z');
  return [start.toISOString(), new Date(start.getTime() + seconds * 1000).toISOString()] as const;
};

describe('formatDuration', () => {
  it('formats seconds, minutes and hours', () => {
    expect(formatDuration(...after(12.5))).toBe('12.50s');
    expect(formatDuration(...after(270))).toBe('4m 30s');
    expect(formatDuration(...after(3900))).toBe('1h 5m');
  });

  it('moves to the next unit when rounding reaches it', () => {
    expect(formatDuration(...after(59.996))).toBe('1m 0s');
    expect(formatDuration(...after(119.6))).toBe('2m 0s');
    expect(formatDuration(...after(3599.6))).toBe('1h 0m');
  });

  it('returns null for missing or reversed timestamps', () => {
    expect(formatDuration(null, '2026-01-01T00:00:00.000Z')).toBeNull();
    expect(formatDuration(...after(-1))).toBeNull();
  });
});
