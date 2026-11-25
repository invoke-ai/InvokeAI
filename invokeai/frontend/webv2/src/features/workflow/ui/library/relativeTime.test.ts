import { describe, expect, it } from 'vitest';

import { formatRelativeTime, formatTimeSince } from './relativeTime';

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;

const NOW = new Date('2026-08-18T12:00:00.000Z');

const ago = (elapsed: number): string => new Date(NOW.getTime() - elapsed).toISOString();
const ahead = (remaining: number): string => new Date(NOW.getTime() + remaining).toISOString();

describe('formatRelativeTime', () => {
  it('reports sub-minute distances in seconds', () => {
    expect(formatRelativeTime(ago(0), NOW)).toBe('now');
    expect(formatRelativeTime(ago(45 * SECOND), NOW)).toBe('45 seconds ago');
  });

  it('climbs through minutes, hours, days, weeks, months and years up to each boundary', () => {
    for (const [elapsed, caption] of [
      [5 * MINUTE, '5 minutes ago'],
      [59 * MINUTE, '59 minutes ago'],
      [3 * HOUR, '3 hours ago'],
      [23 * HOUR, '23 hours ago'],
      // The caption the mock shows verbatim.
      [2 * DAY, '2 days ago'],
      [6 * DAY, '6 days ago'],
      [3 * WEEK, '3 weeks ago'],
      [150 * DAY, '5 months ago'],
      [800 * DAY, '2 years ago'],
    ] as const) {
      expect(formatRelativeTime(ago(elapsed), NOW)).toBe(caption);
    }
  });

  it('uses the natural phrasing for the single-unit steps', () => {
    expect(formatRelativeTime(ago(DAY), NOW)).toBe('yesterday');
    expect(formatRelativeTime(ago(8 * DAY), NOW)).toBe('last week');
  });

  it('handles clock skew from the server without reading as the past', () => {
    expect(formatRelativeTime(ahead(2 * HOUR), NOW)).toBe('in 2 hours');
  });

  it('returns an empty string for values it cannot read, so callers can drop the caption', () => {
    expect(formatRelativeTime('not-a-timestamp', NOW)).toBe('');
    expect(formatRelativeTime('', NOW)).toBe('');
  });
});

describe('formatTimeSince', () => {
  it('measures events before the snapshot from the snapshot', () => {
    expect(formatTimeSince(ago(5 * MINUTE), NOW)).toBe('5 minutes ago');
  });

  it('reads events newer than the snapshot as now, not the future', () => {
    expect(formatTimeSince(ahead(3 * MINUTE), NOW)).toBe('now');
  });

  it('returns an empty string for values it cannot read', () => {
    expect(formatTimeSince('not-a-timestamp', NOW)).toBe('');
  });
});
