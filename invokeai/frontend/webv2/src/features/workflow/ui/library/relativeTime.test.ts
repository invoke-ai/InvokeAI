import { describe, expect, it } from 'vitest';

import { formatRelativeTime } from './relativeTime';

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

  it('climbs to minutes up to the hour boundary', () => {
    expect(formatRelativeTime(ago(5 * MINUTE), NOW)).toBe('5 minutes ago');
    expect(formatRelativeTime(ago(59 * MINUTE), NOW)).toBe('59 minutes ago');
  });

  it('climbs to hours up to the day boundary', () => {
    expect(formatRelativeTime(ago(3 * HOUR), NOW)).toBe('3 hours ago');
    expect(formatRelativeTime(ago(23 * HOUR), NOW)).toBe('23 hours ago');
  });

  it('climbs to days up to the week boundary', () => {
    // The caption the mock shows verbatim.
    expect(formatRelativeTime(ago(2 * DAY), NOW)).toBe('2 days ago');
    expect(formatRelativeTime(ago(6 * DAY), NOW)).toBe('6 days ago');
  });

  it('climbs to weeks, months, and years', () => {
    expect(formatRelativeTime(ago(3 * WEEK), NOW)).toBe('3 weeks ago');
    expect(formatRelativeTime(ago(150 * DAY), NOW)).toBe('5 months ago');
    expect(formatRelativeTime(ago(800 * DAY), NOW)).toBe('2 years ago');
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
