import { describe, expect, it } from 'vitest';

import { normalizeServerTimestamp } from './serverTimestamp';

describe('normalizeServerTimestamp', () => {
  it('parses naive SQLite timestamps as UTC', () => {
    expect(normalizeServerTimestamp('2026-06-11 09:21:04.123')).toBe('2026-06-11T09:21:04.123Z');
  });

  it('passes ISO timestamps through unchanged', () => {
    expect(normalizeServerTimestamp('2026-06-11T09:21:04.123Z')).toBe('2026-06-11T09:21:04.123Z');
  });

  it('passes unparseable values through unchanged', () => {
    expect(normalizeServerTimestamp('not a timestamp')).toBe('not a timestamp');
    expect(normalizeServerTimestamp('2026-06-11 not-a-time')).toBe('2026-06-11 not-a-time');
  });
});
