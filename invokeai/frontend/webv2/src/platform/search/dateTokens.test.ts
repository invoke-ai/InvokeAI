import { describe, expect, it } from 'vitest';

import {
  completeTrailingDateToken,
  formatIsoDate,
  isPossibleDatePrefix,
  isTimestampInRange,
  matchTrailingDateToken,
  parseDateTokens,
} from './dateTokens';

// Tuesday 2026-07-21, mid-day local time.
const NOW = new Date(2026, 6, 21, 13, 30, 0);

describe('parseDateTokens', () => {
  it('parses ISO from/to and strips the tokens from text', () => {
    const parse = parseDateTokens('sunset from:2026-07-01 to:2026-07-15 boulevard', NOW);

    expect(parse.range).toEqual({ from: '2026-07-01', to: '2026-07-15' });
    expect(parse.text).toBe('sunset boulevard');
    expect(parse.invalidTokens).toEqual([]);
  });

  it('resolves today and yesterday against the provided now', () => {
    expect(parseDateTokens('from:today', NOW).range).toEqual({ from: '2026-07-21', to: undefined });
    expect(parseDateTokens('from:yesterday', NOW).range).toEqual({ from: '2026-07-20', to: undefined });
  });

  it('resolves Nd and Nw as days ago', () => {
    expect(parseDateTokens('from:7d', NOW).range).toEqual({ from: '2026-07-14', to: undefined });
    expect(parseDateTokens('from:2w', NOW).range).toEqual({ from: '2026-07-07', to: undefined });
  });

  it('resolves Nm as calendar months ago, clamping the day', () => {
    expect(parseDateTokens('from:1m', NOW).range).toEqual({ from: '2026-06-21', to: undefined });
    // Mar 31 − 1m clamps to the end of February.
    expect(parseDateTokens('from:1m', new Date(2026, 2, 31)).range).toEqual({ from: '2026-02-28', to: undefined });
    expect(parseDateTokens('from:1m', new Date(2028, 2, 31)).range).toEqual({ from: '2028-02-29', to: undefined });
  });

  it('applies tokens left-to-right with last write per bound winning', () => {
    expect(parseDateTokens('from:2026-07-01 from:2026-07-05', NOW).range).toEqual({
      from: '2026-07-05',
      to: undefined,
    });
    expect(parseDateTokens('date:2026-07-10 to:2026-07-15', NOW).range).toEqual({
      from: '2026-07-10',
      to: '2026-07-15',
    });
  });

  it('keeps invalid values in text and reports them', () => {
    const parse = parseDateTokens('sunset from:lastweek', NOW);

    expect(parse.range).toBeUndefined();
    expect(parse.text).toBe('sunset from:lastweek');
    expect(parse.invalidTokens).toEqual([{ key: 'from', raw: 'lastweek' }]);
  });

  it('rejects impossible calendar dates', () => {
    const parse = parseDateTokens('from:2026-02-31', NOW);

    expect(parse.range).toBeUndefined();
    expect(parse.invalidTokens).toEqual([{ key: 'from', raw: '2026-02-31' }]);
  });
});

describe('isTimestampInRange', () => {
  it('accepts both space- and T-separated timestamps', () => {
    const range = { from: '2026-07-14', to: '2026-07-21' };

    expect(isTimestampInRange('2026-07-14 00:00:00.000', range)).toBe(true);
    expect(isTimestampInRange('2026-07-21T23:59:59.999', range)).toBe(true);
    expect(isTimestampInRange('2026-07-13 23:59:59.999', range)).toBe(false);
    expect(isTimestampInRange('2026-07-22T00:00:00.000', range)).toBe(false);
  });

  it('fails closed for empty or malformed timestamps', () => {
    expect(isTimestampInRange('', { from: '2026-07-14' })).toBe(false);
    expect(isTimestampInRange('not-a-date', { from: '2026-07-14' })).toBe(false);
  });
});

describe('matchTrailingDateToken / completeTrailingDateToken', () => {
  it('matches an in-progress trailing token with its source index', () => {
    expect(matchTrailingDateToken('sunset from:y')).toEqual({ key: 'from', partialValue: 'y', start: 7 });
    expect(matchTrailingDateToken('date:')).toEqual({ key: 'date', partialValue: '', start: 0 });
  });

  it('completes the trailing token in place with a trailing space', () => {
    expect(completeTrailingDateToken('sunset from:y', 'yesterday')).toBe('sunset from:yesterday ');
    expect(completeTrailingDateToken('from:', '7d')).toBe('from:7d ');
    expect(completeTrailingDateToken('no token here', 'today')).toBe('no token here');
  });
});

describe('isPossibleDatePrefix', () => {
  it('accepts prefixes of keywords, relative values, and ISO dates', () => {
    for (const value of ['', 'y', 'yester', 'today', 't', '7', '7d', '202', '2026', '2026-', '2026-07-', '2026-07-2']) {
      expect(isPossibleDatePrefix(value), value).toBe(true);
    }
  });
});

describe('formatIsoDate', () => {
  it('formats the calendar day itself regardless of timezone offset', () => {
    // Parts-based construction must preserve the written day in negative UTC offsets.
    expect(formatIsoDate('2026-07-21', 'en-US')).toBe('Jul 21');
    expect(formatIsoDate('2026-07-21', 'de-DE')).toBe('21. Juli');
  });
});
