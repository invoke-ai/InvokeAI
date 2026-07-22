import { describe, expect, it } from 'vitest';

import {
  completeTrailingDateToken,
  formatDateRangeLabel,
  formatIsoDate,
  isPossibleDatePrefix,
  isTimestampInRange,
  matchTrailingDateToken,
  parseDateTokens,
} from './dateTokens';

// Tuesday 2026-07-21, mid-day local time.
const NOW = new Date(2026, 6, 21, 13, 30, 0);

describe('parseDateTokens', () => {
  it('returns plain text untouched', () => {
    const parse = parseDateTokens('sunset boulevard', NOW);

    expect(parse).toEqual({
      hasDateTokens: false,
      invalidTokens: [],
      range: undefined,
      text: 'sunset boulevard',
    });
  });

  it('parses ISO from/to and strips the tokens from text', () => {
    const parse = parseDateTokens('sunset from:2026-07-01 to:2026-07-15 boulevard', NOW);

    expect(parse.range).toEqual({ from: '2026-07-01', to: '2026-07-15' });
    expect(parse.text).toBe('sunset boulevard');
    expect(parse.hasDateTokens).toBe(true);
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

  it('is case-insensitive for keys and values', () => {
    const parse = parseDateTokens('FROM:Today TO:YESTERDAY', NOW);

    // from > to, so the bounds swap.
    expect(parse.range).toEqual({ from: '2026-07-20', to: '2026-07-21' });
  });

  it('expands date: into both bounds', () => {
    expect(parseDateTokens('date:2026-07-18', NOW).range).toEqual({ from: '2026-07-18', to: '2026-07-18' });
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

  it('swaps bounds when from is after to', () => {
    expect(parseDateTokens('from:2026-07-15 to:2026-07-01', NOW).range).toEqual({
      from: '2026-07-01',
      to: '2026-07-15',
    });
  });

  it('keeps invalid values in text and reports them', () => {
    const parse = parseDateTokens('sunset from:lastweek', NOW);

    expect(parse.range).toBeUndefined();
    expect(parse.text).toBe('sunset from:lastweek');
    expect(parse.hasDateTokens).toBe(true);
    expect(parse.invalidTokens).toEqual([{ key: 'from', raw: 'lastweek' }]);
  });

  it('rejects impossible calendar dates', () => {
    const parse = parseDateTokens('from:2026-02-31', NOW);

    expect(parse.range).toBeUndefined();
    expect(parse.invalidTokens).toEqual([{ key: 'from', raw: '2026-02-31' }]);
  });

  it('mixes valid and invalid tokens without losing either', () => {
    const parse = parseDateTokens('from:7d to:zzz sunset', NOW);

    expect(parse.range).toEqual({ from: '2026-07-14', to: undefined });
    expect(parse.text).toBe('to:zzz sunset');
    expect(parse.invalidTokens).toEqual([{ key: 'to', raw: 'zzz' }]);
  });

  it('ignores tokens without a leading boundary', () => {
    const parse = parseDateTokens('wherefrom:2026-07-01', NOW);

    expect(parse.hasDateTokens).toBe(false);
    expect(parse.text).toBe('wherefrom:2026-07-01');
  });

  it('reports an empty trailing value as invalid', () => {
    const parse = parseDateTokens('from:', NOW);

    expect(parse.range).toBeUndefined();
    expect(parse.invalidTokens).toEqual([{ key: 'from', raw: '' }]);
  });

  it('allows future dates', () => {
    expect(parseDateTokens('from:2030-01-01', NOW).range).toEqual({ from: '2030-01-01', to: undefined });
  });

  it('collapses leftover whitespace after stripping tokens', () => {
    expect(parseDateTokens('  sunset   from:7d   boulevard  ', NOW).text).toBe('sunset boulevard');
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

  it('treats bounds as optional', () => {
    expect(isTimestampInRange('2026-07-01 10:00:00', { from: '2026-06-01' })).toBe(true);
    expect(isTimestampInRange('2026-07-01 10:00:00', { to: '2026-06-30' })).toBe(false);
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

  it('does not match mid-query or boundary-less tokens', () => {
    expect(matchTrailingDateToken('from:7d sunset')).toBeNull();
    expect(matchTrailingDateToken('wherefrom:7d')).toBeNull();
    expect(matchTrailingDateToken('sunset')).toBeNull();
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

  it('rejects values that can no longer become valid', () => {
    for (const value of ['lastweek', 'yesterdays', '7x', '2026-07-21x', '20260721000']) {
      expect(isPossibleDatePrefix(value), value).toBe(false);
    }
  });
});

describe('formatIsoDate / formatDateRangeLabel', () => {
  it('formats the calendar day itself regardless of timezone offset', () => {
    // A UTC-based Date('2026-07-21') would render Jul 20 in negative offsets;
    // parts-based construction must always yield the written day.
    expect(formatIsoDate('2026-07-21', 'en-US')).toBe('Jul 21');
  });

  it('covers all four range shapes', () => {
    expect(formatDateRangeLabel({ from: '2026-07-14', to: '2026-07-21' }, 'en-US')).toBe('Jul 14 – Jul 21');
    expect(formatDateRangeLabel({ from: '2026-07-14' }, 'en-US')).toBe('From Jul 14');
    expect(formatDateRangeLabel({ to: '2026-07-21' }, 'en-US')).toBe('Through Jul 21');
    expect(formatDateRangeLabel({ from: '2026-07-21', to: '2026-07-21' }, 'en-US')).toBe('Jul 21');
    expect(formatDateRangeLabel({}, 'en-US')).toBe('');
  });
});
