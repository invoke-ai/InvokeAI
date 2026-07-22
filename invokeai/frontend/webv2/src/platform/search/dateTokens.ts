/**
 * Closed grammar for date-filter search tokens: `from:`, `to:`, and `date:`
 * accepting ISO `YYYY-MM-DD` or the relative values `today`, `yesterday`, and
 * `Nd`/`Nw`/`Nm` (that many days/weeks/calendar-months ago — a single date,
 * not a span). Shared by the gallery search box and the command palette.
 *
 * Invalid token values are never silently dropped: they are reported in
 * `invalidTokens` AND left in `text`, so plain text search still runs over
 * them (this also keeps prompts that literally contain `from:something`
 * findable). Only valid tokens are stripped from `text`.
 *
 * All dates are calendar days. Relative values resolve against the user's
 * local calendar; the backend compares the resulting date-only strings
 * against UTC timestamps, so day boundaries are effectively UTC — the same
 * semantics as the `by_date:` virtual boards.
 */

export type DateTokenKey = 'from' | 'to' | 'date';

export interface DateRange {
  /** Inclusive lower-bound calendar day, YYYY-MM-DD. */
  from?: string;
  /** Inclusive upper-bound calendar day, YYYY-MM-DD. */
  to?: string;
}

export interface InvalidDateToken {
  key: DateTokenKey;
  /** The unparseable value as typed, without the `key:` prefix. */
  raw: string;
}

export interface DateTokenParse {
  /** Query with recognized valid tokens removed and whitespace collapsed. */
  text: string;
  /** Resolved bounds; present only when at least one valid token resolved. */
  range?: DateRange;
  invalidTokens: readonly InvalidDateToken[];
  /** Any from:/to:/date: token present, valid or not. */
  hasDateTokens: boolean;
}

const TOKEN_PATTERN = /(^|\s)(from|to|date):(\S*)/gi;
const TRAILING_TOKEN_PATTERN = /(^|\s)(from|to|date):(\S*)$/i;
const ISO_DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;
const RELATIVE_PATTERN = /^(\d{1,3})(d|w|m)$/;

const pad = (value: number): string => String(value).padStart(2, '0');

const toIsoDate = (date: Date): string => `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;

/** Days N months before `now`, clamping the day-of-month (Mar 31 − 1m → Feb 28/29). */
const monthsAgo = (now: Date, months: number): Date => {
  const target = new Date(now.getFullYear(), now.getMonth() - months, 1);
  const lastDay = new Date(target.getFullYear(), target.getMonth() + 1, 0).getDate();
  target.setDate(Math.min(now.getDate(), lastDay));
  return target;
};

const resolveDateValue = (value: string, now: Date): string | null => {
  const lower = value.toLowerCase();

  if (lower === 'today') {
    return toIsoDate(now);
  }

  if (lower === 'yesterday') {
    return toIsoDate(new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1));
  }

  const relative = RELATIVE_PATTERN.exec(lower);

  if (relative) {
    const count = Number(relative[1]);

    if (relative[2] === 'm') {
      return toIsoDate(monthsAgo(now, count));
    }

    const days = relative[2] === 'w' ? count * 7 : count;

    return toIsoDate(new Date(now.getFullYear(), now.getMonth(), now.getDate() - days));
  }

  const iso = ISO_DATE_PATTERN.exec(lower);

  if (iso) {
    const year = Number(iso[1]);
    const month = Number(iso[2]);
    const day = Number(iso[3]);
    const candidate = new Date(year, month - 1, day);
    const isRealDate =
      candidate.getFullYear() === year && candidate.getMonth() === month - 1 && candidate.getDate() === day;

    return isRealDate ? lower : null;
  }

  return null;
};

export const parseDateTokens = (query: string, now: Date = new Date()): DateTokenParse => {
  const invalidTokens: InvalidDateToken[] = [];
  let from: string | undefined;
  let to: string | undefined;
  let hasValidToken = false;
  let hasDateTokens = false;

  const text = query
    .replace(TOKEN_PATTERN, (match, boundary: string, rawKey: string, rawValue: string) => {
      hasDateTokens = true;
      const key = rawKey.toLowerCase() as DateTokenKey;
      const resolved = resolveDateValue(rawValue, now);

      if (resolved === null) {
        invalidTokens.push({ key, raw: rawValue });
        return match;
      }

      hasValidToken = true;

      // Tokens apply left-to-right; the last write per bound wins.
      if (key === 'from' || key === 'date') {
        from = resolved;
      }
      if (key === 'to' || key === 'date') {
        to = resolved;
      }

      return boundary;
    })
    .replace(/\s+/g, ' ')
    .trim();

  if (from !== undefined && to !== undefined && from > to) {
    [from, to] = [to, from];
  }

  return {
    hasDateTokens,
    invalidTokens,
    range: hasValidToken ? { from, to } : undefined,
    text,
  };
};

/**
 * True when the timestamp's calendar day falls within the range (bounds
 * inclusive and optional). Works for both space- and T-separated ISO
 * timestamps; empty/absent timestamps fail closed.
 */
export const isTimestampInRange = (isoTimestamp: string, range: DateRange): boolean => {
  const day = isoTimestamp.slice(0, 10);

  if (!ISO_DATE_PATTERN.test(day)) {
    return false;
  }

  return (range.from === undefined || day >= range.from) && (range.to === undefined || day <= range.to);
};

/** The in-progress token at the end of the query, if any, for suggestion UIs. */
export const matchTrailingDateToken = (
  query: string
): { key: DateTokenKey; partialValue: string; start: number } | null => {
  const match = TRAILING_TOKEN_PATTERN.exec(query);

  if (!match || match.index === undefined) {
    return null;
  }

  return {
    key: match[2]?.toLowerCase() as DateTokenKey,
    partialValue: match[3] ?? '',
    start: match.index + (match[1]?.length ?? 0),
  };
};

/** Replaces the trailing token's value with `value` and appends a space. */
export const completeTrailingDateToken = (query: string, value: string): string => {
  const trailing = matchTrailingDateToken(query);

  if (!trailing) {
    return query;
  }

  return `${query.slice(0, trailing.start)}${trailing.key}:${value} `;
};

/**
 * True while `partialValue` could still become a valid value with more
 * typing — used to suppress invalid-token feedback during normal entry.
 */
export const isPossibleDatePrefix = (partialValue: string): boolean => {
  const lower = partialValue.toLowerCase();

  if ('today'.startsWith(lower) || 'yesterday'.startsWith(lower)) {
    return true;
  }

  if (/^\d{1,3}[dwm]?$/.test(lower)) {
    return true;
  }

  // An in-progress ISO date: any prefix of the YYYY-MM-DD shape.
  return /^\d{1,4}(-(\d{0,2}(-\d{0,2})?)?)?$/.test(lower) && lower.length <= 10;
};

/**
 * Formats a YYYY-MM-DD string as a short local date ("Jul 14"). Builds the
 * Date from parts in local time — never via `new Date(isoString)`, which
 * would parse as UTC and shift the day in negative-offset timezones.
 */
export const formatIsoDate = (isoDate: string, locale?: string): string => {
  const parts = ISO_DATE_PATTERN.exec(isoDate);

  if (!parts) {
    return isoDate;
  }

  const local = new Date(Number(parts[1]), Number(parts[2]) - 1, Number(parts[3]));

  return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short' }).format(local);
};

/**
 * Human-readable label for an applied range, e.g. "From Jul 14",
 * "Through Jul 21", "Jul 14 – Jul 21", or "Jul 21" for a single day.
 * English-only; i18n surfaces compose their own labels from formatIsoDate.
 */
export const formatDateRangeLabel = (range: DateRange, locale?: string): string => {
  if (range.from !== undefined && range.to !== undefined) {
    return range.from === range.to
      ? formatIsoDate(range.from, locale)
      : `${formatIsoDate(range.from, locale)} – ${formatIsoDate(range.to, locale)}`;
  }

  if (range.from !== undefined) {
    return `From ${formatIsoDate(range.from, locale)}`;
  }

  if (range.to !== undefined) {
    return `Through ${formatIsoDate(range.to, locale)}`;
  }

  return '';
};
