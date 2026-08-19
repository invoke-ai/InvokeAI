/**
 * "2 days ago" captions for the library's `last_run_at` stamps. The clock is
 * injected rather than read here, so the caption a test asserts is the caption
 * the panel renders — `Date.now()` lives at the call site only.
 */

const DIVISIONS: readonly { amount: number; unit: Intl.RelativeTimeFormatUnit }[] = [
  { amount: 60, unit: 'second' },
  { amount: 60, unit: 'minute' },
  { amount: 24, unit: 'hour' },
  { amount: 7, unit: 'day' },
  { amount: 4.345, unit: 'week' },
  { amount: 12, unit: 'month' },
  { amount: Number.POSITIVE_INFINITY, unit: 'year' },
];

// `numeric: 'auto'` so the single-unit steps read as "yesterday"/"last week"
// instead of "1 day ago" — matching the Launchpad's project timestamps.
const formatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

/** Empty string for timestamps that cannot be read, so callers drop the caption entirely. */
export const formatRelativeTime = (timestamp: string, now: Date): string => {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return '';
  }

  let duration = (date.getTime() - now.getTime()) / 1000;

  for (const division of DIVISIONS) {
    if (Math.abs(duration) < division.amount) {
      return formatter.format(Math.round(duration), division.unit);
    }

    duration /= division.amount;
  }

  return '';
};
