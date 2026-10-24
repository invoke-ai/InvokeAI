/** Inject current time so relative captions remain deterministic. */

const DIVISIONS: readonly { amount: number; unit: Intl.RelativeTimeFormatUnit }[] = [
  { amount: 60, unit: 'second' },
  { amount: 60, unit: 'minute' },
  { amount: 24, unit: 'hour' },
  { amount: 7, unit: 'day' },
  { amount: 4.345, unit: 'week' },
  { amount: 12, unit: 'month' },
  { amount: Number.POSITIVE_INFINITY, unit: 'year' },
];

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

/**
 * Caption for a past event (last run, last update), read against a clock snapshot the component takes once per mount:
 * the React Compiler memoizes render output, so reading the clock during render would be stale anyway. A timestamp
 * newer than the snapshot is either server clock skew or an event since mount; both mean "just happened", so it reads
 * as "now" rather than "in X". Use `formatRelativeTime` directly where future times are meaningful.
 */
export const formatTimeSince = (timestamp: string, snapshot: Date): string =>
  formatRelativeTime(timestamp, new Date(Math.max(snapshot.getTime(), Date.parse(timestamp))));
