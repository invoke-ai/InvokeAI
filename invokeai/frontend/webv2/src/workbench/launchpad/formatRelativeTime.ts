/**
 * Compact "Edited 5 minutes ago" timestamps for the Home grid and the Open
 * Project dialog. Expects ISO timestamps (the project library normalizes the
 * backend's SQLite format at its boundary).
 */

const DIVISIONS: { amount: number; unit: Intl.RelativeTimeFormatUnit }[] = [
  { amount: 60, unit: 'second' },
  { amount: 60, unit: 'minute' },
  { amount: 24, unit: 'hour' },
  { amount: 7, unit: 'day' },
  { amount: 4.345, unit: 'week' },
  { amount: 12, unit: 'month' },
  { amount: Number.POSITIVE_INFINITY, unit: 'year' },
];

const formatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

export const formatRelativeTime = (timestamp: string): string => {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return '';
  }

  let duration = (date.getTime() - Date.now()) / 1000;

  for (const division of DIVISIONS) {
    if (Math.abs(duration) < division.amount) {
      return formatter.format(Math.round(duration), division.unit);
    }

    duration /= division.amount;
  }

  return '';
};
