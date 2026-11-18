import i18next from 'i18next';

/**
 * Compact "Edited 5 minutes ago" timestamps for the Home grid, the Open Project dialog and the notification list, in
 * the active language. Expects ISO timestamps (the project library normalizes the backend's SQLite format at its
 * boundary).
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

const formatters = new Map<string, Intl.RelativeTimeFormat>();

const getFormatter = (locale: string): Intl.RelativeTimeFormat => {
  let formatter = formatters.get(locale);

  if (!formatter) {
    formatter = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
    formatters.set(locale, formatter);
  }

  return formatter;
};

export const formatRelativeTime = (timestamp: string): string => {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return '';
  }

  const formatter = getFormatter(i18next.resolvedLanguage ?? 'en');
  let duration = (date.getTime() - Date.now()) / 1000;

  for (const division of DIVISIONS) {
    if (Math.abs(duration) < division.amount) {
      return formatter.format(Math.round(duration), division.unit);
    }

    duration /= division.amount;
  }

  return '';
};
