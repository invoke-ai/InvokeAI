/**
 * The backend stamps timestamps via SQLite ("2026-06-11 09:21:04.123") —
 * UTC, but with no timezone marker, which `Date` would misread as local
 * time. Normalize to ISO once, at the boundary that parses the value.
 */
export const normalizeServerTimestamp = (value: string): string => {
  if (!/^\d{4}-\d{2}-\d{2} /.test(value)) {
    return value;
  }

  const date = new Date(`${value.replace(' ', 'T')}Z`);

  return Number.isNaN(date.getTime()) ? value : date.toISOString();
};
