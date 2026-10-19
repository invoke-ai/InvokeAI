/** SQLite timestamps are UTC without a zone marker; normalize before Date can interpret them as local time. */
export const normalizeServerTimestamp = (value: string): string => {
  if (!/^\d{4}-\d{2}-\d{2} /.test(value)) {
    return value;
  }

  const date = new Date(`${value.replace(' ', 'T')}Z`);

  return Number.isNaN(date.getTime()) ? value : date.toISOString();
};
