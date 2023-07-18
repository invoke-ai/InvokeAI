import dateFormat from 'dateformat';

/**
 * Get a `now` timestamp with 1s precision, formatted as ISO datetime.
 */
export const getTimestamp = () =>
  dateFormat(new Date(), `yyyy-mm-dd'T'HH:MM:ss:lo`);
