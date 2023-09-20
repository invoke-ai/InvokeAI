export const getSecondsFromTimestamps = (start: string, end: string) =>
  Number(((Date.parse(end) - Date.parse(start)) / 1000).toFixed(2));
