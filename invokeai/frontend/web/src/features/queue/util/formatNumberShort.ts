export const formatNumberShort = (num: number) =>
  Intl.NumberFormat('en-US', {
    notation: 'standard',
  }).format(num);
