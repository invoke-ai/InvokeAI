export const roundDownToMultiple = (num: number, multiple: number): number => {
  return Math.floor(num / multiple) * multiple;
};

export const roundToMultiple = (num: number, multiple: number): number => {
  return Math.round(num / multiple) * multiple;
};
