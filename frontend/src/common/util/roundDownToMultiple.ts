export const roundDownToMultiple = (num: number, multiple: number): number => {
  return Math.floor(num / multiple) * multiple;
};
