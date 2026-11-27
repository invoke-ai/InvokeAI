export const roundDownToMultiple = (num: number, multiple: number): number => {
  return Math.floor(num / multiple) * multiple;
};
export const roundUpToMultiple = (num: number, multiple: number): number => {
  return Math.ceil(num / multiple) * multiple;
};

export const roundToMultiple = (num: number, multiple: number): number => {
  return Math.round(num / multiple) * multiple;
};

export const roundToMultipleMin = (num: number, multiple: number): number => {
  return Math.max(Math.round(num / multiple) * multiple, multiple);
};
