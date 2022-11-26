const roundToHundreth = (val: number): number => {
  return Math.round(val * 100) / 100;
};

export default roundToHundreth;
