const randomFloat = (min: number, max: number): number => {
  return Math.random() * (max - min + Number.EPSILON) + min;
};

export default randomFloat;
