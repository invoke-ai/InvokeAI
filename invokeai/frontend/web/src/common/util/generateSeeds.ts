import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { random } from 'lodash-es';

export type GenerateSeedsArg = {
  count: number;
  start?: number;
  min?: number;
  max?: number;
};

export const generateSeeds = ({ count, start, min = NUMPY_RAND_MIN, max = NUMPY_RAND_MAX }: GenerateSeedsArg) => {
  const first = start ?? random(min, max);
  const seeds: number[] = [];
  for (let i = first; i < first + count; i++) {
    seeds.push(i % max);
  }
  return seeds;
};
