import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { random } from 'es-toolkit/compat';

type GenerateSeedsArg = {
  count: number;
  start?: number;
  min?: number;
  max?: number;
};

/**
 * Generate `count` consecutive seeds starting at `start` (or a random seed), wrapping around within the
 * inclusive range [`min`, `max`]. `max` itself is a valid seed, so it is kept and the next seed wraps to `min`.
 */
export const generateSeeds = ({ count, start, min = NUMPY_RAND_MIN, max = NUMPY_RAND_MAX }: GenerateSeedsArg) => {
  const first = start ?? random(min, max);
  const rangeSize = max - min + 1;
  const seeds: number[] = [];
  for (let i = 0; i < count; i++) {
    const offset = (((first - min + i) % rangeSize) + rangeSize) % rangeSize;
    seeds.push(min + offset);
  }
  return seeds;
};
