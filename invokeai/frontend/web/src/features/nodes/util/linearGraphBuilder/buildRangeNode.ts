import { v4 as uuidv4 } from 'uuid';

import { RootState } from 'app/store/store';
import { RandomRangeInvocation, RangeInvocation } from 'services/api';

export const buildRangeNode = (
  state: RootState
): RangeInvocation | RandomRangeInvocation => {
  const nodeId = uuidv4();
  const { shouldRandomizeSeed, iterations, seed } = state.generation;

  if (shouldRandomizeSeed) {
    return {
      id: nodeId,
      type: 'random_range',
      size: iterations,
    };
  }

  return {
    id: nodeId,
    type: 'range',
    start: seed,
    stop: seed + iterations,
  };
};
