import type { FloatStartStepCountGenerator, IntegerStartStepCountGenerator } from 'features/nodes/types/field';

export const numberStartStepCountGenerator = ({
  start,
  step,
  count,
}: FloatStartStepCountGenerator | IntegerStartStepCountGenerator): number[] => {
  return Array.from({ length: count }, (_, i) => start + i * step);
};
