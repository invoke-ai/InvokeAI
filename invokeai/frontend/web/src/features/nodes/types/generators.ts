import { z } from 'zod';

export const zFloatRangeStartStepCountGenerator = z.object({
  type: z.literal('float-range-generator-start-step-count').default('float-range-generator-start-step-count'),
  start: z.number().default(0),
  step: z.number().default(1),
  count: z.number().int().default(10),
});
export type FloatRangeStartStepCountGenerator = z.infer<typeof zFloatRangeStartStepCountGenerator>;
export const floatRangeStartStepCountGenerator = (generator: FloatRangeStartStepCountGenerator): number[] => {
  const { start, step, count } = generator;
  return Array.from({ length: count }, (_, i) => start + i * step);
};
export const getDefaultFloatRangeStartStepCountGenerator = (): FloatRangeStartStepCountGenerator =>
  zFloatRangeStartStepCountGenerator.parse({});

export const zIntegerRangeStartStepCountGenerator = z.object({
  type: z.literal('integer-range-generator-start-step-count').default('integer-range-generator-start-step-count'),
  start: z.number().int().default(0),
  step: z.number().int().default(1),
  count: z.number().int().default(10),
});
export type IntegerRangeStartStepCountGenerator = z.infer<typeof zIntegerRangeStartStepCountGenerator>;
export const integerRangeStartStepCountGenerator = (generator: IntegerRangeStartStepCountGenerator): number[] => {
  const { start, step, count } = generator;
  return Array.from({ length: count }, (_, i) => start + i * step);
};
export const getDefaultIntegerRangeStartStepCountGenerator = (): IntegerRangeStartStepCountGenerator =>
  zIntegerRangeStartStepCountGenerator.parse({});
