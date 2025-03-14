import type { PartialDeep } from 'type-fest';

type NumberConstraints = { min: number; max: number; multipleOf?: number };

/**
 * Constrain a number to a range and round to the nearest multiple of a given value.
 * @param v
 * @param constraints
 * @param overrides
 * @returns
 */
export const constrainNumber = (
  v: number,
  constraints: NumberConstraints,
  overrides?: PartialDeep<NumberConstraints>
) => {
  const min = overrides?.min ?? constraints.min;
  const max = overrides?.max ?? constraints.max;
  const multipleOf = overrides?.multipleOf ?? constraints.multipleOf;

  if (multipleOf === undefined) {
    return Math.min(Math.max(v, min), max);
  }

  // First clamp to range
  v = Math.min(Math.max(v, min), max);

  // Round to nearest multiple of multipleOf
  const roundedValue = Math.round(v / multipleOf) * multipleOf;

  // Ensure the result is still within the range
  return Math.min(Math.max(roundedValue, min), max);
};
