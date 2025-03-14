import type { PartialDeep } from 'type-fest';

type NumberConstraints = { min: number; max: number; step: number };

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
  const multipleOf = overrides?.step ?? constraints.step;

  if (multipleOf === undefined) {
    return Math.min(Math.max(v, min), max);
  }

  // Round to nearest multiple of multipleOf
  let roundedValue = Math.round(v / multipleOf) * multipleOf;

  // If the value is out of range, find the nearest valid multiple within range
  if (roundedValue < min) {
    roundedValue = Math.ceil(min / multipleOf) * multipleOf;
  } else if (roundedValue > max) {
    roundedValue = Math.floor(max / multipleOf) * multipleOf;
  }

  // Ensure the result is still within the range
  // This handles cases where min or max aren't multiples of step
  return Math.min(Math.max(roundedValue, min), max);
};
