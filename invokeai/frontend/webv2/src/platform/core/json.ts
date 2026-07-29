const isJsonObject = (value: unknown): value is Record<string, unknown> => {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value) as unknown;
  return prototype === Object.prototype || prototype === null;
};

/**
 * Structural equality for JSON-safe values.
 *
 * Unknown inputs are accepted at integration boundaries. Distinct non-JSON
 * objects are unequal; the same value or object identity still follows
 * `Object.is`.
 */
export const areJsonValuesStructurallyEqual = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) {
    return true;
  }

  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => areJsonValuesStructurallyEqual(value, right[index]))
    );
  }

  if (!isJsonObject(left) || !isJsonObject(right)) {
    return false;
  }

  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);

  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key) => Object.hasOwn(right, key) && areJsonValuesStructurallyEqual(left[key], right[key]))
  );
};
