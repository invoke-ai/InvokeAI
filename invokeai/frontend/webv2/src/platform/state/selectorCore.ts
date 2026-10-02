export type EqualityFn<T> = (left: T, right: T) => boolean;

export interface SelectorCache<Snapshot, Selected> {
  read: (snapshot: Snapshot) => Selected;
  setSelector: (selector: (snapshot: Snapshot) => Selected, isEqual: EqualityFn<Selected>) => void;
}

export const areArraysEqual = <Value>(left: readonly Value[], right: readonly Value[]): boolean =>
  left.length === right.length && left.every((value, index) => Object.is(value, right[index]));

export const shallowEqual = <T>(left: T, right: T): boolean => {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return areArraysEqual(left, right);
  }
  if (left instanceof Set && right instanceof Set) {
    return left.size === right.size && Array.from(left).every((value) => right.has(value));
  }
  if (left instanceof Map && right instanceof Map) {
    return (
      left.size === right.size &&
      Array.from(left).every(([key, value]) => right.has(key) && Object.is(right.get(key), value))
    );
  }
  if (typeof left !== 'object' || left === null || typeof right !== 'object' || right === null) {
    return false;
  }

  const leftRecord = left as Record<PropertyKey, unknown>;
  const rightRecord = right as Record<PropertyKey, unknown>;
  const leftKeys = Reflect.ownKeys(leftRecord);
  if (leftKeys.length !== Reflect.ownKeys(rightRecord).length) {
    return false;
  }

  return leftKeys.every(
    (key) => Object.prototype.hasOwnProperty.call(rightRecord, key) && Object.is(leftRecord[key], rightRecord[key])
  );
};

export const createSelectorCache = <Snapshot, Selected>(
  initialSelector: (snapshot: Snapshot) => Selected,
  initialIsEqual: EqualityFn<Selected> = shallowEqual
): SelectorCache<Snapshot, Selected> => {
  let hasSelection = false;
  let isEqual = initialIsEqual;
  let selection: Selected;
  let selector = initialSelector;

  return {
    read: (snapshot) => {
      const next = selector(snapshot);
      if (hasSelection && isEqual(selection, next)) {
        return selection;
      }
      hasSelection = true;
      selection = next;
      return next;
    },
    setSelector: (nextSelector, nextIsEqual) => {
      selector = nextSelector;
      isEqual = nextIsEqual;
    },
  };
};

export const createStableSelector = <Input, Selected>(
  selector: (input: Input) => Selected,
  isEqual: EqualityFn<Selected> = shallowEqual
): ((input: Input) => Selected) => {
  const cache = createSelectorCache(selector, isEqual);
  return (input) => cache.read(input);
};
