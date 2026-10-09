import { useEffect, useState } from 'react';

type EqualityFn<Value> = (left: Value, right: Value) => boolean;

/**
 * Follow `live` once it has held still for `delayMs`, so a burst of changes shows only where it settles. A change that
 * `settlesImmediately(previous, next)` accepts is adopted in the same render instead: a new subject, whose value must
 * never show the previous subject's for the length of the delay.
 */
export const useDebouncedValue = <Value>(
  live: Value,
  delayMs: number,
  {
    isEqual = Object.is,
    settlesImmediately,
  }: { isEqual?: EqualityFn<Value>; settlesImmediately?: (previous: Value, next: Value) => boolean } = {}
): Value => {
  const [settled, setSettled] = useState(live);
  const adoptNow = !isEqual(settled, live) && settlesImmediately?.(settled, live) === true;

  // Adjusted during render (React's pattern for state derived from a changing input), so no frame shows the old value.
  if (adoptNow) {
    setSettled(live);
  }

  // A direct effect on purpose: a timer keyed to a changing value is neither a mount registration (useMountEffect) nor
  // an external store, and wrapping it as either would be the effect evasion the webv2 rules forbid.
  useEffect(() => {
    if (isEqual(settled, live)) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setSettled(live);
    }, delayMs);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [delayMs, isEqual, live, settled]);

  return adoptNow ? live : settled;
};
