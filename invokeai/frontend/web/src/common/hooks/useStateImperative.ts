import { useCallback, useEffect, useRef, useState } from 'react';

const isInitialValueFunction = <T>(value: T | (() => T)): value is () => T => {
  return typeof value === 'function';
};

/**
 * Extension of useState that provides imperative access to state value.
 *
 * @param initialValue - Initial state value or function that returns it
 * @returns [state, setState, getState] - Standard state tuple plus getter
 *
 * @remarks
 * - Only use getState() in event handlers and effects, not during rendering
 * - In Concurrent Mode, getState() may return stale values before commit
 */
export const useStateImperative = <T>(
  initialValue: T | (() => T)
): readonly [T, (newValue: T | ((prevState: T) => T)) => void, () => T] => {
  const [state, setState] = useState(isInitialValueFunction(initialValue) ? initialValue() : initialValue);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const getState = useCallback(() => {
    return stateRef.current;
  }, []);

  return [state, setState, getState] as const;
};
