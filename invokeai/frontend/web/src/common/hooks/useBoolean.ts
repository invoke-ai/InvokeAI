import type { WritableAtom } from 'nanostores';
import { useCallback, useMemo, useState } from 'react';

export const useBoolean = (initialValue: boolean) => {
  const [isTrue, set] = useState(initialValue);
  const setTrue = useCallback(() => set(true), []);
  const setFalse = useCallback(() => set(false), []);
  const toggle = useCallback(() => set((v) => !v), []);

  const api = useMemo(
    () => ({
      isTrue,
      set,
      setTrue,
      setFalse,
      toggle,
    }),
    [isTrue, set, setTrue, setFalse, toggle]
  );

  return api;
};

export const buildUseBoolean = ($boolean: WritableAtom<boolean>) => {
  return () => {
    const setTrue = useCallback(() => {
      $boolean.set(true);
    }, []);
    const setFalse = useCallback(() => {
      $boolean.set(false);
    }, []);
    const set = useCallback((value: boolean) => {
      $boolean.set(value);
    }, []);
    const toggle = useCallback(() => {
      $boolean.set(!$boolean.get());
    }, []);

    const api = useMemo(
      () => ({
        setTrue,
        setFalse,
        set,
        toggle,
        $boolean,
      }),
      [set, setFalse, setTrue, toggle]
    );

    return api;
  };
};
