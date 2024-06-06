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
