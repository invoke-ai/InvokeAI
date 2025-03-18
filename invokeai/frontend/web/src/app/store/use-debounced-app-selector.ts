import type { Selector } from '@reduxjs/toolkit';
import { useAppStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import { useEffect, useState } from 'react';

/**
 * A hook that returns a debounced value from the app state.
 *
 * @param selector The redux selector
 * @param debounceMs The debounce time in milliseconds
 * @returns The debounced value
 */
export const useDebouncedAppSelector = <T>(selector: Selector<RootState, T>, debounceMs: number = 300) => {
  const store = useAppStore();
  const [value, setValue] = useState<T>(() => selector(store.getState()));

  useEffect(() => {
    let prevValue = selector(store.getState());
    let timeout: number | null = null;

    const unsubscribe = store.subscribe(() => {
      const value = selector(store.getState());
      if (value !== prevValue) {
        if (timeout !== null) {
          window.clearTimeout(timeout);
        }
        timeout = window.setTimeout(() => {
          setValue(value);
          prevValue = value;
        }, debounceMs);
      }
    });

    return () => {
      unsubscribe();
      if (timeout !== null) {
        window.clearTimeout(timeout);
      }
    };
  }, [debounceMs, selector, store]);

  return value;
};
