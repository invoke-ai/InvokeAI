import { useCallback, useMemo, useRef } from 'react';

export const useTimeoutCallback = (callback: () => void, delay: number, onCancel?: () => void) => {
  const timeoutRef = useRef<number | null>(null);
  const cancel = useCallback(() => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
      onCancel?.();
    }
  }, [onCancel]);
  const callWithTimeout = useCallback(() => {
    cancel();
    timeoutRef.current = window.setTimeout(() => {
      callback();
      timeoutRef.current = null;
    }, delay);
  }, [callback, cancel, delay]);
  const api = useMemo(() => [callWithTimeout, cancel] as const, [callWithTimeout, cancel]);
  return api;
};
