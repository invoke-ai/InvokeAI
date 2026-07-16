import { useSyncExternalStore } from 'react';

let version = 0;
const listeners = new Set<() => void>();

export const notifyMediaCookieRefreshed = () => {
  version += 1;
  listeners.forEach((listener) => listener());
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const getSnapshot = () => version;

export const useMediaCookieRefreshVersion = () => useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
