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

export const getMediaUrl = (url: string | undefined, mediaCookieVersion: number): string | undefined => {
  if (!url || mediaCookieVersion === 0) {
    return url;
  }
  const hashIndex = url.indexOf('#');
  const base = hashIndex === -1 ? url : url.slice(0, hashIndex);
  const hash = hashIndex === -1 ? '' : url.slice(hashIndex);
  const separator = base.includes('?') ? '&' : '?';
  return `${base}${separator}media_cookie_version=${mediaCookieVersion}${hash}`;
};

export const useMediaUrl = (url: string | undefined) => getMediaUrl(url, useMediaCookieRefreshVersion());
