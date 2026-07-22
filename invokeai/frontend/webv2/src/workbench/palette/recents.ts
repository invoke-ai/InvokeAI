/**
 * Ring buffer of recently executed palette entry ids, persisted per browser
 * (not per project — muscle memory should not reset when switching projects).
 * Stored ids can go stale (commands renamed, widgets closed); readers filter
 * them against the live entry list at render time.
 */

const STORAGE_KEY = 'invokeai:v7:webv2:palette-recents';
const MAX_RECENTS = 20;

const isBrowser = (): boolean => typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';

export const getRecentEntryIds = (): string[] => {
  if (!isBrowser()) {
    return [];
  }

  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '[]') as unknown;

    return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [];
  } catch {
    return [];
  }
};

export const recordRecentEntry = (id: string): void => {
  if (!isBrowser()) {
    return;
  }

  const next = [id, ...getRecentEntryIds().filter((existing) => existing !== id)].slice(0, MAX_RECENTS);

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Quota or private-mode failures are non-fatal; recents are a convenience.
  }
};
