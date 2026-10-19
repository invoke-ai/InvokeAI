import { isSupportedFilterType } from './filterGraphs';

const STORAGE_KEY = 'invokeai:v7:webv2:last-filter-type';

/** Remember the last filter per user for layers without one; unknown or unreadable preferences use the default. */
export const readLastUsedFilterType = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored !== null && isSupportedFilterType(stored) ? stored : null;
  } catch {
    return null;
  }
};

export const recordLastUsedFilterType = (type: string): void => {
  if (typeof window === 'undefined' || !isSupportedFilterType(type)) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, type);
  } catch {
    // Storage failures are non-fatal; this is a convenience.
  }
};
