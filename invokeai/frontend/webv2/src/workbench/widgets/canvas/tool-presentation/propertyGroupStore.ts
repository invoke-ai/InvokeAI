import { createExternalStore } from '@platform/state/externalStore';

const STORAGE_KEY = 'invokeai:v7:webv2:tool-property-collapsed';

/**
 * Per-user collapsed-state overrides for tool property groups, keyed by group
 * id. Only overrides are stored: a group absent here uses its declared
 * default, so new groups can ship collapsed without migration.
 */
const readStored = (): Record<string, boolean> => {
  if (typeof window === 'undefined') {
    return {};
  }
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}') as unknown;
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean')
    );
  } catch {
    return {};
  }
};

const store = createExternalStore<Record<string, boolean>>(readStored());

export const setPropertyGroupCollapsed = (groupId: string, collapsed: boolean): void => {
  const next = { ...store.getSnapshot(), [groupId]: collapsed };
  store.setSnapshot(next);
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Storage failures are non-fatal; collapse is a convenience.
  }
};

export const usePropertyGroupCollapsed = (groupId: string, defaultCollapsed: boolean): boolean =>
  store.useSelector((snapshot) => snapshot[groupId] ?? defaultCollapsed, Object.is);

/** Test seam: clears every override (storage included). */
export const resetPropertyGroupCollapse = (): void => {
  store.setSnapshot({});
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // As above.
  }
};
