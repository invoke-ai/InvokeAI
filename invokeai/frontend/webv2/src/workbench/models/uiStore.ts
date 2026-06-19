import { createExternalStore } from '@workbench/externalStore';

import type { FoundModel } from './types';

import { DEFAULT_LIBRARY_FILTERS, type ModelLibraryFilters } from './library';

/**
 * Session-lived UI state for the model manager. Views unmount whenever the user
 * switches launchpad tabs or detail tabs; keeping selection,
 * filters, and in-progress source forms here means nothing is forgotten when
 * they come back. Deliberately not persisted to localStorage — it resets with
 * the page, like a scroll position.
 */

/**
 * The detail-pane tabs. The library list is now a persistent left column, so it
 * is no longer a tab; the install queue is a persistent detail-pane footer. What
 * remains is the selected model's detail, the unified Add Models search, and API
 * keys.
 */
export type ModelManagerTab = 'details' | 'add' | 'keys';

/** A resolved HuggingFace checkpoint-repo lookup, kept across tab switches. */
export interface HFLookupState {
  repo: string;
  urls: string[];
}

export interface ModelsUiSnapshot {
  activeTab: ModelManagerTab;
  /** Model focused in the manager library's detail pane. */
  activeModelKey: string | null;
  selectedKeys: ReadonlySet<string>;
  filters: ModelLibraryFilters;
  /** Last folder-scan query and its results, kept across tab switches. */
  scan: { path: string; results: FoundModel[] } | null;
  /** Last HuggingFace repo lookup, kept across tab switches. */
  hfLookup: HFLookupState | null;
  /** Whether the always-visible install queue footer is expanded. */
  queueExpanded: boolean;
  /** Scroll offsets per library-list instance, restored on remount. */
  libraryScrollOffsets: Record<string, number>;
}

const store = createExternalStore<ModelsUiSnapshot>({
  activeModelKey: null,
  activeTab: 'add',
  filters: DEFAULT_LIBRARY_FILTERS,
  hfLookup: null,
  libraryScrollOffsets: {},
  queueExpanded: false,
  scan: null,
  selectedKeys: new Set(),
});

export const updateModelsUi = (next: Partial<ModelsUiSnapshot>): void => {
  store.patchSnapshot(next);
};

export const toggleModelSelection = (key: string): void => {
  const selectedKeys = new Set(store.getSnapshot().selectedKeys);

  if (selectedKeys.has(key)) {
    selectedKeys.delete(key);
  } else {
    selectedKeys.add(key);
  }

  updateModelsUi({ selectedKeys });
};

/** Drop deleted models from selection/active slots so stale keys never linger. */
export const pruneModelsUiKeys = (deletedKeys: string[]): void => {
  const { activeModelKey, selectedKeys } = store.getSnapshot();
  const deleted = new Set(deletedKeys);

  updateModelsUi({
    activeModelKey: activeModelKey !== null && deleted.has(activeModelKey) ? null : activeModelKey,
    selectedKeys: new Set([...selectedKeys].filter((key) => !deleted.has(key))),
  });
};

/** Jump the model manager's detail pane to a specific tab. */
export const openModelManagerTab = (activeTab: ModelManagerTab): void => {
  updateModelsUi({ activeTab });
};

/** Focus a model and reveal it in the detail tab (e.g. from a library row). */
export const openModelDetail = (modelKey: string): void => {
  updateModelsUi({ activeModelKey: modelKey, activeTab: 'details' });
};

/** Expand the always-visible install queue footer (e.g. from a "View queue" link). */
export const openInstallQueue = (): void => {
  updateModelsUi({ queueExpanded: true });
};

export const setQueueExpanded = (queueExpanded: boolean): void => {
  updateModelsUi({ queueExpanded });
};

export const saveLibraryScrollOffset = (instanceId: string, offset: number): void => {
  const snapshot = store.getSnapshot();

  // Silent: scroll offsets are read on mount, never subscribed to.
  store.setSnapshotSilently({
    ...snapshot,
    libraryScrollOffsets: { ...snapshot.libraryScrollOffsets, [instanceId]: offset },
  });
};

export const getLibraryScrollOffset = (instanceId: string): number =>
  store.getSnapshot().libraryScrollOffsets[instanceId] ?? 0;

export const useModelsUi = (): ModelsUiSnapshot => store.useSnapshot();
