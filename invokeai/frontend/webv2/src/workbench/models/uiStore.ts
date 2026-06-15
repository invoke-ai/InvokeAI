import { createExternalStore } from '@workbench/externalStore';
import { DEFAULT_LIBRARY_FILTERS, type ModelLibraryFilters } from './library';
import type { FoundModel } from './types';

/**
 * Session-lived UI state for the model manager. Widget views unmount whenever
 * the user switches tabs, regions, or center views; keeping selection,
 * filters, and in-progress source forms here means nothing is forgotten when
 * they come back. Deliberately not persisted to localStorage — it resets with
 * the page, like a scroll position.
 */

export type ModelsCenterTab = 'library' | 'add' | 'queue';
export type AddModelsTab = 'starter' | 'url' | 'huggingface' | 'scan' | 'keys';

/** A resolved HuggingFace checkpoint-repo lookup, kept across tab switches. */
export interface HFLookupState {
  repo: string;
  urls: string[];
}

export interface ModelsUiSnapshot {
  centerTab: ModelsCenterTab;
  addTab: AddModelsTab;
  /** Model focused in the center library's detail pane. */
  activeModelKey: string | null;
  /** Model drilled into from the side panel. */
  panelModelKey: string | null;
  selectedKeys: ReadonlySet<string>;
  filters: ModelLibraryFilters;
  /** Last folder-scan query and its results, kept across tab switches. */
  scan: { path: string; results: FoundModel[] } | null;
  /** Last HuggingFace repo lookup, kept across tab switches. */
  hfLookup: HFLookupState | null;
  /** Scroll offsets per library-list instance, restored on remount. */
  libraryScrollOffsets: Record<string, number>;
}

const store = createExternalStore<ModelsUiSnapshot>({
  activeModelKey: null,
  addTab: 'starter',
  centerTab: 'library',
  filters: DEFAULT_LIBRARY_FILTERS,
  hfLookup: null,
  libraryScrollOffsets: {},
  panelModelKey: null,
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
  const { activeModelKey, panelModelKey, selectedKeys } = store.getSnapshot();
  const deleted = new Set(deletedKeys);

  updateModelsUi({
    activeModelKey: activeModelKey !== null && deleted.has(activeModelKey) ? null : activeModelKey,
    panelModelKey: panelModelKey !== null && deleted.has(panelModelKey) ? null : panelModelKey,
    selectedKeys: new Set([...selectedKeys].filter((key) => !deleted.has(key))),
  });
};

/** Jump the center view to a specific model manager tab (used by the panel). */
export const openModelsCenterTab = (centerTab: ModelsCenterTab, addTab?: AddModelsTab): void => {
  updateModelsUi({ centerTab, ...(addTab ? { addTab } : {}) });
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
