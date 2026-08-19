import type { FoundModel } from '@features/models/core/types';

import { DEFAULT_LIBRARY_FILTERS, type ModelLibraryFilters } from '@features/models/core/library';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

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
  /**
   * A pending Add Models search, handed over by a surface outside the manager
   * (see {@link requestAddModelsSearch}). Strictly one-shot: the view takes it
   * at mount and {@link clearAddModelsSeed} empties it, so the box keeps its own
   * local state and still resets when the view unmounts. `null` = nothing
   * pending, which is not the same as a seeded empty string.
   */
  addModelsSeed: string | null;
  /** Model focused in the manager library's detail pane. */
  activeModelKey: string | null;
  selectedKeys: ReadonlySet<string>;
  filters: ModelLibraryFilters;
  /** Last folder-scan query and its results, kept across tab switches. */
  scan: { path: string; results: FoundModel[] } | null;
  /** Last HuggingFace repo lookup, kept across tab switches. */
  hfLookup: HFLookupState | null;
  /** Starter bundle selected in Add Models, kept across tab switches. */
  selectedBundleName: string | null;
  /** Whether the always-visible install queue footer is expanded. */
  queueExpanded: boolean;
  /** Scroll offsets per library-list instance, restored on remount. */
  libraryScrollOffsets: Record<string, number>;
  /** Compact/full row density per picker id, remembered across opens. */
  pickerCompactViews: Record<string, boolean>;
}

const createInitialModelsUiSnapshot = (): ModelsUiSnapshot => ({
  activeModelKey: null,
  activeTab: 'add',
  addModelsSeed: null,
  filters: { ...DEFAULT_LIBRARY_FILTERS },
  hfLookup: null,
  libraryScrollOffsets: {},
  pickerCompactViews: {},
  queueExpanded: false,
  scan: null,
  selectedBundleName: null,
  selectedKeys: new Set(),
});

const store = createExternalStore<ModelsUiSnapshot>(createInitialModelsUiSnapshot());

registerAccountOwnedResource({
  clear: () => store.setSnapshot(createInitialModelsUiSnapshot()),
  name: 'models-ui',
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

/**
 * Jump to Add Models with a starter bundle preselected (e.g. from the
 * Launchpad notice). Scan and HF lookup results are cleared because Add
 * Models hides the bundle strip whenever either has results — the user asked
 * to look at the bundle.
 */
export const openAddModelsWithBundle = (bundleName: string): void => {
  updateModelsUi({ activeTab: 'add', hfLookup: null, scan: null, selectedBundleName: bundleName });
};

/**
 * Open Add Models searching for one model. Called just before navigating to the
 * manager from elsewhere in the app (e.g. a model a workflow needs but the
 * account does not have), so the page the user lands on is already offering the
 * thing they clicked — the installed-models list would only confirm its absence.
 *
 * Scan and HuggingFace results are cleared for the same reason
 * {@link openAddModelsWithBundle} clears them: Add Models hides the starter
 * catalog whenever either has results, and the catalog is what this search is
 * for.
 */
export const requestAddModelsSearch = (query: string): void => {
  updateModelsUi({
    activeTab: 'add',
    addModelsSeed: query,
    hfLookup: null,
    scan: null,
    selectedBundleName: null,
  });
};

/** Reads a pending seed without consuming it — safe to call from a `useState` initializer, which StrictMode double-invokes. */
export const getAddModelsSeed = (): string => store.getSnapshot().addModelsSeed ?? '';

/**
 * Consumes the pending seed. Silent because the only reader took it in the same
 * commit and nothing subscribes to it: a notify here would be a store write
 * whose only effect is re-rendering the view that just read it.
 */
export const clearAddModelsSeed = (): void => {
  const snapshot = store.getSnapshot();

  if (snapshot.addModelsSeed !== null) {
    store.setSnapshotSilently({ ...snapshot, addModelsSeed: null });
  }
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

/** Row density is a per-picker preference, so sibling pickers of a kind agree. */
export const setPickerCompactView = (pickerId: string, isCompact: boolean): void => {
  const snapshot = store.getSnapshot();

  updateModelsUi({ pickerCompactViews: { ...snapshot.pickerCompactViews, [pickerId]: isCompact } });
};

export const getModelsUiSnapshotForTests = (): ModelsUiSnapshot => store.getSnapshot();

export const useModelsUiSelector = store.useSelector;

export const useModelsUi = (): ModelsUiSnapshot => store.useSnapshot();
