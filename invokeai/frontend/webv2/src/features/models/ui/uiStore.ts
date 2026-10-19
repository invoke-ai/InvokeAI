import type { FoundModel, ModelTaxonomyType } from '@features/models/core/types';

import { DEFAULT_LIBRARY_FILTERS, type ModelLibraryFilters } from '@features/models/core/library';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Preserve manager selection, filters, and forms across view unmounts for the session; intentionally reset on page
 * reload.
 */

export type ModelManagerTab = 'details' | 'add' | 'keys';

/** A resolved HuggingFace checkpoint-repo lookup, kept across tab switches. */
export interface HFLookupState {
  repo: string;
  urls: string[];
}

export interface ModelsUiSnapshot {
  activeTab: ModelManagerTab;
  /** Consume Add Models search once on mount; null means no seed, distinct from a seeded empty string. */
  addModelsSeed: string | null;
  /** One-shot starter type filter for links identifying a missing model kind rather than a name. */
  addModelsTypeSeed: ModelTaxonomyType | null;
  /** Model focused in the manager library's detail pane. */
  activeModelKey: string | null;
  /**
   * Whether the detail, not the library, is the pane shown when the manager is too narrow for both; null until the
   * user opens or leaves one, when the manager decides from whether the library is empty.
   */
  detailOpen: boolean | null;
  /** Consume provider reveal once so returning to Keys does not replay an old highlight. */
  highlightProviderId: string | null;
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
  /** Whether the expanded queue fills the detail pane instead of docking as a footer. */
  queueMaximized: boolean;
  /** Scroll offsets per library-list instance, restored on remount. */
  libraryScrollOffsets: Record<string, number>;
  /** Compact/full row density per picker id, remembered across opens. */
  pickerCompactViews: Record<string, boolean>;
  /** Base-architecture chips toggled on per picker id, remembered across opens. */
  pickerBaseFilters: Record<string, readonly string[]>;
}

const createInitialModelsUiSnapshot = (): ModelsUiSnapshot => ({
  activeModelKey: null,
  activeTab: 'add',
  addModelsSeed: null,
  addModelsTypeSeed: null,
  detailOpen: null,
  filters: { ...DEFAULT_LIBRARY_FILTERS },
  hfLookup: null,
  highlightProviderId: null,
  libraryScrollOffsets: {},
  pickerBaseFilters: {},
  pickerCompactViews: {},
  queueExpanded: false,
  queueMaximized: false,
  scan: null,
  selectedBundleName: null,
  selectedKeys: new Set(),
});

const store = createExternalStore<ModelsUiSnapshot>(createInitialModelsUiSnapshot());

registerAccountOwnedResource({
  clear: () => store.setSnapshot(createInitialModelsUiSnapshot()),
  name: 'models-ui',
});

/**
 * Every opener reveals the detail and every clear returns to the library, so no caller has to remember the
 * single-pane state: setting a tab or a different model opens the detail; clearing the open model while its Details
 * tab shows closes it. An explicit `detailOpen` wins.
 */
const withDetailOpen = (current: ModelsUiSnapshot, next: Partial<ModelsUiSnapshot>): Partial<ModelsUiSnapshot> => {
  if (next.detailOpen !== undefined) {
    return next;
  }

  if (
    next.activeTab !== undefined ||
    (typeof next.activeModelKey === 'string' && next.activeModelKey !== current.activeModelKey)
  ) {
    return { ...next, detailOpen: true };
  }

  if (next.activeModelKey === null && current.activeModelKey !== null && current.activeTab === 'details') {
    return { ...next, detailOpen: false };
  }

  return next;
};

export const updateModelsUi = (next: Partial<ModelsUiSnapshot>): void => {
  store.patchSnapshot(withDetailOpen(store.getSnapshot(), next));
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

// Revealing a tab must also restore the pane a maximized queue is covering.

/** Jump the model manager's detail pane to a specific tab. */
export const openModelManagerTab = (activeTab: ModelManagerTab): void => {
  updateModelsUi({ activeTab, queueMaximized: false });
};

/** Focus a model and reveal it in the detail tab (e.g. from a library row). */
export const openModelDetail = (modelKey: string): void => {
  updateModelsUi({ activeModelKey: modelKey, activeTab: 'details', queueMaximized: false });
};

/** Return a single-pane manager to its library; side by side nothing changes. */
export const closeModelDetail = (): void => {
  updateModelsUi({ detailOpen: false });
};

/**
 * Decide the starting pane once, when the library first loads: an empty library opens on Add Models, any other on
 * the list. Later installs and deletes never move the user; only openers and Back do.
 */
export const settleInitialModelsPane = (isLibraryEmpty: boolean): void => {
  if (store.getSnapshot().detailOpen === null) {
    updateModelsUi({ detailOpen: isLibraryEmpty });
  }
};

/** Name the requested provider so its possibly offscreen key card can reveal itself. */
export const openExternalProviderKeys = (providerId: string): void => {
  updateModelsUi({ activeTab: 'keys', highlightProviderId: providerId, queueMaximized: false });
};

/** Consumes the pending highlight so it fires once per request. */
export const clearHighlightedProvider = (): void => {
  updateModelsUi({ highlightProviderId: null });
};

/** Clear scan/repo results when opening a bundle because those panels otherwise hide it. */
export const openAddModelsWithBundle = (bundleName: string): void => {
  updateModelsUi({
    activeTab: 'add',
    hfLookup: null,
    queueMaximized: false,
    scan: null,
    selectedBundleName: bundleName,
  });
};

/**
 * Seed Add Models search for external requirement links and clear scan/repo results that would hide the starter
 * catalog.
 */
export const requestAddModelsSearch = (query: string): void => {
  updateModelsUi({
    activeTab: 'add',
    addModelsSeed: query,
    addModelsTypeSeed: null,
    hfLookup: null,
    queueMaximized: false,
    scan: null,
    selectedBundleName: null,
  });
};

/** Seed the starter catalog's type filter for external links identifying a missing model kind. */
export const requestAddModelsTypeFilter = (typeFilter: ModelTaxonomyType): void => {
  updateModelsUi({
    activeTab: 'add',
    addModelsSeed: null,
    addModelsTypeSeed: typeFilter,
    hfLookup: null,
    queueMaximized: false,
    scan: null,
    selectedBundleName: null,
  });
};

/** Reads a pending seed without consuming it — safe to call from a `useState` initializer, which StrictMode double-invokes. */
export const getAddModelsSeed = (): string => store.getSnapshot().addModelsSeed ?? '';

/** Pure read, like {@link getAddModelsSeed}. */
export const getAddModelsTypeSeed = (): ModelTaxonomyType | null => store.getSnapshot().addModelsTypeSeed;

/** Consume seeds silently; subscribers do not read them and the mounting reader already captured them. */
export const clearAddModelsSeeds = (): void => {
  const snapshot = store.getSnapshot();

  if (snapshot.addModelsSeed !== null || snapshot.addModelsTypeSeed !== null) {
    store.setSnapshotSilently({ ...snapshot, addModelsSeed: null, addModelsTypeSeed: null });
  }
};

/** Expand the always-visible install queue footer (e.g. from a "View queue" link). */
export const openInstallQueue = (): void => {
  updateModelsUi({ queueExpanded: true });
};

export const setQueueExpanded = (queueExpanded: boolean): void => {
  updateModelsUi({ queueExpanded });
};

export const setQueueMaximized = (queueMaximized: boolean): void => {
  updateModelsUi({ queueMaximized });
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

/** Base filters are a per-picker preference too; an empty list means "every base". */
export const setPickerBaseFilters = (pickerId: string, bases: readonly string[]): void => {
  const snapshot = store.getSnapshot();

  updateModelsUi({ pickerBaseFilters: { ...snapshot.pickerBaseFilters, [pickerId]: bases } });
};

export const getModelsUiSnapshotForTests = (): ModelsUiSnapshot => store.getSnapshot();

export const useModelsUiSelector = store.useSelector;

export const useModelsUi = (): ModelsUiSnapshot => store.useSnapshot();
