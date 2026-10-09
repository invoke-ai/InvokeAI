import { DEFAULT_NODE_PACK_FILTERS, type NodePackFilters } from '@features/nodes/core/library';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** Keep manager UI state session-lived so tab navigation preserves selection, search, and pending install input. */

export type NodesManagerTab = 'details' | 'add';

export interface NodesUiSnapshot {
  activeTab: NodesManagerTab;
  activePackName: string | null;
  activityExpanded: boolean;
  /**
   * Whether the detail, not the library, is the pane shown when the manager is too narrow for both; null until the
   * user opens or leaves one, when the manager decides from whether the library is empty.
   */
  detailOpen: boolean | null;
  filters: NodePackFilters;
  /** Typed install source; survives the detail tabs unmounting their content. */
  installSource: string;
}

const INITIAL_NODES_UI_SNAPSHOT: NodesUiSnapshot = {
  // Default to Add so empty installs offer an actionable entry point.
  activeTab: 'add',
  activePackName: null,
  activityExpanded: false,
  detailOpen: null,
  filters: { ...DEFAULT_NODE_PACK_FILTERS },
  installSource: '',
};

const store = createExternalStore<NodesUiSnapshot>(INITIAL_NODES_UI_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => store.setSnapshot(INITIAL_NODES_UI_SNAPSHOT),
  name: 'nodes-ui',
});

/**
 * Every opener reveals the detail and every clear returns to the library, so no caller has to remember the
 * single-pane state: setting a tab or a different pack opens the detail; clearing the open pack while its Details
 * tab shows closes it. An explicit `detailOpen` wins.
 */
const withDetailOpen = (current: NodesUiSnapshot, next: Partial<NodesUiSnapshot>): Partial<NodesUiSnapshot> => {
  if (next.detailOpen !== undefined) {
    return next;
  }

  if (
    next.activeTab !== undefined ||
    (typeof next.activePackName === 'string' && next.activePackName !== current.activePackName)
  ) {
    return { ...next, detailOpen: true };
  }

  if (next.activePackName === null && current.activePackName !== null && current.activeTab === 'details') {
    return { ...next, detailOpen: false };
  }

  return next;
};

export const updateNodesUi = (next: Partial<NodesUiSnapshot>): void =>
  store.patchSnapshot(withDetailOpen(store.getSnapshot(), next));

export const openNodePackDetail = (activePackName: string): void => {
  updateNodesUi({ activePackName, activeTab: 'details' });
};

export const openNodesManagerTab = (activeTab: NodesManagerTab): void => {
  updateNodesUi({ activeTab });
};

/** Return a single-pane manager to its library; side by side nothing changes. */
export const closeNodePackDetail = (): void => {
  updateNodesUi({ detailOpen: false });
};

/**
 * Decide the starting pane once, when the library first loads: an empty library opens on Add Nodes, any other on
 * the list. Later installs and uninstalls never move the user; only openers and Back do.
 */
export const settleInitialNodesPane = (isLibraryEmpty: boolean): void => {
  if (store.getSnapshot().detailOpen === null) {
    updateNodesUi({ detailOpen: isLibraryEmpty });
  }
};

export const setNodeActivityExpanded = (activityExpanded: boolean): void => {
  updateNodesUi({ activityExpanded });
};

export const useNodesUiSelector = store.useSelector;

export const useNodesUi = (): NodesUiSnapshot => store.useSnapshot();
