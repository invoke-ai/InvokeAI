import { DEFAULT_NODE_PACK_FILTERS, type NodePackFilters } from '@features/nodes/core/library';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** Keep manager UI state session-lived so tab navigation preserves selection, search, and pending install input. */

export type NodesManagerTab = 'details' | 'add';

export interface NodesUiSnapshot {
  activeTab: NodesManagerTab;
  activePackName: string | null;
  activityExpanded: boolean;
  filters: NodePackFilters;
  /** Typed install source; survives the detail tabs unmounting their content. */
  installSource: string;
}

const INITIAL_NODES_UI_SNAPSHOT: NodesUiSnapshot = {
  // Default to Add so empty installs offer an actionable entry point.
  activeTab: 'add',
  activePackName: null,
  activityExpanded: false,
  filters: { ...DEFAULT_NODE_PACK_FILTERS },
  installSource: '',
};

const store = createExternalStore<NodesUiSnapshot>(INITIAL_NODES_UI_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => store.setSnapshot(INITIAL_NODES_UI_SNAPSHOT),
  name: 'nodes-ui',
});

export const updateNodesUi = (next: Partial<NodesUiSnapshot>): void => store.patchSnapshot(next);

export const openNodePackDetail = (activePackName: string): void => {
  updateNodesUi({ activePackName, activeTab: 'details' });
};

export const openNodesManagerTab = (activeTab: NodesManagerTab): void => {
  updateNodesUi({ activeTab });
};

export const setNodeActivityExpanded = (activityExpanded: boolean): void => {
  updateNodesUi({ activityExpanded });
};

export const useNodesUiSelector = store.useSelector;

export const useNodesUi = (): NodesUiSnapshot => store.useSnapshot();
