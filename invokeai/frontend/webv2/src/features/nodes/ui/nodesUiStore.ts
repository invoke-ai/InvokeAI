import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Session-lived UI state for the Launchpad nodes manager. Keeping the active
 * detail tab, selected pack, search term, and add-node form tab in an external
 * store (the same pattern as the models UI store) means nothing resets while
 * the user navigates within the manager.
 */

export type NodesManagerTab = 'details' | 'add';
export type AddNodesTab = 'git' | 'scan';

export interface NodesUiSnapshot {
  activeTab: NodesManagerTab;
  addTab: AddNodesTab;
  activePackName: string | null;
  activityExpanded: boolean;
  searchTerm: string;
}

const INITIAL_NODES_UI_SNAPSHOT: NodesUiSnapshot = {
  activeTab: 'details',
  activePackName: null,
  activityExpanded: false,
  addTab: 'git',
  searchTerm: '',
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
