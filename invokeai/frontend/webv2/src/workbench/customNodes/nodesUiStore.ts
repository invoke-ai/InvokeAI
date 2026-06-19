import { createExternalStore } from '@workbench/externalStore';

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

const store = createExternalStore<NodesUiSnapshot>({
  activeTab: 'details',
  activePackName: null,
  activityExpanded: false,
  addTab: 'git',
  searchTerm: '',
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

export const useNodesUi = (): NodesUiSnapshot => store.useSnapshot();
