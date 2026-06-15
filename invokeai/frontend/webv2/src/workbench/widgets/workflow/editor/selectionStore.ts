import { createExternalStore } from '@workbench/externalStore';

/**
 * Session-lived editor selection, shared with surfaces outside the flow (the
 * node inspector, the form builder's zoom-to-node). The editor writes
 * `selectedNodeIds` from its selection events; other surfaces ask for a
 * selection via `requestNodeSelection`, which the editor applies and focuses.
 */

export interface WorkflowSelectionSnapshot {
  hoveredNodeId: string | null;
  selectedNodeIds: string[];
  /** A pending outside request: the editor selects + zooms to these, then clears. */
  selectionRequest: { nodeIds: string[]; token: number } | null;
}

export const workflowSelectionStore = createExternalStore<WorkflowSelectionSnapshot>({
  hoveredNodeId: null,
  selectedNodeIds: [],
  selectionRequest: null,
});

const areSameIds = (a: string[], b: string[]): boolean => a.length === b.length && a.every((id, i) => id === b[i]);

export const reportNodeSelection = (selectedNodeIds: string[]): void => {
  if (!areSameIds(workflowSelectionStore.getSnapshot().selectedNodeIds, selectedNodeIds)) {
    workflowSelectionStore.patchSnapshot({ selectedNodeIds });
  }
};

export const reportNodeHover = (hoveredNodeId: string | null): void => {
  if (workflowSelectionStore.getSnapshot().hoveredNodeId !== hoveredNodeId) {
    workflowSelectionStore.patchSnapshot({ hoveredNodeId });
  }
};

export const requestNodeSelection = (nodeIds: string[]): void => {
  const previousToken = workflowSelectionStore.getSnapshot().selectionRequest?.token ?? 0;

  workflowSelectionStore.patchSnapshot({ selectionRequest: { nodeIds, token: previousToken + 1 } });
};

export const clearNodeSelectionRequest = (): void => {
  workflowSelectionStore.patchSnapshot({ selectionRequest: null });
};
