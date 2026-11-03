import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Share session selection outside the flow; editor events publish IDs while external requests ask the editor to
 * select and focus.
 */

export interface WorkflowSelectionSnapshot {
  hoveredNodeId: string | null;
  selectedNodeIds: string[];
  /** A pending outside request: the editor selects + zooms to these, then clears. */
  selectionRequest: { nodeIds: string[]; token: number } | null;
}

const INITIAL_WORKFLOW_SELECTION_SNAPSHOT: WorkflowSelectionSnapshot = {
  hoveredNodeId: null,
  selectedNodeIds: [],
  selectionRequest: null,
};

export const workflowSelectionStore = createExternalStore<WorkflowSelectionSnapshot>(
  INITIAL_WORKFLOW_SELECTION_SNAPSHOT
);

registerAccountOwnedResource({
  clear: () => workflowSelectionStore.setSnapshot(INITIAL_WORKFLOW_SELECTION_SNAPSHOT),
  name: 'workflow-selection',
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

/**
 * Selects one node and zooms the editor to it: at once in a mounted editor, else when the next one mounts. Callers
 * show the Workflow editor first; the linear form panel does not apply selection requests.
 */
export const revealWorkflowNode = (nodeId: string): void => requestNodeSelection([nodeId]);

export const clearNodeSelectionRequest = (): void => {
  workflowSelectionStore.patchSnapshot({ selectionRequest: null });
};
