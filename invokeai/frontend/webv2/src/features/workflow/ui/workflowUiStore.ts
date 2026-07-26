import type { FieldType, XYPosition } from '@features/workflow/contracts';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

export type AddNodeConnectionFilter =
  | {
      kind: 'source';
      sourceHandle: string;
      sourceNodeId: string;
      sourceType: FieldType | null;
    }
  | {
      kind: 'target';
      targetHandle: string;
      targetNodeId: string;
      targetType: FieldType | null;
    };

/**
 * Session-lived UI coordination for the workflow widget. Menu items live
 * inside the shared widget actions menu while their dialogs and the import
 * file input live in the always-mounted header actions — this store is the
 * bridge between the two.
 */

export interface WorkflowUiSnapshot {
  addNodeConnection: AddNodeConnectionFilter | null;
  addNodePosition: XYPosition | null;
  isAddNodeOpen: boolean;
  isLibraryOpen: boolean;
  isNewWorkflowConfirmOpen: boolean;
  /** Bumped to ask the dialog host to open the JSON file picker. */
  importRequestCount: number;
  /** Library workflow a shell surface (command palette) asked to load; consumed by the widget chrome. */
  pendingLibraryWorkflowLoad: LibraryWorkflowLoadRequest | null;
}

export interface LibraryWorkflowLoadRequest {
  requestId: number;
  workflowId: string;
}

let nextLibraryWorkflowLoadRequestId = 0;

const INITIAL_WORKFLOW_UI_SNAPSHOT: WorkflowUiSnapshot = {
  addNodeConnection: null,
  addNodePosition: null,
  importRequestCount: 0,
  isAddNodeOpen: false,
  isLibraryOpen: false,
  isNewWorkflowConfirmOpen: false,
  pendingLibraryWorkflowLoad: null,
};

export const workflowUiStore = createExternalStore<WorkflowUiSnapshot>(INITIAL_WORKFLOW_UI_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => {
    workflowUiStore.setSnapshot(INITIAL_WORKFLOW_UI_SNAPSHOT);
  },
  name: 'workflow-ui',
});

export const setWorkflowLibraryOpen = (isOpen: boolean): void => {
  workflowUiStore.patchSnapshot({ isLibraryOpen: isOpen });
};

export const setAddNodeOpen = (
  isOpen: boolean,
  position: XYPosition | null = null,
  connection: AddNodeConnectionFilter | null = null
): void => {
  workflowUiStore.patchSnapshot({
    addNodeConnection: isOpen ? connection : null,
    addNodePosition: isOpen ? position : null,
    isAddNodeOpen: isOpen,
  });
};

export const setNewWorkflowConfirmOpen = (isOpen: boolean): void => {
  workflowUiStore.patchSnapshot({ isNewWorkflowConfirmOpen: isOpen });
};

export const requestLibraryWorkflowLoad = (workflowId: string): void => {
  nextLibraryWorkflowLoadRequestId += 1;
  workflowUiStore.patchSnapshot({
    pendingLibraryWorkflowLoad: { requestId: nextLibraryWorkflowLoadRequestId, workflowId },
  });
};

export const clearPendingLibraryWorkflowLoad = (requestId: number): void => {
  if (workflowUiStore.getSnapshot().pendingLibraryWorkflowLoad?.requestId === requestId) {
    workflowUiStore.patchSnapshot({ pendingLibraryWorkflowLoad: null });
  }
};

export const requestWorkflowImport = (): void => {
  workflowUiStore.patchSnapshot({ importRequestCount: workflowUiStore.getSnapshot().importRequestCount + 1 });
};
