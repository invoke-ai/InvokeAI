import type { FieldType, XYPosition } from '@features/workflow/contracts';

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
}

export const workflowUiStore = createExternalStore<WorkflowUiSnapshot>({
  addNodeConnection: null,
  addNodePosition: null,
  importRequestCount: 0,
  isAddNodeOpen: false,
  isLibraryOpen: false,
  isNewWorkflowConfirmOpen: false,
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

export const requestWorkflowImport = (): void => {
  workflowUiStore.patchSnapshot({ importRequestCount: workflowUiStore.getSnapshot().importRequestCount + 1 });
};
