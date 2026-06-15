import { createExternalStore } from '@workbench/externalStore';
import type { FieldType, XYPosition } from '@workbench/workflows/types';

export interface AddNodeConnectionFilter {
  sourceHandle: string;
  sourceNodeId: string;
  sourceType: FieldType;
}

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
  /**
   * The workflow surface currently responsible for rendering the shared
   * dialogs and file input. Both surfaces (center editor, left panel) can be
   * mounted at once; exactly one — the first to mount — hosts them.
   */
  dialogHostId: string | null;
}

export const workflowUiStore = createExternalStore<WorkflowUiSnapshot>({
  addNodeConnection: null,
  addNodePosition: null,
  dialogHostId: null,
  importRequestCount: 0,
  isAddNodeOpen: false,
  isLibraryOpen: false,
  isNewWorkflowConfirmOpen: false,
});

const dialogHostIds: string[] = [];

export const claimWorkflowDialogHost = (hostId: string): void => {
  dialogHostIds.push(hostId);
  workflowUiStore.patchSnapshot({ dialogHostId: dialogHostIds[0] ?? null });
};

export const releaseWorkflowDialogHost = (hostId: string): void => {
  const index = dialogHostIds.indexOf(hostId);

  if (index !== -1) {
    dialogHostIds.splice(index, 1);
  }

  workflowUiStore.patchSnapshot({ dialogHostId: dialogHostIds[0] ?? null });
};

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
