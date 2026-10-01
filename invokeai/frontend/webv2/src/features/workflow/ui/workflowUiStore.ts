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

/** Bridge transient menu actions to always-mounted workflow dialogs and file input through session UI state. */

/** The library dialog's views: the project's own workflows, bundled templates, or the account's templates. */
export type WorkflowLibraryTab = 'project' | 'default' | 'user';

/** What a surface asked the publication host to do with one project workflow. */
export type WorkflowPublicationIntent =
  | { kind: 'save-as-new'; workflowId: string }
  | { kind: 'update-source'; workflowId: string };

export interface WorkflowUiSnapshot {
  addNodeConnection: AddNodeConnectionFilter | null;
  addNodePosition: XYPosition | null;
  isAddNodeOpen: boolean;
  isLibraryOpen: boolean;
  /** Which view the library dialog shows; remembered for the session. */
  libraryTab: WorkflowLibraryTab;
  /** The project workflow selected on the This-project view; only meaningful for the project it was made in. */
  librarySelection: { projectId: string; workflowId: string } | null;
  /** Bumped to ask the dialog host to open the JSON file picker. */
  importRequestCount: number;
  /** A rename the dialog host should offer; it names the workflow it was asked for. */
  renameRequest: { requestId: number; workflowId: string } | null;
  /** A workflow a shell surface (command palette, an image's context menu) asked to load; consumed by the widget chrome. */
  pendingWorkflowLoad: WorkflowLoadRequest | null;
  /** A publication the always-mounted host should start a dialog for. */
  publicationIntent: WorkflowPublicationIntent | null;
  /** A template the project already holds copies of; the host asks whether to open, add, or replace a copy. */
  libraryCopyChoice: LibraryCopyChoiceRequest | null;
}

/** The library template being opened; the name labels the dialog and the undo step. */
export interface LibraryOpenItem {
  name: string;
  workflow_id: string;
}

export interface LibraryCopyChoiceRequest {
  item: LibraryOpenItem;
  /** The project whose copies the choice is about; another active project never sees it. */
  projectId: string;
  requestId: number;
}

let nextLibraryCopyChoiceRequestId = 0;

export type WorkflowLoadSource =
  /** `name` titles the copy question when the project already holds the template. */
  | { kind: 'library'; name?: string; workflowId: string }
  /** An already-fetched workflow document (an image's embedded workflow); `label` names the undo step. */
  | { kind: 'document'; label: string; raw: unknown };

export interface WorkflowLoadRequest {
  requestId: number;
  source: WorkflowLoadSource;
}

let nextWorkflowLoadRequestId = 0;

const INITIAL_WORKFLOW_UI_SNAPSHOT: WorkflowUiSnapshot = {
  addNodeConnection: null,
  addNodePosition: null,
  importRequestCount: 0,
  isAddNodeOpen: false,
  renameRequest: null,
  isLibraryOpen: false,
  librarySelection: null,
  libraryTab: 'project',
  pendingWorkflowLoad: null,
  publicationIntent: null,
  libraryCopyChoice: null,
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

export const setWorkflowLibraryTab = (libraryTab: WorkflowLibraryTab): void => {
  if (workflowUiStore.getSnapshot().libraryTab !== libraryTab) {
    workflowUiStore.patchSnapshot({ libraryTab });
  }
};

export const setWorkflowLibrarySelection = (selection: { projectId: string; workflowId: string } | null): void => {
  workflowUiStore.patchSnapshot({ librarySelection: selection });
};

/** Opens the library on the project's own workflows with one of them selected. */
export const openWorkflowLibraryAtProjectWorkflow = (projectId: string, workflowId: string): void => {
  workflowUiStore.patchSnapshot({
    isLibraryOpen: true,
    librarySelection: { projectId, workflowId },
    libraryTab: 'project',
  });
};

export const requestLibraryCopyChoice = (projectId: string, item: LibraryOpenItem): void => {
  nextLibraryCopyChoiceRequestId += 1;
  workflowUiStore.patchSnapshot({
    libraryCopyChoice: {
      item: { name: item.name, workflow_id: item.workflow_id },
      projectId,
      requestId: nextLibraryCopyChoiceRequestId,
    },
  });
};

export const clearLibraryCopyChoice = (requestId: number): void => {
  if (workflowUiStore.getSnapshot().libraryCopyChoice?.requestId === requestId) {
    workflowUiStore.patchSnapshot({ libraryCopyChoice: null });
  }
};

export const requestWorkflowPublication = (intent: WorkflowPublicationIntent): void => {
  workflowUiStore.patchSnapshot({ publicationIntent: intent });
};

export const clearWorkflowPublicationIntent = (): void => {
  if (workflowUiStore.getSnapshot().publicationIntent !== null) {
    workflowUiStore.patchSnapshot({ publicationIntent: null });
  }
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

const requestWorkflowLoad = (source: WorkflowLoadSource): void => {
  nextWorkflowLoadRequestId += 1;
  workflowUiStore.patchSnapshot({ pendingWorkflowLoad: { requestId: nextWorkflowLoadRequestId, source } });
};

export const requestLibraryWorkflowLoad = (workflowId: string, name?: string): void =>
  requestWorkflowLoad({ kind: 'library', workflowId, ...(name ? { name } : {}) });

export const requestWorkflowDocumentLoad = (raw: unknown, label: string): void =>
  requestWorkflowLoad({ kind: 'document', label, raw });

export const clearPendingWorkflowLoad = (requestId: number): void => {
  if (workflowUiStore.getSnapshot().pendingWorkflowLoad?.requestId === requestId) {
    workflowUiStore.patchSnapshot({ pendingWorkflowLoad: null });
  }
};

let nextRenameRequestId = 0;

export const requestWorkflowRename = (workflowId: string): void => {
  nextRenameRequestId += 1;
  workflowUiStore.patchSnapshot({ renameRequest: { requestId: nextRenameRequestId, workflowId } });
};

export const requestWorkflowImport = (): void => {
  workflowUiStore.patchSnapshot({ importRequestCount: workflowUiStore.getSnapshot().importRequestCount + 1 });
};
