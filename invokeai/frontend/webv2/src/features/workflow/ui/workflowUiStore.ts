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

export type WorkflowLibraryTab = 'project' | 'recent' | 'user' | 'default' | 'shared';

/** How This project orders its workflows: the project's own order, or a key. Client side, kept for the session. */
export type ProjectWorkflowSort = 'order' | 'name' | 'updated' | 'lastRun';

export interface ProjectWorkflowOrder {
  direction: 'asc' | 'desc';
  sort: ProjectWorkflowSort;
}

export type WorkflowPublicationIntent =
  | { kind: 'save-as-new'; workflowId: string }
  | { kind: 'update-source'; workflowId: string };

export interface WorkflowUiSnapshot {
  addNodeConnection: AddNodeConnectionFilter | null;
  addNodePosition: XYPosition | null;
  isAddNodeOpen: boolean;
  isLibraryOpen: boolean;
  /** Which place the library dialog shows; remembered for the session. */
  libraryTab: WorkflowLibraryTab;
  /** The library's search text and checked tags. Closing the dialog clears both; a place change clears the tags. */
  libraryQuery: { search: string; tags: readonly string[] };
  /** This project's order; the template places' order is an account preference. */
  libraryProjectOrder: ProjectWorkflowOrder;
  /** Meaningful only for the project and the opening it was made in; null shows the active workflow. */
  librarySelection: { projectId: string; workflowId: string } | null;
  /** Bumped to ask the dialog host to open the JSON file picker. */
  importRequestCount: number;
  renameRequest: { requestId: number; workflowId: string } | null;
  /** Asked for by a shell surface (palette, image menu); consumed by the widget chrome. */
  pendingWorkflowLoad: WorkflowLoadRequest | null;
  publicationIntent: WorkflowPublicationIntent | null;
  /** A template the project already holds copies of; the host asks whether to open, add, or replace a copy. */
  libraryCopyChoice: LibraryCopyChoiceRequest | null;
}

export interface LibraryOpenItem {
  name: string;
  workflow_id: string;
  /** The revision the library list showed; absent when the request named only an id. */
  revision?: number;
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

const EMPTY_LIBRARY_QUERY: WorkflowUiSnapshot['libraryQuery'] = { search: '', tags: [] };

const INITIAL_WORKFLOW_UI_SNAPSHOT: WorkflowUiSnapshot = {
  addNodeConnection: null,
  addNodePosition: null,
  importRequestCount: 0,
  isAddNodeOpen: false,
  renameRequest: null,
  isLibraryOpen: false,
  libraryProjectOrder: { direction: 'asc', sort: 'order' },
  libraryQuery: EMPTY_LIBRARY_QUERY,
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

/**
 * An opening that names no workflow starts This project on the active workflow, so an old selection never aims the
 * rail's actions elsewhere. Closing clears the search and tags; the place stays.
 */
export const setWorkflowLibraryOpen = (isOpen: boolean): void => {
  const wasOpen = workflowUiStore.getSnapshot().isLibraryOpen;

  workflowUiStore.patchSnapshot(
    isOpen
      ? wasOpen
        ? { isLibraryOpen: true }
        : { isLibraryOpen: true, librarySelection: null }
      : { isLibraryOpen: false, libraryQuery: EMPTY_LIBRARY_QUERY }
  );
};

/** Switches place. Tags belong to the place they were checked in, so they clear; the search carries over. */
export const setWorkflowLibraryTab = (libraryTab: WorkflowLibraryTab): void => {
  const { libraryQuery, libraryTab: current } = workflowUiStore.getSnapshot();

  if (current !== libraryTab) {
    workflowUiStore.patchSnapshot({
      libraryQuery: libraryQuery.tags.length > 0 ? { ...libraryQuery, tags: [] } : libraryQuery,
      libraryTab,
    });
  }
};

export const setWorkflowLibrarySearch = (search: string): void => {
  const { libraryQuery } = workflowUiStore.getSnapshot();

  if (libraryQuery.search !== search) {
    workflowUiStore.patchSnapshot({ libraryQuery: { ...libraryQuery, search } });
  }
};

export const setWorkflowLibraryTags = (tags: readonly string[]): void => {
  workflowUiStore.patchSnapshot({ libraryQuery: { ...workflowUiStore.getSnapshot().libraryQuery, tags } });
};

export const setWorkflowLibraryProjectOrder = (libraryProjectOrder: ProjectWorkflowOrder): void => {
  workflowUiStore.patchSnapshot({ libraryProjectOrder });
};

export const setWorkflowLibrarySelection = (selection: { projectId: string; workflowId: string } | null): void => {
  workflowUiStore.patchSnapshot({ librarySelection: selection });
};

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
      item: { name: item.name, revision: item.revision, workflow_id: item.workflow_id },
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
