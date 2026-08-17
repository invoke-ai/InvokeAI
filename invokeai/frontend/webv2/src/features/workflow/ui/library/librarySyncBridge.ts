/**
 * Module-level bridge to the single mounted `WorkflowDialogHost`'s library
 * autosaver, mirroring `editor/flowInstanceStore.ts`'s pattern for surfaces
 * that need to reach a widget-scoped instance without threading it through
 * props or importing the chrome component (which would cycle back through
 * this directory). `WorkflowDialogHost` registers its autosaver's
 * `markSynced` in a mount effect; `WorkflowLibraryDialog`'s load path and
 * `useSaveWorkflowToLibrary` call `markLibraryGraphSynced` after a
 * load/save so the freshly-synced content is not immediately queued for
 * another autosave pass.
 */

type MarkSyncedFn = (serialized: Record<string, unknown>) => void;

let markSyncedImpl: MarkSyncedFn | null = null;

export const registerLibraryGraphSyncedHandler = (fn: MarkSyncedFn): void => {
  markSyncedImpl = fn;
};

export const releaseLibraryGraphSyncedHandler = (fn: MarkSyncedFn): void => {
  if (markSyncedImpl === fn) {
    markSyncedImpl = null;
  }
};

export const markLibraryGraphSynced: MarkSyncedFn = (serialized) => markSyncedImpl?.(serialized);
