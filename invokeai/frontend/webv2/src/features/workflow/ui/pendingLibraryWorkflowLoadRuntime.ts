import type { LibraryWorkflowLoadRequest } from './workflowUiStore';

import { clearPendingLibraryWorkflowLoad, workflowUiStore } from './workflowUiStore';

export interface PendingLibraryWorkflowLoadRuntimeDeps {
  clearRequest: (requestId: number) => void;
  getRequest: () => LibraryWorkflowLoadRequest | null;
  load: (workflowId: string) => Promise<void>;
  subscribe: (listener: () => void) => () => void;
}

/**
 * Serial request consumer. A newer token replaces queued work while one load
 * is in flight, and compare-and-clear prevents an old completion from
 * consuming that newer request.
 */
export const startPendingLibraryWorkflowLoadRuntime = (deps: PendingLibraryWorkflowLoadRuntimeDeps): (() => void) => {
  let isRunning = true;
  let inFlight = false;

  const consume = (): void => {
    const request = deps.getRequest();

    if (!isRunning || inFlight || !request) {
      return;
    }

    inFlight = true;
    void deps
      .load(request.workflowId)
      .catch(() => undefined)
      .finally(() => {
        deps.clearRequest(request.requestId);
        inFlight = false;

        if (isRunning) {
          consume();
        }
      });
  };

  const unsubscribe = deps.subscribe(consume);
  consume();

  return () => {
    isRunning = false;
    unsubscribe();
  };
};

export const startWorkflowUiPendingLoadRuntime = (load: (workflowId: string) => Promise<void>): (() => void) =>
  startPendingLibraryWorkflowLoadRuntime({
    clearRequest: clearPendingLibraryWorkflowLoad,
    getRequest: () => workflowUiStore.getSnapshot().pendingLibraryWorkflowLoad,
    load,
    subscribe: workflowUiStore.subscribe,
  });
