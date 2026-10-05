import type { WorkflowLoadRequest, WorkflowLoadSource } from './workflowUiStore';

import { clearPendingWorkflowLoad, workflowUiStore } from './workflowUiStore';

export interface PendingWorkflowLoadRuntimeDeps {
  clearRequest: (requestId: number) => void;
  getRequest: () => WorkflowLoadRequest | null;
  load: (source: WorkflowLoadSource) => Promise<void>;
  subscribe: (listener: () => void) => () => void;
}

/**
 * Serial request consumer. A newer token replaces queued work while one load
 * is in flight, and compare-and-clear prevents an old completion from
 * consuming that newer request.
 */
export const startPendingWorkflowLoadRuntime = (deps: PendingWorkflowLoadRuntimeDeps): (() => void) => {
  let isRunning = true;
  let inFlight = false;

  const consume = (): void => {
    const request = deps.getRequest();

    if (!isRunning || inFlight || !request) {
      return;
    }

    inFlight = true;
    void deps
      .load(request.source)
      .catch(() => undefined)
      .finally(() => {
        if (!isRunning) {
          return;
        }

        deps.clearRequest(request.requestId);
        inFlight = false;
        consume();
      });
  };

  const unsubscribe = deps.subscribe(consume);
  consume();

  return () => {
    isRunning = false;
    unsubscribe();
  };
};

export const startWorkflowUiPendingLoadRuntime = (load: (source: WorkflowLoadSource) => Promise<void>): (() => void) =>
  startPendingWorkflowLoadRuntime({
    clearRequest: clearPendingWorkflowLoad,
    getRequest: () => workflowUiStore.getSnapshot().pendingWorkflowLoad,
    load,
    subscribe: workflowUiStore.subscribe,
  });
