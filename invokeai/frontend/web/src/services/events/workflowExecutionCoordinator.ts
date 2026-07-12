import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { LRUCache } from 'lru-cache';
import type { S } from 'services/api/types';
import {
  clearCompletedInvocationKeysForQueueItem,
  markInvocationAsCompleted,
} from 'services/events/invocationTracking';
import {
  getCompletedInvocationIdsFromCompletedSession,
  getNodeExecutionStatesFromCompletedSession,
  getResetNodeExecutionStatesOnQueueItemStarted,
  getUpdatedNodeExecutionStateOnInvocationError,
  getUpdatedNodeExecutionStateOnInvocationProgress,
  getUpdatedNodeExecutionStateOnInvocationStarted,
} from 'services/events/nodeExecutionState';
import {
  createWorkflowExecutionState,
  isTerminalQueueStatus,
  transitionWorkflowExecutionState,
  type WorkflowExecutionState,
} from 'services/events/workflowExecutionState';

type TerminalQueueStatus = Extract<S['SessionQueueItem']['status'], 'completed' | 'failed' | 'canceled'>;

type ReconciliationRequest = {
  abort?: () => void;
  unsubscribe?: () => void;
  unwrap: () => Promise<S['SessionQueueItem']>;
};

type WorkflowExecutionCoordinatorDeps = {
  clearCanvasWorkflowIntegrationProcessing: () => void;
  completedInvocationKeysByItemId: Map<number, Set<string>>;
  getAllNodeExecutionStates: () => Record<string, NodeExecutionState | undefined>;
  getCurrentUserId: () => string | null;
  getNodeExecutionState: (nodeId: string) => NodeExecutionState | undefined;
  logReconciliationError: (error: unknown, itemId: number) => void;
  onInvocationComplete: (data: S['InvocationCompleteEvent']) => void;
  reconcileQueueItem: (itemId: number) => ReconciliationRequest;
  setNodeExecutionState: (nodeId: string, state: NodeExecutionState) => void;
  upsertNodeExecutionState: (nodeId: string, state: NodeExecutionState) => void;
};

export const createWorkflowExecutionCoordinator = (deps: WorkflowExecutionCoordinatorDeps) => {
  const workflowExecutionStates = new LRUCache<number, WorkflowExecutionState>({ max: 100 });
  const pendingWorkflowReconciliationRequests = new Map<number, ReconciliationRequest>();
  let activeWorkflowQueueItemId: number | null = null;

  const transitionWorkflowEvent = (
    itemId: number,
    event: Parameters<typeof transitionWorkflowExecutionState>[1]
  ): boolean => {
    const state = workflowExecutionStates.get(itemId) ?? createWorkflowExecutionState();
    const transition = transitionWorkflowExecutionState(state, event);
    workflowExecutionStates.set(itemId, transition.state);
    return transition.shouldApply;
  };

  const cleanupWorkflowExecutionState = (itemId: number) => {
    const req = pendingWorkflowReconciliationRequests.get(itemId);
    req?.abort?.();
    req?.unsubscribe?.();
    pendingWorkflowReconciliationRequests.delete(itemId);
    // The workflow execution state entry is intentionally kept. A canceled queue item can emit a
    // few trailing invocation events (e.g. a denoise step callback racing the cancelation), and the
    // retained terminal state is what rejects them. Item ids are never reused, and the LRU cache
    // bounds memory.
    clearCompletedInvocationKeysForQueueItem(deps.completedInvocationKeysByItemId, itemId);
  };

  const cancelPendingWorkflowReconciliations = () => {
    for (const req of pendingWorkflowReconciliationRequests.values()) {
      req.abort?.();
      req.unsubscribe?.();
    }
    pendingWorkflowReconciliationRequests.clear();
  };

  const onQueueCleared = (data: S['QueueClearedEvent']): boolean => {
    // A user-scoped clear (multiuser mode) deletes only that user's queue items, but the event is
    // broadcast to every queue subscriber (sanitized to user_id="redacted" for non-owner,
    // non-admin recipients). Another user's clear leaves this client's items untouched, so it must
    // not mark them canceled or abort their pending reconciliations. An unscoped clear
    // (user_id=null — an admin or single-user clear) deletes everything and always applies.
    const clearedUserId = data.user_id ?? null;
    if (clearedUserId !== null && clearedUserId !== deps.getCurrentUserId()) {
      return false;
    }
    // Clearing the queue deletes its items without emitting per-item terminal status events, so the
    // tracked items must be marked terminal here for trailing invocation events to be rejected.
    cancelPendingWorkflowReconciliations();
    for (const itemId of [...workflowExecutionStates.keys()]) {
      const state = workflowExecutionStates.get(itemId);
      if (!state || isTerminalQueueStatus(state.queueStatus)) {
        continue;
      }
      workflowExecutionStates.set(itemId, { ...state, queueStatus: 'canceled' });
    }
    return true;
  };

  const reconcileWorkflowQueueItemResults = (itemId: number, status: TerminalQueueStatus) => {
    const req = deps.reconcileQueueItem(itemId);
    pendingWorkflowReconciliationRequests.set(itemId, req);
    req
      .unwrap()
      .then((queueItem) => {
        if (activeWorkflowQueueItemId !== itemId || queueItem.status !== status) {
          return;
        }

        const completedInvocationIds = getCompletedInvocationIdsFromCompletedSession(queueItem.session);
        transitionWorkflowEvent(itemId, {
          type: 'session_results_reconciled',
          itemId,
          status,
          completedInvocationIds,
        });
        for (const invocationId of completedInvocationIds) {
          markInvocationAsCompleted(deps.completedInvocationKeysByItemId, itemId, invocationId);
        }
        for (const nodeExecutionState of getNodeExecutionStatesFromCompletedSession(queueItem.session)) {
          deps.upsertNodeExecutionState(nodeExecutionState.nodeId, nodeExecutionState);
        }
      })
      .catch((error) => {
        deps.logReconciliationError(error, itemId);
      })
      .finally(() => {
        pendingWorkflowReconciliationRequests.delete(itemId);
        req.unsubscribe?.();
      });
  };

  const onInvocationStarted = (data: S['InvocationStartedEvent']) => {
    if (
      !transitionWorkflowEvent(data.item_id, {
        type: 'invocation_started',
        itemId: data.item_id,
        invocationId: data.invocation.id,
      })
    ) {
      return;
    }

    const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationStarted(
      deps.getNodeExecutionState(data.invocation_source_id),
      data,
      deps.completedInvocationKeysByItemId
    );
    if (updatedNodeExecutionState) {
      deps.upsertNodeExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
    }
  };

  const onInvocationProgress = (data: S['InvocationProgressEvent']) => {
    if (
      !transitionWorkflowEvent(data.item_id, {
        type: 'invocation_progress',
        itemId: data.item_id,
        invocationId: data.invocation.id,
      })
    ) {
      return false;
    }

    if (data.origin === 'workflows') {
      const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationProgress(
        deps.getNodeExecutionState(data.invocation_source_id),
        data,
        deps.completedInvocationKeysByItemId
      );
      if (updatedNodeExecutionState) {
        deps.upsertNodeExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
      }
    }

    return true;
  };

  const onInvocationError = (data: S['InvocationErrorEvent']) => {
    if (
      !transitionWorkflowEvent(data.item_id, {
        type: 'invocation_error',
        itemId: data.item_id,
        invocationId: data.invocation.id,
      })
    ) {
      if (data.origin === 'canvas_workflow_integration') {
        deps.clearCanvasWorkflowIntegrationProcessing();
      }
      return;
    }

    const updatedNodeExecutionState = getUpdatedNodeExecutionStateOnInvocationError(
      deps.getNodeExecutionState(data.invocation_source_id),
      data
    );
    if (updatedNodeExecutionState) {
      deps.upsertNodeExecutionState(updatedNodeExecutionState.nodeId, updatedNodeExecutionState);
    }
    if (data.origin === 'canvas_workflow_integration') {
      deps.clearCanvasWorkflowIntegrationProcessing();
    }
  };

  const onInvocationComplete = (data: S['InvocationCompleteEvent']) => {
    if (
      data.origin === 'workflows' &&
      activeWorkflowQueueItemId !== null &&
      activeWorkflowQueueItemId !== data.item_id
    ) {
      markInvocationAsCompleted(deps.completedInvocationKeysByItemId, data.item_id, data.invocation.id);
      deps.onInvocationComplete(data);
      return;
    }

    transitionWorkflowEvent(data.item_id, {
      type: 'invocation_complete',
      itemId: data.item_id,
      invocationId: data.invocation.id,
    });
    deps.onInvocationComplete(data);
  };

  const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
    if (
      !transitionWorkflowEvent(data.item_id, {
        type: 'queue_item_status_changed',
        itemId: data.item_id,
        status: data.status,
      })
    ) {
      return false;
    }

    if (data.origin === 'workflows') {
      if (activeWorkflowQueueItemId !== null && activeWorkflowQueueItemId !== data.item_id) {
        cleanupWorkflowExecutionState(activeWorkflowQueueItemId);
      }
      activeWorkflowQueueItemId = data.item_id;
    }

    if (data.status === 'in_progress') {
      const nextNodeExecutionStates = getResetNodeExecutionStatesOnQueueItemStarted(
        deps.getAllNodeExecutionStates(),
        data.item_id,
        deps.completedInvocationKeysByItemId
      );
      if (!nextNodeExecutionStates) {
        return true;
      }
      for (const nes of Object.values(nextNodeExecutionStates)) {
        if (!nes) {
          continue;
        }
        deps.setNodeExecutionState(nes.nodeId, nes);
      }
    } else if (data.status === 'completed' || data.status === 'failed' || data.status === 'canceled') {
      if (data.origin === 'workflows') {
        reconcileWorkflowQueueItemResults(data.item_id, data.status);
      } else {
        cleanupWorkflowExecutionState(data.item_id);
      }
    }

    return true;
  };

  return {
    cancelPendingWorkflowReconciliations,
    onInvocationComplete,
    onInvocationError,
    onInvocationProgress,
    onInvocationStarted,
    onQueueCleared,
    onQueueItemStatusChanged,
  };
};
