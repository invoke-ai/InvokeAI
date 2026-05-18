import type { S } from 'services/api/types';

type QueueStatus = NonNullable<S['SessionQueueItem']['status']>;
type InvocationStatus = 'in_progress' | 'completed' | 'failed';

type WorkflowExecutionEvent =
  | {
      type: 'queue_item_status_changed';
      itemId: number;
      status: QueueStatus;
    }
  | {
      type: 'completed_session_reconciled';
      itemId: number;
      completedInvocationIds: string[];
    }
  | {
      type: 'invocation_started' | 'invocation_progress' | 'invocation_complete' | 'invocation_error';
      itemId: number;
      invocationId: string;
    };

export type WorkflowExecutionState = {
  itemId: number | null;
  queueStatus: QueueStatus | null;
  invocations: Record<string, InvocationStatus>;
};

type WorkflowExecutionTransition = {
  state: WorkflowExecutionState;
  shouldApply: boolean;
};

const TERMINAL_QUEUE_STATUSES = new Set<QueueStatus>(['completed', 'failed', 'canceled']);
const TERMINAL_INVOCATION_STATUSES = new Set<InvocationStatus>(['completed', 'failed']);

const isTerminalQueueStatus = (status: QueueStatus | null) => status !== null && TERMINAL_QUEUE_STATUSES.has(status);

export const createWorkflowExecutionState = (): WorkflowExecutionState => ({
  itemId: null,
  queueStatus: null,
  invocations: {},
});

export const transitionWorkflowExecutionState = (
  state: WorkflowExecutionState,
  event: WorkflowExecutionEvent
): WorkflowExecutionTransition => {
  const nextState: WorkflowExecutionState = {
    itemId: state.itemId ?? event.itemId,
    queueStatus: state.queueStatus,
    invocations: { ...state.invocations },
  };

  if (event.type === 'queue_item_status_changed') {
    if (isTerminalQueueStatus(state.queueStatus) && !isTerminalQueueStatus(event.status)) {
      return { state, shouldApply: false };
    }

    nextState.queueStatus = event.status;
    return { state: nextState, shouldApply: true };
  }

  if (event.type === 'completed_session_reconciled') {
    if (state.queueStatus === 'failed' || state.queueStatus === 'canceled') {
      return { state, shouldApply: false };
    }

    nextState.queueStatus = 'completed';
    for (const invocationId of event.completedInvocationIds) {
      nextState.invocations[invocationId] = 'completed';
    }
    return { state: nextState, shouldApply: true };
  }

  const invocationStatus = state.invocations[event.invocationId];
  if (invocationStatus && TERMINAL_INVOCATION_STATUSES.has(invocationStatus)) {
    return { state, shouldApply: false };
  }

  if (event.type === 'invocation_started' || event.type === 'invocation_progress') {
    if (isTerminalQueueStatus(state.queueStatus)) {
      return { state, shouldApply: false };
    }
    nextState.invocations[event.invocationId] = 'in_progress';
    return { state: nextState, shouldApply: true };
  }

  if (event.type === 'invocation_error') {
    if (state.queueStatus === 'completed' || state.queueStatus === 'canceled') {
      return { state, shouldApply: false };
    }
    nextState.invocations[event.invocationId] = 'failed';
    return { state: nextState, shouldApply: true };
  }

  if (state.queueStatus === 'failed' || state.queueStatus === 'canceled') {
    return { state, shouldApply: false };
  }
  nextState.invocations[event.invocationId] = 'completed';
  return { state: nextState, shouldApply: true };
};
