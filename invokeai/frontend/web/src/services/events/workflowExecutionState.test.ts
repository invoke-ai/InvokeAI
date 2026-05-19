import { describe, expect, it } from 'vitest';

import { createWorkflowExecutionState, transitionWorkflowExecutionState } from './workflowExecutionState';

describe(transitionWorkflowExecutionState.name, () => {
  it('allows invocation completion after the queue item has already completed', () => {
    let state = createWorkflowExecutionState();

    const queueTransition = transitionWorkflowExecutionState(state, {
      type: 'queue_item_status_changed',
      itemId: 1,
      status: 'completed',
    });

    expect(queueTransition.shouldApply).toBe(true);
    state = queueTransition.state;

    const completionTransition = transitionWorkflowExecutionState(state, {
      type: 'invocation_complete',
      itemId: 1,
      invocationId: 'prepared-node-1',
    });

    expect(completionTransition.shouldApply).toBe(true);
    expect(completionTransition.state.invocations['prepared-node-1']).toBe('completed');
  });

  it('ignores late progress and error events after an invocation completed', () => {
    let state = createWorkflowExecutionState();

    const completionTransition = transitionWorkflowExecutionState(state, {
      type: 'invocation_complete',
      itemId: 1,
      invocationId: 'prepared-node-1',
    });

    expect(completionTransition.shouldApply).toBe(true);
    state = completionTransition.state;

    expect(
      transitionWorkflowExecutionState(state, {
        type: 'invocation_progress',
        itemId: 1,
        invocationId: 'prepared-node-1',
      }).shouldApply
    ).toBe(false);

    expect(
      transitionWorkflowExecutionState(state, {
        type: 'invocation_error',
        itemId: 1,
        invocationId: 'prepared-node-1',
      }).shouldApply
    ).toBe(false);
  });

  it('ignores stale non-terminal queue status after a terminal queue status', () => {
    let state = createWorkflowExecutionState();

    const completedTransition = transitionWorkflowExecutionState(state, {
      type: 'queue_item_status_changed',
      itemId: 1,
      status: 'completed',
    });

    expect(completedTransition.shouldApply).toBe(true);
    state = completedTransition.state;

    const staleTransition = transitionWorkflowExecutionState(state, {
      type: 'queue_item_status_changed',
      itemId: 1,
      status: 'in_progress',
    });

    expect(staleTransition.shouldApply).toBe(false);
    expect(staleTransition.state.queueStatus).toBe('completed');
  });

  it('keeps a failed queue item from being overwritten by a late invocation completion', () => {
    let state = createWorkflowExecutionState();

    const failedTransition = transitionWorkflowExecutionState(state, {
      type: 'queue_item_status_changed',
      itemId: 1,
      status: 'failed',
    });

    expect(failedTransition.shouldApply).toBe(true);
    state = failedTransition.state;

    const lateCompletionTransition = transitionWorkflowExecutionState(state, {
      type: 'invocation_complete',
      itemId: 1,
      invocationId: 'prepared-node-1',
    });

    expect(lateCompletionTransition.shouldApply).toBe(false);
    expect(lateCompletionTransition.state.queueStatus).toBe('failed');
  });

  it('treats reconciled completed invocations as terminal', () => {
    let state = createWorkflowExecutionState();

    const reconciliationTransition = transitionWorkflowExecutionState(state, {
      type: 'completed_session_reconciled',
      itemId: 1,
      completedInvocationIds: ['prepared-node-1'],
    });

    expect(reconciliationTransition.shouldApply).toBe(true);
    state = reconciliationTransition.state;
    expect(state.queueStatus).toBe('completed');
    expect(state.invocations['prepared-node-1']).toBe('completed');

    const lateCompletionTransition = transitionWorkflowExecutionState(state, {
      type: 'invocation_complete',
      itemId: 1,
      invocationId: 'prepared-node-1',
    });

    expect(lateCompletionTransition.shouldApply).toBe(false);
  });
});
