import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { S } from 'services/api/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createWorkflowExecutionCoordinator } from './workflowExecutionCoordinator';

const createDeferredQueueItemRequest = () => {
  let resolve!: (value: S['SessionQueueItem']) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<S['SessionQueueItem']>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return {
    abort: vi.fn(),
    reject,
    resolve,
    unsubscribe: vi.fn(),
    unwrap: () => promise,
  };
};

const buildQueueStatusEvent = (
  overrides: Partial<S['QueueItemStatusChangedEvent']>
): S['QueueItemStatusChangedEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: 'workflows',
    destination: 'gallery',
    user_id: 'user-1',
    status: 'completed',
    status_sequence: 1,
    batch_status: {
      batch_id: 'batch-1',
      queue_id: 'default',
      pending: 0,
      in_progress: 0,
      completed: 1,
      failed: 0,
      canceled: 0,
      total: 1,
    },
    error_type: null,
    error_message: null,
    error_traceback: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    completed_at: '2026-01-01T00:00:00Z',
    ...overrides,
  }) as S['QueueItemStatusChangedEvent'];

const buildInvocationStartedEvent = (
  overrides: Partial<S['InvocationStartedEvent']> = {}
): S['InvocationStartedEvent'] =>
  ({
    queue_id: 'default',
    item_id: 2,
    batch_id: 'batch-2',
    origin: 'workflows',
    destination: 'gallery',
    user_id: 'user-1',
    session_id: 'session-2',
    invocation_source_id: 'node-1',
    invocation: {
      id: 'prepared-node-1',
      type: 'test_node',
    },
    ...overrides,
  }) as S['InvocationStartedEvent'];

const buildInvocationProgressEvent = (
  overrides: Partial<S['InvocationProgressEvent']> = {}
): S['InvocationProgressEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: null,
    destination: null,
    user_id: 'user-1',
    session_id: 'session-1',
    invocation_source_id: 'node-1',
    invocation: {
      id: 'prepared-node-1',
      type: 'test_node',
    },
    message: 'denoising',
    percentage: 0.5,
    image: null,
    ...overrides,
  }) as S['InvocationProgressEvent'];

const buildInvocationCompleteEvent = (
  overrides: Partial<S['InvocationCompleteEvent']> = {}
): S['InvocationCompleteEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: 'workflows',
    destination: 'gallery',
    user_id: 'user-1',
    session_id: 'session-1',
    invocation_source_id: 'node-1',
    invocation: {
      id: 'prepared-node-1',
      type: 'test_node',
    },
    result: {
      type: 'image_output',
      image: { image_name: 'image.png' },
      width: 512,
      height: 512,
    },
    ...overrides,
  }) as S['InvocationCompleteEvent'];

const buildInvocationErrorEvent = (overrides: Partial<S['InvocationErrorEvent']> = {}): S['InvocationErrorEvent'] =>
  ({
    queue_id: 'default',
    item_id: 1,
    batch_id: 'batch-1',
    origin: 'workflows',
    destination: 'gallery',
    user_id: 'user-1',
    session_id: 'session-1',
    invocation_source_id: 'node-1',
    invocation: {
      id: 'prepared-node-1',
      type: 'test_node',
    },
    error_type: 'TestError',
    error_message: 'boom',
    error_traceback: 'traceback',
    ...overrides,
  }) as S['InvocationErrorEvent'];

const buildQueueClearedEvent = (overrides: Partial<S['QueueClearedEvent']> = {}): S['QueueClearedEvent'] =>
  ({
    queue_id: 'default',
    user_id: null,
    ...overrides,
  }) as S['QueueClearedEvent'];

const buildQueueItem = (status: S['SessionQueueItem']['status']): S['SessionQueueItem'] =>
  ({
    item_id: 1,
    queue_id: 'default',
    batch_id: 'batch-1',
    session_id: 'session-1',
    origin: 'workflows',
    destination: 'gallery',
    status,
    priority: 0,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    session: {
      source_prepared_mapping: {
        'node-1': ['prepared-node-1'],
      },
      results: {
        'prepared-node-1': {
          type: 'image_output',
          image: { image_name: 'old-image.png' },
          width: 512,
          height: 512,
        },
      },
    },
  }) as unknown as S['SessionQueueItem'];

const createCoordinatorHarness = (currentUserId: string | null = 'user-1') => {
  const completedInvocationKeysByItemId = new Map<number, Set<string>>();
  const nodeExecutionStates: Record<string, NodeExecutionState | undefined> = {};
  const onInvocationComplete = vi.fn();
  const clearCanvasWorkflowIntegrationProcessing = vi.fn();
  const logReconciliationError = vi.fn();
  const queueItemRequests = new Map<number, ReturnType<typeof createDeferredQueueItemRequest>>();

  const coordinator = createWorkflowExecutionCoordinator({
    clearCanvasWorkflowIntegrationProcessing,
    completedInvocationKeysByItemId,
    getAllNodeExecutionStates: () => nodeExecutionStates,
    getCurrentUserId: () => currentUserId,
    getNodeExecutionState: (nodeId) => nodeExecutionStates[nodeId],
    logReconciliationError,
    onInvocationComplete,
    reconcileQueueItem: (itemId) => {
      const req = createDeferredQueueItemRequest();
      queueItemRequests.set(itemId, req);
      return req;
    },
    setNodeExecutionState: (nodeId, state) => {
      nodeExecutionStates[nodeId] = state;
    },
    upsertNodeExecutionState: (nodeId, state) => {
      nodeExecutionStates[nodeId] = { ...nodeExecutionStates[nodeId], ...state };
    },
  });

  return {
    clearCanvasWorkflowIntegrationProcessing,
    completedInvocationKeysByItemId,
    coordinator,
    nodeExecutionStates,
    onInvocationComplete,
    queueItemRequests,
  };
};

describe(createWorkflowExecutionCoordinator.name, () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('does not let stale reconciliation overwrite a newer in-progress workflow item', async () => {
    const { coordinator, nodeExecutionStates, queueItemRequests } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows' })
    );
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 2, status: 'in_progress', origin: 'workflows' })
    );
    coordinator.onInvocationStarted(buildInvocationStartedEvent({ item_id: 2 }));

    expect(nodeExecutionStates['node-1']?.status).toBe(zNodeStatus.enum.IN_PROGRESS);

    queueItemRequests.get(1)?.resolve(buildQueueItem('completed'));
    await Promise.resolve();
    await Promise.resolve();

    expect(nodeExecutionStates['node-1']?.status).toBe(zNodeStatus.enum.IN_PROGRESS);
    expect(nodeExecutionStates['node-1']?.outputs).toEqual([]);
  });

  it('does not reset node execution states when a workflow-call parent queue item resumes', () => {
    const { completedInvocationKeysByItemId, coordinator, nodeExecutionStates } = createCoordinatorHarness();
    nodeExecutionStates['call-node'] = {
      nodeId: 'call-node',
      status: zNodeStatus.enum.COMPLETED,
      progress: null,
      progressImage: null,
      outputs: [
        {
          type: 'workflow_return_output',
          values: { result: [3] },
        } as unknown as S['InvocationCompleteEvent']['result'],
      ],
      error: null,
    };
    completedInvocationKeysByItemId.set(1, new Set(['prepared-call-node']));

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'in_progress' }));

    expect(nodeExecutionStates['call-node']?.status).toBe(zNodeStatus.enum.COMPLETED);
    expect(nodeExecutionStates['call-node']?.outputs).toHaveLength(1);
  });

  it('does not let a late invocation_complete from an old workflow item overwrite the active workflow item', () => {
    const { coordinator, nodeExecutionStates, onInvocationComplete } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows' })
    );
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 2, status: 'in_progress', origin: 'workflows' })
    );
    coordinator.onInvocationStarted(buildInvocationStartedEvent({ item_id: 2 }));
    coordinator.onInvocationComplete(buildInvocationCompleteEvent({ item_id: 1 }));

    expect(onInvocationComplete).toHaveBeenCalledTimes(1);
    expect(nodeExecutionStates['node-1']?.status).toBe(zNodeStatus.enum.IN_PROGRESS);
    expect(nodeExecutionStates['node-1']?.outputs).toEqual([]);
  });

  it('still runs invocation_complete side effects after a workflow item failed', () => {
    const { coordinator, onInvocationComplete } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'failed', origin: 'workflows' }));
    coordinator.onInvocationComplete(buildInvocationCompleteEvent({ item_id: 1 }));

    expect(onInvocationComplete).toHaveBeenCalledTimes(1);
  });

  it('reconciles completed sibling outputs from failed workflow queue items', async () => {
    const { coordinator, nodeExecutionStates, queueItemRequests } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'failed', origin: 'workflows' }));

    queueItemRequests.get(1)?.resolve(buildQueueItem('failed'));
    await Promise.resolve();
    await Promise.resolve();

    expect(nodeExecutionStates['node-1']?.status).toBe(zNodeStatus.enum.COMPLETED);
    expect(nodeExecutionStates['node-1']?.outputs).toHaveLength(1);
  });

  it('ignores duplicate terminal queue events', () => {
    const { coordinator, queueItemRequests } = createCoordinatorHarness();

    expect(
      coordinator.onQueueItemStatusChanged(
        buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows' })
      )
    ).toBe(true);
    expect(
      coordinator.onQueueItemStatusChanged(
        buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows' })
      )
    ).toBe(false);

    expect(queueItemRequests.size).toBe(1);
  });

  it('rejects trailing invocation progress after a queue item is canceled', () => {
    const { coordinator } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'in_progress', origin: null }));
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 1 }))).toBe(true);

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'canceled', origin: null }));

    // A denoise step callback may race the cancelation and emit progress after the terminal status
    // event. It must not be applied - it would repopulate the progress bar after it was cleared.
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 1 }))).toBe(false);
  });

  it('rejects trailing invocation progress after the queue is cleared', () => {
    const { coordinator, queueItemRequests } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'in_progress', origin: null }));
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 1 }))).toBe(true);

    // Clearing the queue deletes items without emitting per-item terminal status events, so the
    // coordinator is told directly. A workflow item with a pending reconciliation is also tracked
    // to verify the reconciliation is aborted.
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 2, status: 'completed', origin: 'workflows' })
    );
    expect(coordinator.onQueueCleared(buildQueueClearedEvent({ user_id: null }))).toBe(true);

    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 1 }))).toBe(false);
    expect(queueItemRequests.get(2)?.abort).toHaveBeenCalled();
  });

  it('applies a queue clear scoped to the current user', () => {
    const { coordinator, queueItemRequests } = createCoordinatorHarness('user-1');

    coordinator.onQueueItemStatusChanged(buildQueueStatusEvent({ item_id: 1, status: 'in_progress', origin: null }));
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 1 }))).toBe(true);
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 2, status: 'completed', origin: 'workflows' })
    );

    expect(coordinator.onQueueCleared(buildQueueClearedEvent({ user_id: 'user-1' }))).toBe(true);

    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 1 }))).toBe(false);
    expect(queueItemRequests.get(2)?.abort).toHaveBeenCalled();
  });

  it("does not disturb this user's items when another user's scoped clear is broadcast", async () => {
    // In multiuser mode a user-scoped clear only deletes that user's rows, but the queue_cleared
    // event is broadcast to every queue subscriber (sanitized to user_id="redacted" for
    // non-owner, non-admin recipients). This client's items were not touched: its in-flight
    // reconciliation must complete and its in-progress item must keep accepting events.
    const { coordinator, nodeExecutionStates, queueItemRequests } = createCoordinatorHarness('user-b');

    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows', user_id: 'user-b' })
    );
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 2, status: 'in_progress', origin: null, user_id: 'user-b' })
    );
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 2, user_id: 'user-b' }))).toBe(
      true
    );

    expect(coordinator.onQueueCleared(buildQueueClearedEvent({ user_id: 'redacted' }))).toBe(false);

    expect(queueItemRequests.get(1)?.abort).not.toHaveBeenCalled();
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 2, user_id: 'user-b' }))).toBe(
      true
    );

    queueItemRequests.get(1)?.resolve(buildQueueItem('completed'));
    await Promise.resolve();
    await Promise.resolve();

    expect(nodeExecutionStates['node-1']?.status).toBe(zNodeStatus.enum.COMPLETED);
    expect(nodeExecutionStates['node-1']?.outputs).toHaveLength(1);
  });

  it("applies another user's scoped clear to just that user's items on an admin client", async () => {
    // Admins receive every user's events (and the full, non-redacted scoped queue_cleared event),
    // so they may be tracking the cleared user's active item. The clear deletes that user's rows
    // without per-item terminal events: the cleared user's tracked items must be marked terminal
    // so trailing events are rejected, while the admin's own reconciliation and other users' live
    // items are untouched.
    const { coordinator, nodeExecutionStates, queueItemRequests } = createCoordinatorHarness('admin-1');

    // The admin's own completed workflow item, reconciliation in flight.
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows', user_id: 'admin-1' })
    );
    // User A's in-progress item, actively emitting progress on the admin's client.
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 2, status: 'in_progress', origin: null, user_id: 'user-a' })
    );
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 2, user_id: 'user-a' }))).toBe(
      true
    );
    // User C's in-progress item, unaffected by user A's clear.
    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 3, status: 'in_progress', origin: null, user_id: 'user-c' })
    );
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 3, user_id: 'user-c' }))).toBe(
      true
    );

    expect(coordinator.onQueueCleared(buildQueueClearedEvent({ user_id: 'user-a' }))).toBe(true);

    // User A's item is gone: a trailing progress event must be rejected, not repopulate the bar.
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 2, user_id: 'user-a' }))).toBe(
      false
    );
    // User C's item is still live.
    expect(coordinator.onInvocationProgress(buildInvocationProgressEvent({ item_id: 3, user_id: 'user-c' }))).toBe(
      true
    );

    // The admin's own reconciliation was not aborted and completes normally.
    expect(queueItemRequests.get(1)?.abort).not.toHaveBeenCalled();
    queueItemRequests.get(1)?.resolve(buildQueueItem('completed'));
    await Promise.resolve();
    await Promise.resolve();

    expect(nodeExecutionStates['node-1']?.status).toBe(zNodeStatus.enum.COMPLETED);
    expect(nodeExecutionStates['node-1']?.outputs).toHaveLength(1);
  });

  it("aborts the cleared user's in-flight reconciliation on an admin client", () => {
    // The scoped clear deleted the user's rows, so a reconciliation fetch for their terminal item
    // can never succeed and must be aborted.
    const { coordinator, queueItemRequests } = createCoordinatorHarness('admin-1');

    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 1, status: 'completed', origin: 'workflows', user_id: 'user-a' })
    );

    expect(coordinator.onQueueCleared(buildQueueClearedEvent({ user_id: 'user-a' }))).toBe(true);

    expect(queueItemRequests.get(1)?.abort).toHaveBeenCalled();
  });

  it('still clears canvas workflow integration processing on late invocation errors', () => {
    const { clearCanvasWorkflowIntegrationProcessing, coordinator } = createCoordinatorHarness();

    coordinator.onQueueItemStatusChanged(
      buildQueueStatusEvent({ item_id: 1, status: 'canceled', origin: 'workflows' })
    );
    coordinator.onInvocationError(
      buildInvocationErrorEvent({
        item_id: 1,
        origin: 'canvas_workflow_integration',
      })
    );

    expect(clearCanvasWorkflowIntegrationProcessing).toHaveBeenCalledTimes(1);
  });
});
