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

const createCoordinatorHarness = () => {
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
