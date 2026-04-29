import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { S } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  getResetNodeExecutionStatesOnQueueItemStarted,
  getUpdatedNodeExecutionStateOnInvocationComplete,
  getUpdatedNodeExecutionStateOnInvocationError,
  getUpdatedNodeExecutionStateOnInvocationProgress,
  getUpdatedNodeExecutionStateOnInvocationStarted,
} from './nodeExecutionState';

const buildNodeExecutionState = (overrides: Partial<NodeExecutionState> = {}): NodeExecutionState => ({
  nodeId: 'node-1',
  status: zNodeStatus.enum.PENDING,
  progress: null,
  progressImage: null,
  outputs: [],
  error: null,
  ...overrides,
});

const buildInvocationStartedEvent = (
  overrides: Partial<S['InvocationStartedEvent']> = {}
): S['InvocationStartedEvent'] =>
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
      type: 'add',
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
    origin: 'workflows',
    destination: 'gallery',
    user_id: 'user-1',
    session_id: 'session-1',
    invocation_source_id: 'node-1',
    invocation: {
      id: 'prepared-node-1',
      type: 'add',
    },
    percentage: 0.42,
    image: {
      dataURL: 'data:image/png;base64,abc',
      width: 64,
      height: 64,
    },
    message: 'working',
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
      type: 'add',
    },
    result: {
      type: 'integer_output',
      value: 42,
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
      type: 'add',
    },
    error_type: 'TestError',
    error_message: 'boom',
    error_traceback: 'traceback',
    ...overrides,
  }) as S['InvocationErrorEvent'];

describe(getResetNodeExecutionStatesOnQueueItemStarted.name, () => {
  it('resets node execution states when a queue item starts for the first time', () => {
    const updated = getResetNodeExecutionStatesOnQueueItemStarted(
      {
        'node-1': buildNodeExecutionState({
          status: zNodeStatus.enum.COMPLETED,
          progress: 1,
          outputs: [{ type: 'integer_output', value: 3 } as unknown as S['InvocationCompleteEvent']['result']],
        }),
      },
      1,
      new Map()
    );

    expect(updated?.['node-1']).toEqual({
      nodeId: 'node-1',
      status: zNodeStatus.enum.PENDING,
      progress: null,
      progressImage: null,
      outputs: [],
      error: null,
    });
  });

  it('does not reset node execution states when a workflow-call parent queue item resumes', () => {
    const workflowReturnOutput = {
      type: 'workflow_return_output',
      values: { result: [3] },
    } as unknown as S['InvocationCompleteEvent']['result'];
    const existing = {
      'call-node': buildNodeExecutionState({
        nodeId: 'call-node',
        status: zNodeStatus.enum.COMPLETED,
        outputs: [workflowReturnOutput],
      }),
    };

    const updated = getResetNodeExecutionStatesOnQueueItemStarted(existing, 1, new Map([[1, new Set(['call-node'])]]));

    expect(updated).toBeUndefined();
  });
});

describe(getUpdatedNodeExecutionStateOnInvocationStarted.name, () => {
  it('creates an execution state when started arrives before initialization', () => {
    const event = buildInvocationStartedEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationStarted(undefined, event, new Map<number, Set<string>>());

    expect(updated?.nodeId).toBe(event.invocation_source_id);
    expect(updated?.status).toBe(zNodeStatus.enum.IN_PROGRESS);
    expect(updated?.outputs).toEqual([]);
  });

  it('marks the node in progress on invocation start', () => {
    const updated = getUpdatedNodeExecutionStateOnInvocationStarted(
      buildNodeExecutionState(),
      buildInvocationStartedEvent(),
      new Map<number, Set<string>>()
    );

    expect(updated?.status).toBe(zNodeStatus.enum.IN_PROGRESS);
  });

  it('ignores a late started event after that invocation already completed', () => {
    const event = buildInvocationStartedEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationStarted(
      buildNodeExecutionState({ status: zNodeStatus.enum.COMPLETED, progress: 1 }),
      event,
      new Map([[event.item_id, new Set([event.invocation.id])]])
    );

    expect(updated).toBeUndefined();
  });
});

describe(getUpdatedNodeExecutionStateOnInvocationProgress.name, () => {
  it('creates an execution state when progress arrives before initialization', () => {
    const event = buildInvocationProgressEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationProgress(undefined, event, new Map<number, Set<string>>());

    expect(updated?.nodeId).toBe(event.invocation_source_id);
    expect(updated?.status).toBe(zNodeStatus.enum.IN_PROGRESS);
    expect(updated?.progress).toBe(event.percentage);
    expect(updated?.progressImage).toEqual(event.image);
  });

  it('marks the node in progress and preserves progress updates', () => {
    const event = buildInvocationProgressEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationProgress(
      buildNodeExecutionState(),
      event,
      new Map<number, Set<string>>()
    );

    expect(updated?.status).toBe(zNodeStatus.enum.IN_PROGRESS);
    expect(updated?.progress).toBe(event.percentage);
    expect(updated?.progressImage).toEqual(event.image);
  });

  it('ignores a late progress event after that invocation already completed', () => {
    const event = buildInvocationProgressEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationProgress(
      buildNodeExecutionState({ status: zNodeStatus.enum.COMPLETED, progress: 1 }),
      event,
      new Map([[event.item_id, new Set([event.invocation.id])]])
    );

    expect(updated).toBeUndefined();
  });
});

describe(getUpdatedNodeExecutionStateOnInvocationComplete.name, () => {
  it('creates an execution state when completion arrives before initialization', () => {
    const event = buildInvocationCompleteEvent();
    const completedInvocationKeysByItemId = new Map<number, Set<string>>();
    const updated = getUpdatedNodeExecutionStateOnInvocationComplete(undefined, event, completedInvocationKeysByItemId);

    expect(updated?.nodeId).toBe(event.invocation_source_id);
    expect(updated?.status).toBe(zNodeStatus.enum.COMPLETED);
    expect(updated?.outputs).toEqual([event.result]);
    expect(completedInvocationKeysByItemId).toEqual(new Map([[event.item_id, new Set([event.invocation.id])]]));
  });

  it('records a completed invocation result once', () => {
    const event = buildInvocationCompleteEvent();
    const completedInvocationKeysByItemId = new Map<number, Set<string>>();

    const updated = getUpdatedNodeExecutionStateOnInvocationComplete(
      buildNodeExecutionState({ status: zNodeStatus.enum.IN_PROGRESS, progress: 0.5 }),
      event,
      completedInvocationKeysByItemId
    );

    expect(updated?.status).toBe(zNodeStatus.enum.COMPLETED);
    expect(updated?.progress).toBe(1);
    expect(updated?.outputs).toEqual([event.result]);
    expect(completedInvocationKeysByItemId).toEqual(new Map([[event.item_id, new Set([event.invocation.id])]]));
  });

  it('ignores duplicate completion events for the same invocation', () => {
    const event = buildInvocationCompleteEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationComplete(
      buildNodeExecutionState({ status: zNodeStatus.enum.COMPLETED, progress: 1, outputs: [event.result] }),
      event,
      new Map([[event.item_id, new Set([event.invocation.id])]])
    );

    expect(updated).toBeUndefined();
  });

  it('allows the same prepared invocation id on a different queue item', () => {
    const firstEvent = buildInvocationCompleteEvent({
      item_id: 1,
      result: { type: 'integer_output', value: 1 } as unknown as S['InvocationCompleteEvent']['result'],
    });
    const secondEvent = buildInvocationCompleteEvent({
      item_id: 2,
      result: { type: 'integer_output', value: 2 } as unknown as S['InvocationCompleteEvent']['result'],
    });
    const completedInvocationKeysByItemId = new Map<number, Set<string>>();

    const firstUpdate = getUpdatedNodeExecutionStateOnInvocationComplete(
      buildNodeExecutionState(),
      firstEvent,
      completedInvocationKeysByItemId
    );
    const secondUpdate = getUpdatedNodeExecutionStateOnInvocationComplete(
      firstUpdate,
      secondEvent,
      completedInvocationKeysByItemId
    );

    expect(secondUpdate?.outputs).toEqual([firstEvent.result, secondEvent.result]);
  });
});

describe(getUpdatedNodeExecutionStateOnInvocationError.name, () => {
  it('creates an execution state when error arrives before initialization', () => {
    const event = buildInvocationErrorEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationError(undefined, event);

    expect(updated?.nodeId).toBe(event.invocation_source_id);
    expect(updated?.status).toBe(zNodeStatus.enum.FAILED);
    expect(updated?.progress).toBeNull();
    expect(updated?.progressImage).toBeNull();
    expect(updated?.error).toEqual({
      error_type: event.error_type,
      error_message: event.error_message,
      error_traceback: event.error_traceback,
    });
  });

  it('marks the node failed and records the error', () => {
    const event = buildInvocationErrorEvent();
    const updated = getUpdatedNodeExecutionStateOnInvocationError(
      buildNodeExecutionState({
        status: zNodeStatus.enum.IN_PROGRESS,
        progress: 0.5,
        progressImage: { dataURL: 'data:image/png;base64,abc', width: 64, height: 64 },
      }),
      event
    );

    expect(updated?.status).toBe(zNodeStatus.enum.FAILED);
    expect(updated?.progress).toBeNull();
    expect(updated?.progressImage).toBeNull();
    expect(updated?.error).toEqual({
      error_type: event.error_type,
      error_message: event.error_message,
      error_traceback: event.error_traceback,
    });
  });
});
