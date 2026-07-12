import type { AppDispatch, AppGetState } from 'app/store/store';
import type { S } from 'services/api/types';
import { $lastProgressEvent } from 'services/events/stores';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { buildOnInvocationComplete } from './onInvocationComplete';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
    trace: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

// Spies for the workflow-editor node execution side effects. Kept lazy so vi.mock hoisting is safe.
const mockUpsertExecutionState = vi.fn();
let mockNodeExecutionStates: Record<string, unknown> = {};
vi.mock('features/nodes/hooks/useNodeExecutionState', () => ({
  $nodeExecutionStates: { get: () => mockNodeExecutionStates },
  upsertExecutionState: (...args: unknown[]) => mockUpsertExecutionState(...args),
}));

const mockGetUpdatedNodeExecutionState = vi.fn();
vi.mock('services/events/nodeExecutionState', () => ({
  getUpdatedNodeExecutionStateOnInvocationComplete: (...args: unknown[]) => mockGetUpdatedNodeExecutionState(...args),
}));

const buildEvent = (overrides: Partial<S['InvocationCompleteEvent']> = {}): S['InvocationCompleteEvent'] =>
  ({
    invocation: { type: 'some_node', id: 'inv-1' },
    invocation_source_id: 'node-1',
    result: {},
    origin: 'workflows',
    destination: 'canvas',
    user_id: 'owner-1',
    item_id: 1,
    ...overrides,
  }) as unknown as S['InvocationCompleteEvent'];

const makeGetState = (user: { user_id: string } | null): AppGetState =>
  (() => ({ auth: { user } })) as unknown as AppGetState;

const seedProgressEvent = () => {
  const progress = { queue_id: 'default', item_id: 1 } as unknown as S['InvocationProgressEvent'];
  $lastProgressEvent.set(progress);
  return progress;
};

beforeEach(() => {
  mockUpsertExecutionState.mockReset();
  mockGetUpdatedNodeExecutionState.mockReset();
  mockNodeExecutionStates = {};
  $lastProgressEvent.set(null);
});

describe('buildOnInvocationComplete cross-user isolation', () => {
  it("ignores another user's completion event entirely (admin receiving via admin room)", async () => {
    // An admin has a local workflow node with the same source id in progress, and a live progress event.
    mockNodeExecutionStates = { 'node-1': { nodeId: 'node-1', status: 'IN_PROGRESS' } };
    mockGetUpdatedNodeExecutionState.mockReturnValue({ nodeId: 'node-1', status: 'COMPLETED' });
    const seededProgress = seedProgressEvent();

    const dispatch = vi.fn() as unknown as AppDispatch;
    const handler = buildOnInvocationComplete(makeGetState({ user_id: 'admin-1' }), dispatch, new Map());

    // The event belongs to a different user.
    await handler(buildEvent({ user_id: 'other-user', origin: 'canvas_workflow_integration' }));

    // No node execution state read/upsert, no canvas/gallery dispatches, progress event untouched.
    expect(mockGetUpdatedNodeExecutionState).not.toHaveBeenCalled();
    expect(mockUpsertExecutionState).not.toHaveBeenCalled();
    expect(dispatch).not.toHaveBeenCalled();
    expect($lastProgressEvent.get()).toBe(seededProgress);
  });

  it("processes the client's own completion event", async () => {
    mockNodeExecutionStates = { 'node-1': { nodeId: 'node-1', status: 'IN_PROGRESS' } };
    mockGetUpdatedNodeExecutionState.mockReturnValue({ nodeId: 'node-1', status: 'COMPLETED' });
    seedProgressEvent();

    const dispatch = vi.fn() as unknown as AppDispatch;
    const handler = buildOnInvocationComplete(makeGetState({ user_id: 'owner-1' }), dispatch, new Map());

    await handler(buildEvent({ user_id: 'owner-1' }));

    expect(mockUpsertExecutionState).toHaveBeenCalledWith('node-1', expect.objectContaining({ nodeId: 'node-1' }));
    // The handler clears the progress event at the end when it processes an event it owns.
    expect($lastProgressEvent.get()).toBeNull();
  });

  it('processes events in single-user mode where there is no authenticated user', async () => {
    mockNodeExecutionStates = { 'node-1': { nodeId: 'node-1', status: 'IN_PROGRESS' } };
    mockGetUpdatedNodeExecutionState.mockReturnValue({ nodeId: 'node-1', status: 'COMPLETED' });

    const dispatch = vi.fn() as unknown as AppDispatch;
    const handler = buildOnInvocationComplete(makeGetState(null), dispatch, new Map());

    await handler(buildEvent({ user_id: 'system' }));

    expect(mockUpsertExecutionState).toHaveBeenCalledWith('node-1', expect.objectContaining({ nodeId: 'node-1' }));
  });
});
