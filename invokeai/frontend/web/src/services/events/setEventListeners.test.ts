import { $nodeExecutionStates } from 'features/nodes/hooks/useNodeExecutionState';
import { toast } from 'features/toast/toast';
import { LIST_TAG } from 'services/api';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { setEventListeners } from './setEventListeners';
import { $lastProgressEvent } from './stores';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
    trace: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

vi.mock('features/toast/toast', () => ({
  toast: vi.fn(),
}));

// Expose the built handlers and the completed-invocation tracking map that setEventListeners
// hands to both the coordinator and the gallery handler, so tests can assert routing: own events
// reach the own handler via the coordinator, foreign events reach only the foreign handler.
const mockOnInvocationComplete = vi.fn();
const mockOnForeignInvocationComplete = vi.fn();
let capturedCompletedInvocationKeys: Map<number, Set<string>> | null = null;
vi.mock('./onInvocationComplete', () => ({
  buildOnInvocationComplete: (
    _getState: unknown,
    _dispatch: unknown,
    completedInvocationKeysByItemId: Map<number, Set<string>>
  ) => {
    capturedCompletedInvocationKeys = completedInvocationKeysByItemId;
    return mockOnInvocationComplete;
  },
  buildOnForeignInvocationComplete: () => mockOnForeignInvocationComplete,
}));

vi.mock('./onModelInstallError', () => ({
  buildOnModelInstallError: () => vi.fn(),
  DiscordLink: () => null,
  GitHubIssuesLink: () => null,
}));

const createMockSocket = () => {
  const handlers = new Map<string, (...args: Array<unknown>) => void>();

  return {
    on: vi.fn((event: string, handler: (...args: Array<unknown>) => void) => {
      handlers.set(event, handler);
    }),
    emit: vi.fn(),
    trigger: (event: string, payload?: unknown) => {
      const handler = handlers.get(event);
      if (!handler) {
        throw new Error(`No handler registered for ${event}`);
      }
      handler(payload);
    },
  };
};

describe('setEventListeners workflow live updates', () => {
  it('invalidates workflow list caches on workflow_created', () => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    const store = {
      dispatch,
      getState: vi.fn(() => ({})),
    };

    setEventListeners({
      socket: socket as never,
      store: store as never,
      setIsConnected: vi.fn(),
    });

    socket.trigger('workflow_created', { workflow_id: 'wf-1', is_public: true });

    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.arrayContaining([
          { type: 'Workflow', id: LIST_TAG },
          'WorkflowTags',
          'WorkflowTagCounts',
          'WorkflowCategoryCounts',
        ]),
      })
    );
  });

  it('ignores unrelated events for workflow cache invalidation', () => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    const store = {
      dispatch,
      getState: vi.fn(() => ({})),
    };

    setEventListeners({
      socket: socket as never,
      store: store as never,
      setIsConnected: vi.fn(),
    });

    socket.trigger('download_started', { source: 'x', download_path: '/tmp/x' });

    expect(dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.arrayContaining([{ type: 'Workflow', id: LIST_TAG }]),
      })
    );
  });

  it('clears selected workflow ids from call_saved_workflow nodes on workflow_deleted', () => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    const store = {
      dispatch,
      getState: vi.fn(() => ({
        nodes: {
          present: {
            nodes: [
              {
                id: 'call-saved-workflows-node',
                type: 'invocation',
                data: {
                  id: 'call-saved-workflows-node',
                  type: 'call_saved_workflow',
                  inputs: {
                    workflow_id: {
                      value: 'wf-1',
                    },
                  },
                },
              },
            ],
          },
        },
      })),
    };

    setEventListeners({
      socket: socket as never,
      store: store as never,
      setIsConnected: vi.fn(),
    });

    socket.trigger('workflow_deleted', { workflow_id: 'wf-1', is_public: false });

    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          nodeId: 'call-saved-workflows-node',
          fieldName: 'workflow_id',
          value: '',
        }),
      })
    );
  });

  it('does not clear selected workflow ids from call_saved_workflow nodes on workflow_updated', () => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    const store = {
      dispatch,
      getState: vi.fn(() => ({
        nodes: {
          present: {
            nodes: [
              {
                id: 'call-saved-workflows-node',
                type: 'invocation',
                data: {
                  id: 'call-saved-workflows-node',
                  type: 'call_saved_workflow',
                  inputs: {
                    workflow_id: {
                      value: 'wf-1',
                    },
                  },
                },
              },
            ],
          },
        },
      })),
    };

    setEventListeners({
      socket: socket as never,
      store: store as never,
      setIsConnected: vi.fn(),
    });

    socket.trigger('workflow_updated', {
      workflow_id: 'wf-1',
      user_id: 'owner-1',
      old_is_public: true,
      new_is_public: true,
    });

    expect(dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          nodeId: 'call-saved-workflows-node',
          fieldName: 'workflow_id',
          value: '',
        }),
      })
    );
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.arrayContaining([{ type: 'Workflow', id: LIST_TAG }]),
      })
    );
  });

  it('clears selected workflow ids when another user revokes shared access', () => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    const store = {
      dispatch,
      getState: vi.fn(() => ({
        auth: {
          user: { user_id: 'viewer-1', is_admin: false },
        },
        nodes: {
          present: {
            nodes: [
              {
                id: 'call-saved-workflows-node',
                type: 'invocation',
                data: {
                  id: 'call-saved-workflows-node',
                  type: 'call_saved_workflow',
                  inputs: {
                    workflow_id: {
                      value: 'wf-shared',
                    },
                  },
                },
              },
            ],
          },
        },
      })),
    };

    setEventListeners({
      socket: socket as never,
      store: store as never,
      setIsConnected: vi.fn(),
    });

    socket.trigger('workflow_access_revoked', { workflow_id: 'wf-shared', user_id: 'owner-1' });

    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          nodeId: 'call-saved-workflows-node',
          fieldName: 'workflow_id',
          value: '',
        }),
      })
    );
  });

  it.each([
    { user_id: 'owner-1', is_admin: false },
    { user_id: 'admin-1', is_admin: true },
  ])('keeps selected workflow ids when access remains for $user_id', (user) => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    const store = {
      dispatch,
      getState: vi.fn(() => ({
        auth: { user },
        nodes: {
          present: {
            nodes: [
              {
                id: 'call-saved-workflows-node',
                type: 'invocation',
                data: {
                  id: 'call-saved-workflows-node',
                  type: 'call_saved_workflow',
                  inputs: { workflow_id: { value: 'wf-shared' } },
                },
              },
            ],
          },
        },
      })),
    };

    setEventListeners({
      socket: socket as never,
      store: store as never,
      setIsConnected: vi.fn(),
    });

    socket.trigger('workflow_access_revoked', { workflow_id: 'wf-shared', user_id: 'owner-1' });

    expect(dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          nodeId: 'call-saved-workflows-node',
          fieldName: 'workflow_id',
          value: '',
        }),
      })
    );
  });
});

/**
 * In multiuser mode admins are subscribed to the "admin" socket room and receive every user's
 * invocation and queue-item events, carrying that user's real user_id. None of them may drive this
 * client's local execution/progress state. The routing happens at the socket listener, before the
 * workflow execution coordinator records the event, so these tests drive the real listeners + real
 * coordinator rather than calling a downstream handler directly.
 */
describe('setEventListeners cross-user isolation', () => {
  const ADMIN_USER = { user_id: 'admin-1', is_admin: true };
  const FOREIGN_USER_ID = 'other-user';
  const NODE_ID = 'node-1';

  const buildInvocationEvent = (overrides: Record<string, unknown> = {}) => ({
    queue_id: 'default',
    item_id: 99,
    batch_id: 'batch-foreign',
    session_id: 'sess-foreign',
    invocation: { id: 'inv-1', type: 'some_node' },
    invocation_source_id: NODE_ID,
    origin: 'workflows',
    destination: 'canvas',
    user_id: FOREIGN_USER_ID,
    ...overrides,
  });

  const buildQueueItemStatusChangedEvent = (overrides: Record<string, unknown> = {}) => ({
    queue_id: 'default',
    item_id: 99,
    batch_id: 'batch-foreign',
    session_id: 'sess-foreign',
    origin: 'workflows',
    destination: 'canvas',
    user_id: FOREIGN_USER_ID,
    status: 'in_progress',
    status_sequence: 1,
    error_type: null,
    error_message: null,
    error_traceback: null,
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-01T00:01:00',
    started_at: '2026-01-01T00:00:30',
    completed_at: null,
    batch_status: { queue_id: 'default', batch_id: 'batch-foreign', origin: 'workflows', destination: 'canvas' },
    queue_status: { queue_id: 'default' },
    ...overrides,
  });

  /**
   * A node execution state the admin owns locally, sharing a node id with the foreign workflow —
   * common node ids make this realistic. Seeded PENDING (i.e. not running) so that any
   * foreign-driven transition to IN_PROGRESS/FAILED/COMPLETED changes the object and is visible to
   * a toEqual assertion.
   */
  const seedNodeExecutionState = () => {
    const state = {
      nodeId: NODE_ID,
      status: 'PENDING',
      progress: null,
      progressImage: null,
      outputs: [],
      error: null,
    } as never;
    $nodeExecutionStates.set({ [NODE_ID]: state });
    return state;
  };

  const seedProgressEvent = () => {
    const progress = { queue_id: 'default', item_id: 1, user_id: ADMIN_USER.user_id } as never;
    $lastProgressEvent.set(progress);
    return progress;
  };

  const setup = (user: { user_id: string; is_admin: boolean } | null) => {
    const socket = createMockSocket();
    const dispatch = vi.fn();
    setEventListeners({
      socket: socket as never,
      store: { dispatch, getState: vi.fn(() => ({ auth: { user } })) } as never,
      setIsConnected: vi.fn(),
    });
    return { socket, dispatch };
  };

  beforeEach(() => {
    mockOnInvocationComplete.mockReset();
    mockOnForeignInvocationComplete.mockReset();
    vi.mocked(toast).mockReset();
    capturedCompletedInvocationKeys = null;
    $nodeExecutionStates.set({});
    $lastProgressEvent.set(null);
  });

  it.each(['invocation_started', 'invocation_progress', 'invocation_error'])(
    "drops another user's %s for an authenticated admin",
    (eventName) => {
      const seededNodeState = seedNodeExecutionState();
      const seededProgress = seedProgressEvent();
      const { socket, dispatch } = setup(ADMIN_USER);

      socket.trigger(
        eventName,
        buildInvocationEvent({
          // canvas origin would otherwise stop the admin's own integration spinner
          origin: 'canvas_workflow_integration',
          percentage: 0.9,
          error_type: 'RuntimeError',
          error_message: 'boom',
        })
      );

      // Node execution state must not be created, overwritten, or reset.
      expect($nodeExecutionStates.get()).toEqual({ [NODE_ID]: seededNodeState });
      // The admin's own progress event must survive.
      expect($lastProgressEvent.get()).toBe(seededProgress);
      // No canvas processing clear, and no gallery/completion handling.
      expect(dispatch).not.toHaveBeenCalled();
      expect(mockOnInvocationComplete).not.toHaveBeenCalled();
      // Completed-invocation tracking must stay empty.
      expect(capturedCompletedInvocationKeys?.size ?? 0).toBe(0);
    }
  );

  it("routes another user's invocation_complete to the foreign gallery handler only", () => {
    const seededNodeState = seedNodeExecutionState();
    const seededProgress = seedProgressEvent();
    const { socket, dispatch } = setup(ADMIN_USER);

    const event = buildInvocationEvent({
      origin: 'canvas_workflow_integration',
      result: { image: { image_name: 'foreign.png' } },
    });
    socket.trigger('invocation_complete', event);

    // The foreign handler is responsible for the (invalidate-only) gallery refresh...
    expect(mockOnForeignInvocationComplete).toHaveBeenCalledWith(event);
    // ...while the coordinator and the own-event handler never see the event, and no personal
    // state is touched.
    expect(mockOnInvocationComplete).not.toHaveBeenCalled();
    expect($nodeExecutionStates.get()).toEqual({ [NODE_ID]: seededNodeState });
    expect($lastProgressEvent.get()).toBe(seededProgress);
    expect(capturedCompletedInvocationKeys?.size ?? 0).toBe(0);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("does not create node execution state for a foreign workflow's node", () => {
    // getUpdatedNodeExecutionStateOnInvocationStarted() creates state even when none existed, so an
    // unfiltered foreign event would materialize a node the admin's editor does not have.
    $nodeExecutionStates.set({});
    const { socket } = setup(ADMIN_USER);

    socket.trigger('invocation_started', buildInvocationEvent({ invocation_source_id: 'foreign-node' }));

    expect($nodeExecutionStates.get()).toEqual({});
  });

  it("ignores another user's full queue_item_status_changed except for queue cache invalidation", () => {
    const seededNodeState = seedNodeExecutionState();
    const seededProgress = seedProgressEvent();
    const { socket, dispatch } = setup(ADMIN_USER);

    // in_progress would reset the admin's node states via the coordinator
    socket.trigger('queue_item_status_changed', buildQueueItemStatusChangedEvent({ status: 'in_progress' }));

    expect($nodeExecutionStates.get()).toEqual({ [NODE_ID]: seededNodeState });
    expect($lastProgressEvent.get()).toBe(seededProgress);
    expect(capturedCompletedInvocationKeys?.size ?? 0).toBe(0);

    // The admin's queue list/badge still refetch, but only via tag invalidation — no payload-driven
    // optimistic cache writes and no reconciliation request for the other user's item.
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ payload: expect.arrayContaining(['SessionQueueStatus']) })
    );
  });

  it('routes a sanitized queue_item_status_changed to the same invalidate-only branch', () => {
    const { socket, dispatch } = setup(ADMIN_USER);

    socket.trigger('queue_item_status_changed', buildQueueItemStatusChangedEvent({ user_id: 'redacted' }));

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ payload: expect.arrayContaining(['SessionQueueStatus']) })
    );
  });

  it("does not toast or clear progress on another user's failed queue item", () => {
    const seededProgress = seedProgressEvent();
    const { socket } = setup(ADMIN_USER);

    socket.trigger(
      'queue_item_status_changed',
      buildQueueItemStatusChangedEvent({ status: 'failed', error_type: 'RuntimeError', error_message: 'boom' })
    );

    expect(toast).not.toHaveBeenCalled();
    expect($lastProgressEvent.get()).toBe(seededProgress);
  });

  it("still processes the admin's own invocation_progress", () => {
    const { socket } = setup(ADMIN_USER);
    const own = buildInvocationEvent({ user_id: ADMIN_USER.user_id, percentage: 0.5 });

    socket.trigger('invocation_progress', own);

    expect($lastProgressEvent.get()).toEqual(own);
  });

  it("routes the admin's own invocation_complete to the own-event handler", () => {
    const { socket } = setup(ADMIN_USER);
    const own = buildInvocationEvent({ user_id: ADMIN_USER.user_id, result: {} });

    socket.trigger('invocation_complete', own);

    expect(mockOnInvocationComplete).toHaveBeenCalledWith(own);
    expect(mockOnForeignInvocationComplete).not.toHaveBeenCalled();
  });

  it('still processes events in single-user mode, where there is no authenticated user', () => {
    const { socket } = setup(null);
    const own = buildInvocationEvent({ user_id: 'system', percentage: 0.5 });

    socket.trigger('invocation_progress', own);

    expect($lastProgressEvent.get()).toEqual(own);
  });
});
