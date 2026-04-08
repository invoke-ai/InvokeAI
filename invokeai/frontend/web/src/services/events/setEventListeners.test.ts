import { LIST_TAG } from 'services/api';
import { describe, expect, it, vi } from 'vitest';

import { setEventListeners } from './setEventListeners';

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

vi.mock('./onInvocationComplete', () => ({
  buildOnInvocationComplete: () => vi.fn(),
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

  it('clears selected workflow ids from call_saved_workflows nodes on workflow_deleted', () => {
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
                  type: 'call_saved_workflows',
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

  it('does not clear selected workflow ids from call_saved_workflows nodes on workflow_updated', () => {
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
                  type: 'call_saved_workflows',
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
});
