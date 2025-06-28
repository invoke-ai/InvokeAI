import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { S } from 'services/api/types';
import type { PartialDeep } from 'type-fest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  IterateNodeFoundInGraphError,
  OutputNodeNotFoundInCompletedSessionError,
  OutputNodeNotFoundInGraphError,
  ResultNotFoundInCompletedSessionError,
  runGraph,
  SessionAbortedError,
  SessionCanceledError,
  SessionFailedError,
  SessionTimeoutError,
} from './run-graph';

// Mock dependencies
vi.mock('app/logging/logger', () => ({
  logger: () => ({
    trace: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:mock-id-123`,
}));

const TEST_ID = 'test-graph';
const TEST_ORIGIN = getPrefixedId(TEST_ID);

// Helper functions for creating mock objects
const createMockGraph = (id = TEST_ID, hasIterateNodes = false): Graph => {
  const mockNodes = hasIterateNodes
    ? { node1: { type: 'iterate' }, node2: { type: 'resize' } }
    : { node1: { type: 'resize' }, node2: { type: 'add' } };

  return {
    id,
    hasNode: vi.fn().mockImplementation((nodeId: string) => nodeId === 'output-node'),
    getGraph: vi.fn().mockReturnValue({
      id,
      nodes: mockNodes,
      edges: {},
    }),
  } as unknown as Graph;
};

const createMockQueueItem = (
  status: S['SessionQueueItem']['status'] = 'completed',
  overrides: PartialDeep<S['SessionQueueItem']> = {}
): S['SessionQueueItem'] =>
  ({
    item_id: 1,
    status,
    batch_id: 'test-batch-id',
    queue_id: 'default',
    origin: 'test',
    destination: 'gallery',
    created_at: '2023-01-01T00:00:00Z',
    updated_at: '2023-01-01T00:00:00Z',
    started_at: status === 'in_progress' || status === 'completed' ? '2023-01-01T00:00:00Z' : null,
    completed_at: status === 'completed' ? '2023-01-01T00:00:00Z' : null,
    session: {
      source_prepared_mapping: {
        'output-node': ['prepared-output-node'],
      },
      results: {
        'prepared-output-node': {
          type: 'image_output',
          image: { image_name: 'test.png' },
          width: 512,
          height: 512,
        },
      },
    },
    error_type: null,
    error_message: null,
    error_traceback: null,
    ...overrides,
  }) as S['SessionQueueItem'];

const createMockExecutor = () => ({
  enqueueBatch: vi.fn(),
  getQueueItem: vi.fn(),
  cancelQueueItem: vi.fn(),
});

const createMockEventHandler = () => {
  const handlers = new Set<(event: S['QueueItemStatusChangedEvent']) => void>();

  return {
    subscribe: vi.fn().mockImplementation((handler) => {
      handlers.add(handler);
    }),
    unsubscribe: vi.fn().mockImplementation((handler) => {
      handlers.delete(handler);
    }),
    // Helper to trigger events in tests
    _triggerEvent: (event: S['QueueItemStatusChangedEvent']) => {
      handlers.forEach((handler) => handler(event));
    },
    _getHandlerCount: () => handlers.size,
  };
};

describe('runGraph', () => {
  let mockExecutor: ReturnType<typeof createMockExecutor>;
  let mockEventHandler: ReturnType<typeof createMockEventHandler>;
  let mockGraph: Graph;

  beforeEach(() => {
    mockExecutor = createMockExecutor();
    mockEventHandler = createMockEventHandler();
    mockGraph = createMockGraph();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('validation', () => {
    it('should reject with OutputNodeNotFoundError if graph does not contain output node', async () => {
      mockGraph.hasNode = vi.fn().mockReturnValue(false);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'non-existent-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      await expect(promise).rejects.toThrow(OutputNodeNotFoundInGraphError);
    });

    it('should reject with IterateNodeFoundError if graph contains iterate nodes', async () => {
      const graphWithIterateNodes = createMockGraph('test', true);

      const promise = runGraph({
        graph: graphWithIterateNodes,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      await expect(promise).rejects.toThrow(IterateNodeFoundInGraphError);
    });
  });

  describe('successful execution', () => {
    it('should enqueue graph and return result on completion', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      // Simulate completion event
      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'completed',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      const result = await promise;

      expect(result).toEqual({
        session: mockQueueItem.session,
        output: mockQueueItem.session.results['prepared-output-node'],
      });
      expect(mockExecutor.enqueueBatch).toHaveBeenCalledWith({
        prepend: undefined,
        batch: {
          graph: mockGraph.getGraph(),
          origin: TEST_ORIGIN,
          destination: undefined,
          runs: 1,
        },
      });
    });

    it('should pass through batch configuration options', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        destination: 'test-destination',
        prepend: true,
      });

      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'completed',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      await promise;

      expect(mockExecutor.enqueueBatch).toHaveBeenCalledWith({
        prepend: true,
        batch: {
          graph: mockGraph.getGraph(),
          origin: TEST_ORIGIN,
          destination: 'test-destination',
          runs: 1,
        },
      });
    });
  });

  describe('error handling', () => {
    it('should reject with SessionExecutionError on failed status with error details', async () => {
      const mockQueueItem = createMockQueueItem('failed', {
        error_type: 'ValidationError',
        error_message: 'Invalid input',
        error_traceback: 'Traceback...',
      });
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'failed',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      await expect(promise).rejects.toThrow(SessionFailedError);
    });

    it('should reject with SessionCancelationError on canceled status', async () => {
      const mockQueueItem = createMockQueueItem('canceled');
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'canceled',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      await expect(promise).rejects.toThrow(SessionCanceledError);
    });

    it('should reject if enqueueBatch fails', async () => {
      // The error we are testing here is provided by the API client. We do not know the exact error type.
      const error = new Error('Enqueue failed');
      mockExecutor.enqueueBatch.mockRejectedValue(error);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      await expect(promise).rejects.toThrow('Enqueue failed');
    });

    it('should reject if getQueueItem fails', async () => {
      // The error we are testing here is provided by the API client. We do not know the exact error type.
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockRejectedValue(new Error('Get queue item failed'));

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'completed',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      await expect(promise).rejects.toThrow('Get queue item failed');
    });
  });

  describe('timeout handling', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should timeout, reject with a SessionTimeoutError, and cancel queue item if timeout is exceeded', async () => {
      let resolveEnqueue: (value: { item_ids: number[] }) => void = () => {};
      const enqueuePromise = new Promise<{ item_ids: number[] }>((resolve) => {
        resolveEnqueue = resolve;
      });

      mockExecutor.enqueueBatch.mockReturnValue(enqueuePromise);
      mockExecutor.cancelQueueItem.mockResolvedValue(createMockQueueItem('canceled'));

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        timeout: 1000,
      });

      // Resolve enqueue to set queue item ID
      resolveEnqueue({ item_ids: [1] });
      await Promise.resolve(); // Let the promise resolution be processed

      // Fast-forward time to trigger timeout
      vi.advanceTimersByTime(1001);

      await expect(promise).rejects.toThrow(SessionTimeoutError);
      expect(mockExecutor.cancelQueueItem).toHaveBeenCalledWith(1);
    });

    it('should not timeout if graph completes before timeout', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        timeout: 1000,
      });

      // Complete before timeout
      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      vi.advanceTimersByTime(500);
      const result = await promise;

      expect(result).toBeDefined();
      expect(mockExecutor.cancelQueueItem).not.toHaveBeenCalled();
    });

    it('should timeout and reject with a SessionTimeoutError without canceling if queue item ID is not yet available', async () => {
      // Don't resolve enqueueBatch to simulate slow enqueue
      const slowEnqueuePromise = new Promise(() => {}); // Never resolves
      mockExecutor.enqueueBatch.mockReturnValue(slowEnqueuePromise);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        timeout: 1000,
      });

      // Fast-forward time to trigger timeout before enqueue completes
      vi.advanceTimersByTime(1001);

      await expect(promise).rejects.toThrow(SessionTimeoutError);
      // Should not attempt to cancel since queue item ID is not available
      expect(mockExecutor.cancelQueueItem).not.toHaveBeenCalled();
    });
  });

  describe('abort signal handling', () => {
    it('should reject with a SessionAbortedError and cancel the queue item when signal is aborted', async () => {
      const controller = new AbortController();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.cancelQueueItem.mockResolvedValue(createMockQueueItem('canceled'));

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        signal: controller.signal,
      });

      setImmediate(() => {
        controller.abort();
      });

      await expect(promise).rejects.toThrow(SessionAbortedError);
      expect(mockExecutor.cancelQueueItem).toHaveBeenCalledWith(1);
    });

    it('should not cancel if graph completes before abort', async () => {
      const controller = new AbortController();
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        signal: controller.signal,
      });

      // Complete before abort
      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      const result = await promise;
      expect(result).toBeDefined();

      // Abort after completion should not cancel
      controller.abort();
      expect(mockExecutor.cancelQueueItem).not.toHaveBeenCalled();
    });

    it('should reject with SessionAbortedError and not cancel the queue item if aborted before enqueue completion', async () => {
      const controller = new AbortController();
      // Don't resolve enqueueBatch to simulate slow enqueue
      const slowEnqueuePromise = new Promise(() => {}); // Never resolves
      mockExecutor.enqueueBatch.mockReturnValue(slowEnqueuePromise);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        signal: controller.signal,
      });

      setImmediate(() => {
        controller.abort();
      });

      await expect(promise).rejects.toThrow(SessionAbortedError);
      // Should not attempt to cancel since queue item ID is not available
      expect(mockExecutor.cancelQueueItem).not.toHaveBeenCalled();
    });
  });

  describe('event filtering', () => {
    it('should ignore events with different origins', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      // Trigger event with different origin - should be ignored
      mockEventHandler._triggerEvent({
        item_id: 2,
        status: 'completed',
        origin: 'different-origin',
      } as S['QueueItemStatusChangedEvent']);

      // Trigger correct event
      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'completed',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      const result = await promise;
      expect(result).toBeDefined();
      expect(mockExecutor.getQueueItem).toHaveBeenCalledTimes(1);
      expect(mockExecutor.getQueueItem).toHaveBeenCalledWith(1);
    });

    it('should ignore pending and in_progress status events', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      // These should be ignored
      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'pending',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'in_progress',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      // This should trigger completion
      setImmediate(() => {
        mockEventHandler._triggerEvent({
          item_id: 1,
          status: 'completed',
          origin: TEST_ORIGIN,
        } as S['QueueItemStatusChangedEvent']);
      });

      const result = await promise;
      expect(result).toBeDefined();
      expect(mockExecutor.getQueueItem).toHaveBeenCalledTimes(1);
    });
  });

  describe('cleanup', () => {
    it('should unsubscribe event handler on completion', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      await promise;

      expect(mockEventHandler.unsubscribe).toHaveBeenCalled();
      expect(mockEventHandler._getHandlerCount()).toBe(0);
    });

    it('should unsubscribe event handler on error', async () => {
      mockExecutor.enqueueBatch.mockRejectedValue(new Error('Enqueue failed'));

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      await expect(promise).rejects.toThrow('Enqueue failed');
      expect(mockEventHandler.unsubscribe).toHaveBeenCalled();
      expect(mockEventHandler._getHandlerCount()).toBe(0);
    });
  });

  describe('output extraction', () => {
    it('should reject with NodeNotFoundError if output node not in session mapping', async () => {
      const mockQueueItem = createMockQueueItem('completed', {
        session: {
          source_prepared_mapping: {
            'different-node': ['prepared-different-node'],
          },
          results: {
            'prepared-different-node': {
              type: 'image_output',
              image: { image_name: 'different.png' },
              width: 512,
              height: 512,
            },
          },
        },
      });
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      await expect(promise).rejects.toThrow(OutputNodeNotFoundInCompletedSessionError);
    });

    it('should reject with ResultNotFoundError if result not found for prepared node', async () => {
      const mockQueueItem = createMockQueueItem('completed', {
        session: {
          source_prepared_mapping: {
            'output-node': ['prepared-output-node'],
          },
          results: {
            'different-prepared-node': {
              type: 'image_output',
              image: { image_name: 'different.png' },
              width: 512,
              height: 512,
            },
          },
        },
      });
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      await expect(promise).rejects.toThrow(ResultNotFoundInCompletedSessionError);
    });
  });

  describe('race conditions and timing', () => {
    it('should handle events arriving before enqueue completes', async () => {
      let enqueueResolve: (value: { item_ids: number[] }) => void;
      const enqueuePromise = new Promise<{ item_ids: number[] }>((resolve) => {
        enqueueResolve = resolve;
      });
      mockExecutor.enqueueBatch.mockReturnValue(enqueuePromise);

      const mockQueueItem = createMockQueueItem();
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      // Trigger completion event before enqueue resolves
      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      // Now resolve the enqueue
      enqueueResolve!({ item_ids: [1] });

      const result = await promise;
      expect(result.session).toBe(mockQueueItem.session);
      expect(result.output).toEqual(mockQueueItem.session.results['prepared-output-node']);
    });

    it('should handle multiple rapid status changes', async () => {
      const mockQueueItem = createMockQueueItem();
      mockExecutor.enqueueBatch.mockResolvedValue({ item_ids: [1] });
      mockExecutor.getQueueItem.mockResolvedValue(mockQueueItem);

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
      });

      // Trigger rapid sequence of events
      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'pending',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'in_progress',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      // Trigger another completed event (should be ignored since already resolved)
      mockEventHandler._triggerEvent({
        item_id: 1,
        status: 'completed',
        origin: TEST_ORIGIN,
      } as S['QueueItemStatusChangedEvent']);

      const result = await promise;
      expect(result.session).toBe(mockQueueItem.session);
      expect(result.output).toEqual(mockQueueItem.session.results['prepared-output-node']);

      // Should only call getQueueItem once despite multiple completion events
      expect(mockExecutor.getQueueItem).toHaveBeenCalledTimes(1);
    });

    it('should cleanup properly when promise is never resolved', async () => {
      // Mock enqueueBatch to never resolve
      const neverResolvingPromise = new Promise(() => {
        // This promise never resolves
      });
      mockExecutor.enqueueBatch.mockReturnValue(neverResolvingPromise);

      const controller = new AbortController();

      const promise = runGraph({
        graph: mockGraph,
        outputNodeId: 'output-node',
        dependencies: { executor: mockExecutor, eventHandler: mockEventHandler },
        signal: controller.signal,
      });

      // Abort the operation while enqueueBatch is still pending
      controller.abort();

      await expect(promise).rejects.toThrow(SessionAbortedError);

      // Verify cleanup happened - event handler should be unsubscribed
      expect(mockEventHandler.unsubscribe).toHaveBeenCalled();
      expect(mockEventHandler._getHandlerCount()).toBe(0);

      // cancelQueueItem should not be called since we don't have a queue item ID yet
      expect(mockExecutor.cancelQueueItem).not.toHaveBeenCalled();
    });
  });
});
