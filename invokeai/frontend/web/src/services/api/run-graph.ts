import { logger } from 'app/logging/logger';
import type { AppDispatch } from 'app/store/store';
import { Mutex } from 'async-mutex';
import type { Deferred } from 'common/util/createDeferredPromise';
import { createDeferredPromise } from 'common/util/createDeferredPromise';
import { withResultAsync, WrappedError } from 'common/util/result';
import { parseify } from 'common/util/serialize';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import type { JsonObject } from 'type-fest';

import { enqueueMutationFixedCacheKeyOptions, queueApi } from './endpoints/queue';
import type { EnqueueBatchArg } from './types';

const log = logger('system');

type QueueStatusEventHandler = {
  subscribe: (handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
  unsubscribe: (handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
};

type GraphExecutor = {
  enqueueBatch: (batch: EnqueueBatchArg) => Promise<{ item_ids: number[] }>;
  getQueueItem: (id: number) => Promise<S['SessionQueueItem']>;
  cancelQueueItem: (id: number) => Promise<S['SessionQueueItem']>;
};

type GraphRunnerDependencies = {
  executor: GraphExecutor;
  eventHandler: QueueStatusEventHandler;
};

export type RunGraphOptions = {
  destination?: string;
  prepend?: boolean;
  timeout?: number;
  signal?: AbortSignal;
};

type RunGraphArg = {
  graph: Graph;
  outputNodeId: string;
  dependencies: GraphRunnerDependencies;
  options?: RunGraphOptions;
};

type RunGraphReturn = {
  session: S['SessionQueueItem']['session'];
  output: S['GraphExecutionState']['results'][string];
};

/**
 * Run a graph and return the output of a specific node.
 *
 * The batch will be enqueued with runs set to 1, meaning it will only run once.
 *
 * Iterate nodes, which cause graph expansion, are not supported by this utility because they cause a single node
 * to have multiple outputs. An error will be thrown if the graph contains any iterate nodes.
 *
 * @param arg.graph The graph to execute as an instance of the Graph class.
 * @param arg.outputNodeId The id of the node whose output will be retrieved.
 * @param arg.dependencies The dependencies for queue operations and event handling.
 * @param arg.options Optional parameters for the run:
 * @param arg.options.destination The destination to assign to the batch. If omitted, the destination is not set.
 * @param arg.options.prepend Whether to prepend the graph to the front of the queue. If omitted, the graph is appended to the
 *    end of the queue.
 * @param arg.options.timeout The timeout for the run in milliseconds. The promise rejects with a SessionTimeoutError when
 *    the run times out. If the queue item was enqueued, a best effort is made to cancel it. **If omitted, there is
 *    no timeout and the run will wait indefinitely for completion.**
 * @param arg.options.signal An optional signal to cancel the operation. The promise rejects with a SessionAbortedError when
 *    the run is canceled via signal. If the queue item was enqueued, a best effort is made to cancel it. **If omitted,
 *    the run cannot easily be canceled.**
 *
 * @returns A promise that resolves to the output and completed session, or rejects with an error:
 * - `OutputNodeNotFoundInGraphError` if the output node is not found in the provided graph.
 * - `IterateNodeFoundInGraphError` if the graph contains any iterate nodes, which are not supported.
 * - `UnexpectedStatusError` if the session has an unexpected status (not completed, failed, canceled).
 * - `OutputNodeNotFoundInCompletedSessionError` if the output node is not found in the completed session.
 * - `ResultNotFoundInCompletedSessionError` if the result for the output node is not found in the completed session.
 * - `SessionFailedError` if the session execution fails, including error type, message, and traceback.
 * - `SessionCanceledError` if the session execution is canceled via the queue.
 * - `SessionAbortedError` if the session execution is aborted via signal. Includes information on whether cancellation
 *    failed and the cancellation error, if any.
 * - `SessionTimeoutError` if the session execution times out. Includes information on whether cancellation failed and
 *    the cancellation error, if any.
 *
 * @example
 *
 * ```ts
 * const dependencies = buildRunGraphDependencies(store, socket);
 * const graph = new Graph();
 * const outputNode = graph.addNode({ id: 'my-resize-node', type: 'img_resize', image: { image_name: 'my-image.png' } });
 * const controller = new AbortController();
 * const result = await runGraph({
 *  graph,
 *  outputNodeId: outputNode.id,
 *  dependencies,
 *  prepend: true,
 *  signal: controller.signal,
 * });
 * // To cancel the operation:
 * controller.abort();
 * ```
 */
export const runGraph = (arg: RunGraphArg): Promise<RunGraphReturn> => {
  // A deferred promise works around the antipattern of async promise executors.
  const { promise, resolve, reject } = createDeferredPromise<RunGraphReturn>();
  _runGraph(arg, resolve, reject);
  return promise;
};

/**
 * Creates production dependencies for runGraph using Redux store and socket.
 */
export const buildRunGraphDependencies = (
  dispatch: AppDispatch,
  socket: {
    on: (event: 'queue_item_status_changed', handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
    off: (event: 'queue_item_status_changed', handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
  }
): GraphRunnerDependencies => ({
  executor: {
    enqueueBatch: (batch) =>
      dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batch, {
          ...enqueueMutationFixedCacheKeyOptions,
          track: false,
        })
      ).unwrap(),
    getQueueItem: (id) => dispatch(queueApi.endpoints.getQueueItem.initiate(id, { subscribe: false })).unwrap(),
    cancelQueueItem: (id) =>
      dispatch(queueApi.endpoints.cancelQueueItem.initiate({ item_id: id }, { track: false })).unwrap(),
  },
  eventHandler: {
    subscribe: (handler) => socket.on('queue_item_status_changed', handler),
    unsubscribe: (handler) => socket.off('queue_item_status_changed', handler),
  },
});

/**
 * Internal business logic for running a graph.
 *
 * This function is not intended to be used directly. Use `runGraph` instead.
 *
 * @param arg The arguments for running the graph.
 * @param _resolve The resolve function for the promise. Do not call this directly; use the `settle` function instead.
 * @param _reject The reject function for the promise. Do not call this directly; use the `settle` function instead.
 */
const _runGraph = async (
  arg: RunGraphArg,
  _resolve: Deferred<RunGraphReturn>['resolve'],
  _reject: Deferred<RunGraphReturn>['reject']
): Promise<void> => {
  const { graph, outputNodeId, dependencies, options } = arg;
  const { destination, prepend, timeout, signal } = options ?? {};

  /**
   * We will use the origin to filter out socket events unrelated to this graph.
   *
   * Ideally we'd use the queue item's id, but there's a race condition for fast-running graphs:
   * - We enqueue the batch, which initiates a network request.
   * - The queue item is created and quickly completed.
   * - The enqueue batch request returns, which includes the queue item id.
   * - We set up listeners for the queue item status change events, but the queue item is already completed, so we
   *   miss the status change event and are left waiting forever.
   *
   * The origin is a unique identifier that we can set before enqueuing the graph. This allows us to set up listeners
   * _before_ enqueuing the graph, ensuring that we don't miss any events.
   */
  const origin = getPrefixedId(graph.id);

  /**
   * The queue item id is set to null initially, but will be updated once the graph is enqueued. It will be used to
   * retrieve the queue item.
   */
  let queueItemId: number | null = null;

  /**
   * Set of cleanup functions for listeners, timeouts, etc that need to be called when the graph is settled.
   */
  const cleanupFunctions: Set<() => void> = new Set();
  const cleanup = () => {
    for (const func of cleanupFunctions) {
      try {
        func();
      } catch (error) {
        log.warn({ error: parseify(error) }, 'Error during runGraph cleanup');
      }
    }
    cleanupFunctions.clear();
  };

  /**
   * We use a mutex to ensure that the promise is resolved or rejected only once, even if multiple events
   * are received or the settle function is called multiple times.
   *
   * A flag allows pending locks to bail if the promise has already been settled.
   */
  const settlementMutex = new Mutex();
  let isSettling = false;

  /**
   * Wraps all logic that settles the promise. This function will handle the cleanup of listeners, timeouts, etc. and
   * resolve or reject the promise.
   *
   * Once the graph execution is finished, all remaining logic should be wrapped in this function to avoid race
   * conditions or multiple resolutions/rejections of the promise.
   *
   * @param settlement A function that returns a `RunGraphReturn` object or a promise that resolves to a
   *    `RunGraphReturn` object. The function should throw an error if the run was not successful.
   */
  const settle = async (settlement: () => Promise<RunGraphReturn> | RunGraphReturn) => {
    await settlementMutex.runExclusive(async () => {
      // If we are already settling, ignore this call to avoid multiple resolutions or rejections.
      // We don't want to _cancel_ pending locks as this would raise.
      if (isSettling) {
        return;
      }
      isSettling = true;

      // Clean up listeners, timeouts, etc. ASAP.
      cleanup();

      // Normalize the settlement function to always return a promise and wrap in a result to handle errors.
      const result = await withResultAsync(() => Promise.resolve(settlement()));

      const ctx: JsonObject = {
        queueItemId,
        graphId: graph.id,
        outputNodeId,
        destination: destination ?? 'not provided',
        prepend: prepend ?? false,
        timeout: timeout ?? 'not provided',
        signal: signal !== undefined ? 'provided' : 'not provided',
      };

      if (result.isOk()) {
        log.debug({ ...ctx, output: parseify(result.value) }, 'Run completed successfully');
        _resolve(result.value);
      } else {
        log.debug({ ...ctx, error: parseify(result.error) }, 'Run failed');
        _reject(result.error);
      }
    });
  };

  if (!graph.hasNode(outputNodeId)) {
    await settle(() => {
      throw new OutputNodeNotFoundInGraphError(outputNodeId, graph);
    });
    return;
  }

  if (graph.getNodes().some((node) => node.type === 'iterate')) {
    await settle(() => {
      throw new IterateNodeFoundInGraphError(graph);
    });
    return;
  }

  // If a timeout value is provided, we create a timer to reject the promise.
  if (timeout !== undefined) {
    const timeoutId = setTimeout(async () => {
      await settle(async () => {
        let cancellationFailed = false;
        let cancellationError: Error | null = null;

        if (queueItemId !== null) {
          try {
            await dependencies.executor.cancelQueueItem(queueItemId);
          } catch (error) {
            cancellationFailed = true;
            cancellationError = WrappedError.wrap(error);
          }
        }

        throw new SessionTimeoutError(queueItemId, cancellationFailed, cancellationError);
      });
    }, timeout);

    cleanupFunctions.add(() => {
      clearTimeout(timeoutId);
    });
  }

  // If a signal is provided, we add an abort handler to reject the promise if the signal is aborted.
  if (signal !== undefined) {
    const abortHandler = async () => {
      await settle(async () => {
        let cancellationFailed = false;
        let cancellationError: Error | null = null;

        if (queueItemId !== null) {
          try {
            await dependencies.executor.cancelQueueItem(queueItemId);
          } catch (error) {
            cancellationFailed = true;
            cancellationError = WrappedError.wrap(error);
          }
        }

        throw new SessionAbortedError(queueItemId, cancellationFailed, cancellationError);
      });
    };

    signal.addEventListener('abort', abortHandler);
    cleanupFunctions.add(() => {
      signal.removeEventListener('abort', abortHandler);
    });
  }

  // Handle the queue item status change events.
  const onQueueItemStatusChanged = async (event: S['QueueItemStatusChangedEvent']) => {
    // Ignore events that are not for this graph
    if (event.origin !== origin) {
      return;
    }

    // Ignore events where the status is pending or in progress - no need to do anything for these
    if (event.status === 'pending' || event.status === 'in_progress') {
      return;
    }

    await settle(async () => {
      // We need to handle any errors, including retrieving the queue item
      const queueItem = await dependencies.executor.getQueueItem(event.item_id);
      const { status, session, error_type, error_message, error_traceback } = queueItem;

      // We are confident that the queue item is not pending or in progress, at this time.
      if (status === 'pending' || status === 'in_progress') {
        throw new UnexpectedStatusError(event.item_id, session, status);
      }

      if (status === 'completed') {
        const output = getOutputFromSession(queueItemId, session, outputNodeId);
        return { session, output };
      }

      if (status === 'failed') {
        throw new SessionFailedError(queueItemId, session, error_type, error_message, error_traceback);
      }

      if (status === 'canceled') {
        throw new SessionCanceledError(queueItemId, session);
      }

      assert<Equals<never, typeof status>>(false);
    });
  };

  dependencies.eventHandler.subscribe(onQueueItemStatusChanged);
  cleanupFunctions.add(() => {
    dependencies.eventHandler.unsubscribe(onQueueItemStatusChanged);
  });

  try {
    const batch: EnqueueBatchArg = {
      prepend,
      batch: {
        graph: graph.getGraph(),
        origin,
        destination,
        runs: 1,
      },
    };
    const { item_ids } = await dependencies.executor.enqueueBatch(batch);
    // We expect exactly one item id to be returned. We control the batch config, so we can safely assert this.
    assert(item_ids.length === 1);
    assert(item_ids[0] !== undefined);
    queueItemId = item_ids[0];
  } catch (error) {
    settle(() => {
      throw WrappedError.wrap(error);
    });
  }
};

const getOutputFromSession = (
  queueItemId: number | null,
  session: S['SessionQueueItem']['session'],
  nodeId: string
): S['SessionQueueItem']['session']['results'][string] => {
  const { results, source_prepared_mapping } = session;
  const preparedNodeId = source_prepared_mapping[nodeId]?.[0];
  if (!preparedNodeId) {
    throw new OutputNodeNotFoundInCompletedSessionError(queueItemId, session, nodeId);
  }
  const result = results[preparedNodeId];
  if (!result) {
    throw new ResultNotFoundInCompletedSessionError(queueItemId, session, nodeId);
  }
  return result;
};

export class OutputNodeNotFoundInGraphError extends Error {
  public readonly outputNodeId: string;
  public readonly graph: Graph;

  constructor(outputNodeId: string, graph: Graph) {
    super(`Output node '${outputNodeId}' not found in the graph.`);
    this.name = this.constructor.name;
    this.outputNodeId = outputNodeId;
    this.graph = graph;
  }
}

export class IterateNodeFoundInGraphError extends Error {
  public readonly graph: Graph;

  constructor(graph: Graph) {
    super('Iterate node(s) found in the graph.');
    this.name = this.constructor.name;
    this.graph = graph;
  }
}

class BaseQueueItemError extends Error {
  public readonly queueItemId: number | null;

  constructor(queueItemId: number | null, message?: string) {
    super(message ?? 'Queue item error occurred');
    this.name = this.constructor.name;
    this.queueItemId = queueItemId;
  }
}

class BaseSessionError extends BaseQueueItemError {
  public readonly session: S['SessionQueueItem']['session'];

  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session'], message?: string) {
    super(queueItemId, message ?? 'Session error occurred');
    this.name = this.constructor.name;
    this.session = session;
  }
}

class UnexpectedStatusError extends BaseSessionError {
  public readonly status: S['SessionQueueItem']['status'];

  constructor(
    queueItemId: number | null,
    session: S['SessionQueueItem']['session'],
    status: S['SessionQueueItem']['status']
  ) {
    super(queueItemId, session, `Session has unexpected status ${status}.`);
    this.name = 'UnexpectedStatusError';
    this.status = status;
  }
}

export class OutputNodeNotFoundInCompletedSessionError extends BaseSessionError {
  public readonly nodeId: string;

  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session'], nodeId: string) {
    super(queueItemId, session, `Node '${nodeId}' not found in session.`);
    this.name = 'OutputNodeNotFoundInCompletedSessionError';
    this.nodeId = nodeId;
  }
}

export class ResultNotFoundInCompletedSessionError extends BaseSessionError {
  public readonly nodeId: string;

  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session'], nodeId: string) {
    super(queueItemId, session, `Result for node '${nodeId}' not found in session.`);
    this.name = 'ResultNotFoundInCompletedSessionError';
    this.nodeId = nodeId;
  }
}

export class SessionFailedError extends BaseSessionError {
  public readonly error_type?: string | null;
  public readonly error_message?: string | null;
  public readonly error_traceback?: string | null;

  constructor(
    queueItemId: number | null,
    session: S['SessionQueueItem']['session'],
    error_type?: string | null,
    error_message?: string | null,
    error_traceback?: string | null
  ) {
    super(queueItemId, session, 'Session execution failed');
    this.name = 'SessionFailedError';
    this.error_type = error_type;
    this.error_traceback = error_traceback;
    this.error_message = error_message;
  }
}

export class SessionCanceledError extends BaseSessionError {
  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session']) {
    super(queueItemId, session, 'Session execution was canceled');
    this.name = 'SessionCanceledError';
  }
}

export class SessionAbortedError extends BaseQueueItemError {
  public readonly cancellationFailed: boolean;
  public readonly cancellationError: Error | null;

  constructor(queueItemId: number | null, cancellationFailed = false, cancellationError: Error | null) {
    const message = cancellationFailed
      ? 'Session execution was aborted via signal and cancellation failed'
      : 'Session execution was aborted via signal';
    super(queueItemId, message);
    this.name = 'SessionAbortedError';
    this.cancellationFailed = cancellationFailed;
    this.cancellationError = cancellationError;
  }
}

export class SessionTimeoutError extends BaseQueueItemError {
  public readonly cancellationFailed: boolean;
  public readonly cancellationError: Error | null;

  constructor(queueItemId: number | null, cancellationFailed = false, cancellationError: Error | null) {
    const message = cancellationFailed
      ? 'Session execution timed out and cancellation failed'
      : 'Session execution timed out';
    super(queueItemId, message);
    this.name = 'SessionTimeoutError';
    this.cancellationFailed = cancellationFailed;
    this.cancellationError = cancellationError;
  }
}
