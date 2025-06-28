import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { Mutex } from 'async-mutex';
import type { Result } from 'common/util/result';
import { ErrResult, OkResult, withResult, withResultAsync } from 'common/util/result';
import { parseify } from 'common/util/serialize';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { enqueueMutationFixedCacheKeyOptions, queueApi } from './endpoints/queue';
import type { EnqueueBatchArg } from './types';

const log = logger('queue');

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
};

/**
 * Create a promise and expose its resolve and reject callbacks.
 */
const createDeferredPromise = <T>(): Deferred<T> => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
};

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

type RunGraphArg = {
  graph: Graph;
  outputNodeId: string;
  dependencies: GraphRunnerDependencies;
  destination?: string;
  prepend?: boolean;
  timeout?: number;
  signal?: AbortSignal;
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
 * @param arg.destination The destination to assign to the batch. If omitted, the destination is not set.
 * @param arg.prepend Whether to prepend the graph to the front of the queue. If omitted, the graph is appended to the end of the queue.
 * @param arg.timeout The timeout for the batch. If omitted, there is no timeout.
 * @param arg.signal An optional signal to cancel the operation. If omitted, the operation cannot be canceled.
 *
 * @returns A promise that resolves to the output and completed session, or rejects with an error if the graph fails or is canceled.
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
  store: AppStore,
  socket: {
    on: (event: 'queue_item_status_changed', handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
    off: (event: 'queue_item_status_changed', handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
  }
): GraphRunnerDependencies => ({
  executor: {
    enqueueBatch: (batch) =>
      store
        .dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batch, {
            ...enqueueMutationFixedCacheKeyOptions,
            track: false,
          })
        )
        .unwrap(),
    getQueueItem: (id) => store.dispatch(queueApi.endpoints.getQueueItem.initiate(id, { subscribe: false })).unwrap(),
    cancelQueueItem: (id) =>
      store.dispatch(queueApi.endpoints.cancelQueueItem.initiate({ item_id: id }, { track: false })).unwrap(),
  },
  eventHandler: {
    subscribe: (handler) => socket.on('queue_item_status_changed', handler),
    unsubscribe: (handler) => socket.off('queue_item_status_changed', handler),
  },
});

/**
 * Internal business logic for running a graph.
 */
const _runGraph = async (
  arg: RunGraphArg,
  resolve: (value: RunGraphReturn) => void,
  reject: (error: Error) => void
): Promise<void> => {
  const { graph, outputNodeId, dependencies, destination, prepend, timeout, signal } = arg;

  if (!graph.hasNode(outputNodeId)) {
    reject(new OutputNodeNotFoundError(outputNodeId, graph));
    return;
  }

  const g = graph.getGraph();

  if (Object.values(g.nodes).some((node) => node.type === 'iterate')) {
    reject(new IterateNodeFoundError(graph));
    return;
  }

  /**
   * We will use the origin to handle events from the graph. Ideally we'd just use the queue item's id, but there's a
   * race condition for fast-running graphs:
   * - We enqueue the batch and wait for the respose from the network request, which will include the queue item id.
   * - The queue item is executed.
   * - We get status change events for the queue item, but we don't have the queue item id yet, so we miss the event.
   *
   * The origin is the only unique identifier that we can set before enqueuing the graph. We set it to something
   * unique and use it to filter for events relevant to this graph.
   */
  const origin = getPrefixedId(graph.id);

  const batch: EnqueueBatchArg = {
    prepend,
    batch: {
      graph: g,
      origin,
      destination,
      runs: 1,
    },
  };

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
        log.warn({ error: parseify(error) }, 'Error during cleanup');
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
  let isSettling = false;
  const settlementMutex = new Mutex();
  const settle = async (settlement: () => Promise<Result<RunGraphReturn, Error>> | Result<RunGraphReturn, Error>) => {
    await settlementMutex.runExclusive(async () => {
      // If we are already settling, ignore this call to avoid multiple resolutions or rejections.
      // We don't want to _cancel_ pending locks as this would raise.
      if (isSettling) {
        return;
      }
      isSettling = true;

      // Clean up listeners, timeouts, etc. ASAP.
      cleanup();

      // Normalize the settlement function to always return a promise.
      const result = await Promise.resolve(settlement());

      if (result.isOk()) {
        resolve(result.value);
      } else {
        reject(result.error);
      }
    });
  };

  // If a timeout value is provided, we create a timer to reject the promise.
  if (timeout !== undefined) {
    const timeoutId = setTimeout(async () => {
      await settle(() => {
        log.trace('Graph canceled by timeout');
        if (queueItemId !== null) {
          // It's possible the cancelation will fail, but we have no way to handle that gracefully. Log a warning
          // and move on to reject.
          dependencies.executor.cancelQueueItem(queueItemId).catch((error) => {
            log.warn({ error: parseify(error) }, 'Failed to cancel queue item during timeout');
          });
        }
        return ErrResult(new SessionTimeoutError(queueItemId));
      });
    }, timeout);

    cleanupFunctions.add(() => {
      clearTimeout(timeoutId);
    });
  }

  // If a signal is provided, we add an abort handler to reject the promise if the signal is aborted.
  if (signal !== undefined) {
    const abortHandler = () => {
      settle(() => {
        log.trace('Graph canceled by signal');
        if (queueItemId !== null) {
          // It's possible the cancelation will fail, but we have no way to handle that gracefully. Log a warning
          // and move on to reject.
          dependencies.executor.cancelQueueItem(queueItemId).catch((error) => {
            log.warn({ error: parseify(error) }, 'Failed to cancel queue item during abort');
          });
        }
        return ErrResult(new SessionAbortedError(queueItemId));
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
      const queueItemResult = await withResultAsync(() => dependencies.executor.getQueueItem(event.item_id));
      if (queueItemResult.isErr()) {
        return ErrResult(queueItemResult.error);
      }

      const queueItem = queueItemResult.value;

      const { status, session, error_type, error_message, error_traceback } = queueItem;

      // We are confident that the queue item is not pending or in progress, at this time.
      assert(status !== 'pending' && status !== 'in_progress');

      if (status === 'completed') {
        const getOutputResult = withResult(() => getOutputFromSession(queueItemId, session, outputNodeId));
        if (getOutputResult.isErr()) {
          return ErrResult(getOutputResult.error);
        }
        const output = getOutputResult.value;
        return OkResult({ session, output });
      }

      if (status === 'failed') {
        return ErrResult(new SessionExecutionError(queueItemId, session, error_type, error_message, error_traceback));
      }

      if (status === 'canceled') {
        return ErrResult(new SessionCancelationError(queueItemId, session));
      }

      assert<Equals<never, typeof status>>(false);
    });
  };

  dependencies.eventHandler.subscribe(onQueueItemStatusChanged);
  cleanupFunctions.add(() => {
    dependencies.eventHandler.unsubscribe(onQueueItemStatusChanged);
  });

  const enqueueResult = await withResultAsync(() => dependencies.executor.enqueueBatch(batch));
  if (enqueueResult.isErr()) {
    // The enqueue operation itself failed - we cannot proceed.
    await settle(() => ErrResult(enqueueResult.error));
    return;
  }

  // Retrieve the queue item id from the enqueue result.
  const { item_ids } = enqueueResult.value;
  // We expect exactly one item id to be returned. We control the batch config, so we can safely assert this.
  assert(item_ids.length === 1);
  assert(item_ids[0] !== undefined);
  queueItemId = item_ids[0];
};

const getOutputFromSession = (
  queueItemId: number | null,
  session: S['SessionQueueItem']['session'],
  nodeId: string
): S['SessionQueueItem']['session']['results'][string] => {
  const { results, source_prepared_mapping } = session;
  const preparedNodeId = source_prepared_mapping[nodeId]?.[0];
  if (!preparedNodeId) {
    throw new NodeNotFoundError(queueItemId, session, nodeId);
  }
  const result = results[preparedNodeId];
  if (!result) {
    throw new ResultNotFoundError(queueItemId, session, nodeId);
  }
  return result;
};

export class OutputNodeNotFoundError extends Error {
  outputNodeId: string;
  graph: Graph;

  constructor(outputNodeId: string, graph: Graph) {
    super(`Output node '${outputNodeId}' not found in the graph.`);
    this.name = this.constructor.name;
    this.outputNodeId = outputNodeId;
    this.graph = graph;
  }
}

export class IterateNodeFoundError extends Error {
  graph: Graph;

  constructor(graph: Graph) {
    super('Iterate node(s) found in the graph.');
    this.name = this.constructor.name;
    this.graph = graph;
  }
}

export class QueueItemError extends Error {
  queueItemId: number | null;

  constructor(queueItemId: number | null, message?: string) {
    super(message ?? 'Queue item error occurred');
    this.name = this.constructor.name;
    this.queueItemId = queueItemId;
  }
}

export class SessionError extends QueueItemError {
  session: S['SessionQueueItem']['session'];

  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session'], message?: string) {
    super(queueItemId, message ?? 'Session error occurred');
    this.name = this.constructor.name;
    this.session = session;
  }
}

export class UnexpectedStatusError extends SessionError {
  status: S['SessionQueueItem']['status'];

  constructor(
    queueItemId: number | null,
    session: S['SessionQueueItem']['session'],
    status: S['SessionQueueItem']['status']
  ) {
    super(queueItemId, session, `Session has unexpected status ${status}.`);
    this.name = this.constructor.name;
    this.status = status;
  }
}

export class NodeNotFoundError extends SessionError {
  nodeId: string;

  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session'], nodeId: string) {
    super(queueItemId, session, `Node '${nodeId}' not found in session.`);
    this.name = this.constructor.name;
    this.nodeId = nodeId;
  }
}

export class ResultNotFoundError extends SessionError {
  nodeId: string;

  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session'], nodeId: string) {
    super(queueItemId, session, `Result for node '${nodeId}' not found in session.`);
    this.name = this.constructor.name;
    this.nodeId = nodeId;
  }
}

export class SessionExecutionError extends SessionError {
  error_type?: string | null;
  error_message?: string | null;
  error_traceback?: string | null;

  constructor(
    queueItemId: number | null,
    session: S['SessionQueueItem']['session'],
    error_type?: string | null,
    error_message?: string | null,
    error_traceback?: string | null
  ) {
    super(queueItemId, session, 'Session execution failed');
    this.name = this.constructor.name;
    this.error_type = error_type;
    this.error_traceback = error_traceback;
    this.error_message = error_message;
  }
}

export class SessionCancelationError extends SessionError {
  constructor(queueItemId: number | null, session: S['SessionQueueItem']['session']) {
    super(queueItemId, session, 'Session execution was canceled');
    this.name = this.constructor.name;
  }
}

export class SessionAbortedError extends QueueItemError {
  constructor(queueItemId: number | null) {
    super(queueItemId, 'Session execution was aborted via signal');
    this.name = this.constructor.name;
  }
}

export class SessionTimeoutError extends QueueItemError {
  constructor(queueItemId: number | null) {
    super(queueItemId, 'Session execution timed out');
    this.name = this.constructor.name;
  }
}
