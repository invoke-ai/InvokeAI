import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { withResult, withResultAsync } from 'common/util/result';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { S } from 'services/api/types';
import { QueueError } from 'services/events/errors';
import { assert } from 'tsafe';

import { enqueueMutationFixedCacheKeyOptions, queueApi } from './endpoints/queue';
import type { EnqueueBatchArg } from './types';

const log = logger('queue');

interface QueueStatusEventHandler {
  subscribe: (handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
  unsubscribe: (handler: (event: S['QueueItemStatusChangedEvent']) => void) => void;
}

interface GraphExecutor {
  enqueueBatch: (batch: EnqueueBatchArg) => Promise<{ item_ids: number[] }>;
  getQueueItem: (id: number) => Promise<S['SessionQueueItem']>;
  cancelQueueItem: (id: number) => Promise<S['SessionQueueItem']>;
}

interface GraphRunnerDependencies {
  executor: GraphExecutor;
  eventHandler: QueueStatusEventHandler;
}

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
 * Creates production dependencies for runGraph using Redux store and socket.
 */
export const createProductionDependencies = (
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
 * @param arg.signal An optional signal to cancel the operation. If omitted, the operation cannot be canceled!
 *
 * @returns A promise that resolves to the output and completed session, or rejects with an error if the graph fails or is canceled.
 *
 * @example
 *
 * ```ts
 * const dependencies = createProductionDependencies(store, socket);
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
  const promise = new Promise<RunGraphReturn>((resolve, reject) => {
    const { graph, outputNodeId, dependencies, destination, prepend, timeout, signal } = arg;

    if (!graph.hasNode(outputNodeId)) {
      reject(new Error(`Graph does not contain output node ${outputNodeId}.`));
      return;
    }

    const g = graph.getGraph();

    if (Object.values(g.nodes).some((node) => node.type === 'iterate')) {
      reject(new Error('Iterate nodes are not supported by this utility.'));
      return;
    }

    /**
     * We will use the origin to handle events from the graph. Ideally we'd just use the queue item's id, but there's a
     * race condition:
     * - The queue item id is not available until the graph is enqueued.
     * - The graph may complete before we get a response back from enqueuing, so our listeners would miss the event.
     *
     * The origin is the only unique identifier that we can set before enqueuing the graph, so we use it to filter
     * queue item status change events.
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
     * Flag to indicate whether the graph has already been resolved. This is used to prevent multiple resolutions.
     */
    let isResolved = false;

    /**
     * The queue item id is set to null initially, but will be updated once the graph is enqueued.
     */
    let queueItemId: number | null = null;

    const cleanupFunctions: Set<() => void> = new Set();
    const cleanup = () => {
      for (const func of cleanupFunctions) {
        try {
          func();
        } catch (error) {
          log.warn({ error: parseify(error) }, 'Error during cleanup');
        }
      }
    };

    if (timeout !== undefined) {
      const timeoutId = setTimeout(() => {
        if (isResolved) {
          return;
        }
        log.trace('Graph canceled by timeout');
        cleanup();
        if (queueItemId !== null) {
          dependencies.executor.cancelQueueItem(queueItemId);
        }
        reject(new Error('Graph timed out'));
      }, timeout);

      cleanupFunctions.add(() => {
        clearTimeout(timeoutId);
      });
    }

    if (signal !== undefined) {
      const abortHandler = () => {
        if (isResolved) {
          return;
        }
        log.trace('Graph canceled by signal');
        cleanup();
        if (queueItemId !== null) {
          dependencies.executor.cancelQueueItem(queueItemId);
        }
        reject(new Error('Graph canceled'));
      };

      signal.addEventListener('abort', abortHandler);
      cleanupFunctions.add(() => {
        signal.removeEventListener('abort', abortHandler);
      });
    }

    const onQueueItemStatusChanged = async (event: S['QueueItemStatusChangedEvent']) => {
      if (isResolved) {
        return;
      }

      // Ignore events that are not for this graph
      if (event.origin !== origin) {
        return;
      }

      // Ignore events where the status is pending or in progress - no need to do anything for these
      if (event.status === 'pending' || event.status === 'in_progress') {
        return;
      }

      // The queue item is finished
      isResolved = true;
      cleanup();

      const queueItemResult = await withResultAsync(() => dependencies.executor.getQueueItem(event.item_id));
      if (queueItemResult.isErr()) {
        reject(queueItemResult.error);
        return;
      }

      const queueItem = queueItemResult.value;

      const { status, session, error_type, error_message, error_traceback } = queueItem;

      if (status === 'completed') {
        const getOutputResult = withResult(() => getOutputFromSession(session, outputNodeId));
        if (getOutputResult.isErr()) {
          reject(getOutputResult.error);
          return;
        }
        const output = getOutputResult.value;

        resolve({ session, output });
        return;
      }

      if (status === 'failed') {
        // We expect the event to have error details, but technically it's possible that it doesn't
        if (error_type && error_message && error_traceback) {
          reject(new QueueError(error_type, error_message, error_traceback));
          return;
        }

        // If we don't have error details, we can't provide a useful error message
        reject(new Error('Queue item failed, but no error details were provided'));
        return;
      }

      if (status === 'canceled') {
        reject(new Error('Graph canceled'));
        return;
      }
    };

    dependencies.eventHandler.subscribe(onQueueItemStatusChanged);
    cleanupFunctions.add(() => {
      dependencies.eventHandler.unsubscribe(onQueueItemStatusChanged);
    });

    // We are ready to enqueue the graph
    dependencies.executor
      .enqueueBatch(batch)
      .then((data) => {
        // We queue a single run of the batch, so we expect only one item_id in the response.
        assert(data.item_ids.length === 1);
        assert(data.item_ids[0] !== undefined, 'Enqueue result is missing first queue item id');
        queueItemId = data.item_ids[0];
      })
      .catch((error) => {
        if (!isResolved) {
          isResolved = true;
          cleanup();
          reject(error);
        }
      });
  });

  return promise;
};

const getOutputFromSession = (
  session: S['SessionQueueItem']['session'],
  nodeId: string
): S['SessionQueueItem']['session']['results'][string] => {
  const { results, source_prepared_mapping } = session;
  const preparedNodeId = source_prepared_mapping[nodeId]?.[0];
  if (!preparedNodeId) {
    throw new NodeNotFoundError(nodeId, session);
  }
  const result = results[preparedNodeId];
  if (!result) {
    throw new ResultNotFoundError(nodeId, session);
  }
  return result;
};

class NodeNotFoundError extends Error {
  session: S['SessionQueueItem']['session'];
  nodeId: string;

  constructor(nodeId: string, session: S['SessionQueueItem']['session']) {
    const availableNodes = Object.keys(session.source_prepared_mapping);
    super(`Node '${nodeId}' not found in session. Available nodes: ${availableNodes.join(', ')}`);
    this.name = this.constructor.name;
    this.session = session;
    this.nodeId = nodeId;
  }
}

class ResultNotFoundError extends Error {
  session: S['SessionQueueItem']['session'];
  nodeId: string;

  constructor(nodeId: string, session: S['SessionQueueItem']['session']) {
    const availableResults = Object.keys(session.results);
    super(`Result for node '${nodeId}' not found in session. Available results: ${availableResults.join(', ')}`);
    this.name = this.constructor.name;
    this.session = session;
    this.nodeId = nodeId;
  }
}
