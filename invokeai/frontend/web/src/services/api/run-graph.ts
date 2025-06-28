import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { withResult, withResultAsync } from 'common/util/result';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { QueueError } from 'services/events/errors';
import type { AppSocket } from 'services/events/types';
import { assert } from 'tsafe';

import { enqueueMutationFixedCacheKeyOptions, queueApi } from './endpoints/queue';
import type { EnqueueBatchArg, S } from './types';

const log = logger('queue');

type RunGraphArg = {
  graph: Graph;
  outputNodeId: string;
  store: AppStore;
  socket: AppSocket;
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
 * @param arg.store The Redux store to use for dispatching actions and accessing state.
 * @param arg.socket The socket to use for listening to events.
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
 * const graph = new Graph();
 * const outputNode = graph.addNode({ id: 'my-resize-node', type: 'img_resize', image: { image_name: 'my-image.png' } });
 * const controller = new AbortController();
 * const imageDTO = await this.manager.stateApi.runGraphAndReturnImageOutput({
 *  graph,
 *  outputNodeId: outputNode.id,
 *  prepend: true,
 *  signal: controller.signal,
 * });
 * // To cancel the operation:
 * controller.abort();
 * ```
 */
export const runGraph = (arg: RunGraphArg): Promise<RunGraphReturn> => {
  const promise = new Promise<RunGraphReturn>((resolve, reject) => {
    const { graph, outputNodeId, store, socket, destination, prepend, timeout, signal } = arg;

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
        func();
      }
    };

    if (timeout !== undefined) {
      const timeoutId = window.setTimeout(() => {
        if (isResolved) {
          return;
        }
        log.trace('Graph canceled by timeout');
        cleanup();
        if (queueItemId !== null) {
          cancelQueueItem(queueItemId, store);
        }
        reject(new Error('Graph timed out'));
      }, timeout);

      cleanupFunctions.add(() => {
        window.clearTimeout(timeoutId);
      });
    }

    if (signal !== undefined) {
      signal.addEventListener('abort', () => {
        if (isResolved) {
          return;
        }
        log.trace('Graph canceled by signal');
        cleanup();
        if (queueItemId !== null) {
          cancelQueueItem(queueItemId, store);
        }
        reject(new Error('Graph canceled'));
      });
      // TODO(psyche): Do we need to somehow clean up the signal? Not sure what is required here.
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

      const queueItemResult = await withResultAsync(() => getQueueItem(event.item_id, store));
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

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);
    cleanupFunctions.add(() => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    });

    // We are ready to enqueue the graph
    const enqueueRequest = store.dispatch(
      queueApi.endpoints.enqueueBatch.initiate(batch, {
        // Use the same cache key for all enqueueBatch requests, so that all consumers of this query get the same status
        // updates.
        ...enqueueMutationFixedCacheKeyOptions,
        // We do not need RTK to track this request in the store
        track: false,
      })
    );

    // Enqueue the graph and get the batch_id, updating the cancel graph callack. We need to do this in a .then() block
    // instead of awaiting the promise to avoid await-ing in a promise executor. Also need to catch any errors.
    enqueueRequest
      .unwrap()
      .then((data) => {
        // We queue a single run of the batch, so we expect only one item_id in the response.
        assert(data.item_ids.length === 1);
        assert(data.item_ids[0] !== undefined, 'Enqueue result is missing first queue item id');
        queueItemId = data.item_ids[0];
      })
      .catch((error) => {
        reject(error);
      });
  });

  return promise;
};

const getQueueItem = (queueItemId: number, store: AppStore): Promise<S['SessionQueueItem']> => {
  return store.dispatch(queueApi.endpoints.getQueueItem.initiate(queueItemId, { subscribe: false })).unwrap();
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

const cancelQueueItem = (queueItemId: number, store: AppStore): Promise<S['SessionQueueItem']> => {
  return store
    .dispatch(queueApi.endpoints.cancelQueueItem.initiate({ item_id: queueItemId }, { track: false }))
    .unwrap();
};

class NodeNotFoundError extends Error {
  session: S['SessionQueueItem']['session'];
  nodeId: string;

  constructor(nodeId: string, session: S['SessionQueueItem']['session']) {
    super();
    this.name = this.constructor.name;
    this.message = `Node '${nodeId}' not found in session.`;
    this.session = session;
    this.nodeId = nodeId;
  }
}

class ResultNotFoundError extends Error {
  session: S['SessionQueueItem']['session'];
  nodeId: string;

  constructor(nodeId: string, session: S['SessionQueueItem']['session']) {
    super();
    this.name = this.constructor.name;
    this.message = `Result for node '${nodeId}' not found in session.`;
    this.session = session;
    this.nodeId = nodeId;
  }
}
