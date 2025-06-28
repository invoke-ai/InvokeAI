import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { withResult, withResultAsync } from 'common/util/result';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { QueueError } from 'services/events/errors';
import type { AppSocket } from 'services/events/types';
import type { Equals } from 'tsafe';
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
  pollingInterval?: number; // Optional polling interval for checking the queue item status
};

type RunGraphReturn = {
  session: S['SessionQueueItem']['session'];
  output: S['GraphExecutionState']['results'][string];
};

/**
 * Run a graph and return an image output. The specified output node must return an image output, else the promise
 * will reject with an error.
 *
 * @param arg The arguments for the function.
 * @param arg.graph The graph to execute.
 * @param arg.outputNodeId The id of the node whose output will be retrieved.
 * @param arg.destination The destination to assign to the batch. If omitted, the destination is not set.
 * @param arg.prepend Whether to prepend the graph to the front of the queue. If omitted, the graph is appended to the end of the queue.
 * @param arg.timeout The timeout for the batch. If omitted, there is no timeout.
 * @param arg.signal An optional signal to cancel the operation. If omitted, the operation cannot be canceled!
 *
 * @returns A promise that resolves to the image output or rejects with an error.
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
  const { graph, outputNodeId, destination, prepend, timeout, signal, store, socket, pollingInterval } = arg;

  if (!graph.hasNode(outputNodeId)) {
    throw new Error(`Graph does not contain node with id: ${outputNodeId}`);
  }

  /**
   * We will use the origin to handle events from the graph. Ideally we'd just use the queue item's id, but there's a
   * race condition:
   * - The queue item id is not available until the graph is enqueued
   * - The graph may complete before we can set up the listeners to handle the completion event
   *
   * The origin is the only unique identifier we have that is guaranteed to be available before the graph is enqueued,
   * so we will use that to filter events.
   */
  const origin = getPrefixedId(graph.id);

  const batch: EnqueueBatchArg = {
    prepend,
    batch: {
      graph: graph.getGraph(),
      origin,
      destination,
      runs: 1,
    },
  };

  const promise = new Promise<RunGraphReturn>((resolve, reject) => {
    /**
     * Track execution state.
     */
    let didSuceed = false;
    /**
     * The queue item id is set to null initially, but will be updated once the graph is enqueued.
     */
    let queueItemId: number | null = null;
    /**
     * If a timeout is provided, we will cancel the graph if it takes too long - but we need a way to clear the timeout
     * if the graph completes or errors before the timeout.
     */
    let timeoutId: number | null = null;

    let pollingIntervalId: number | null = null;

    const queueItemStatusChangedHandler = async (event: S['QueueItemStatusChangedEvent']) => {
      // Ignore events that are not for this graph
      if (event.origin !== origin) {
        return;
      }

      // Ignore events where the status is pending or in progress - no need to do anything for these
      if (event.status === 'pending' || event.status === 'in_progress') {
        return;
      }

      // Once we get here, the event is for the correct graph and the status is either 'completed', 'failed', or 'canceled'.
      cleanup();

      if (event.status === 'completed') {
        const queueItemResult = await withResultAsync(() => getQueueItem(event.item_id, store));
        if (queueItemResult.isErr()) {
          reject(queueItemResult.error);
          return;
        }
        const queueItem = queueItemResult.value;
        const { session } = queueItem;
        const getOutputResult = withResult(() => getOutputFromSession(session, outputNodeId));
        if (getOutputResult.isErr()) {
          reject(getOutputResult.error);
          return;
        }
        const output = getOutputResult.value;

        didSuceed = true;
        resolve({ session, output });
        return;
      }

      if (event.status === 'failed') {
        // We expect the event to have error details, but technically it's possible that it doesn't
        const { error_type, error_message, error_traceback } = event;
        if (error_type && error_message && error_traceback) {
          reject(new QueueError(error_type, error_message, error_traceback));
        } else {
          reject(new Error('Queue item failed, but no error details were provided'));
        }
        return;
      }

      if (event.status === 'canceled') {
        reject(new Error('Graph canceled'));
        return;
      }

      assert<Equals<never, typeof event.status>>(false);
    };

    if (pollingInterval !== undefined) {
      const pollForResult = async () => {
        if (queueItemId === null) {
          return;
        }
        const _queueItemId = queueItemId;
        const getQueueItemResult = await withResultAsync(() => getQueueItem(_queueItemId, store));
        if (getQueueItemResult.isErr()) {
          reject(getQueueItemResult.error);
          return;
        }
        const queueItem = getQueueItemResult.value;
        if (queueItem.status === 'pending' || queueItem.status === 'in_progress') {
          return;
        }

        cleanup();

        const { session } = queueItem;
        const getOutputResult = withResult(() => getOutputFromSession(session, outputNodeId));
        if (getOutputResult.isErr()) {
          reject(getOutputResult.error);
          return;
        }
        const output = getOutputResult.value;
        didSuceed = true;
        resolve({ session, output });
        return;
      };

      pollingIntervalId = window.setInterval(pollForResult, pollingInterval);
    }

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

    socket.on('queue_item_status_changed', queueItemStatusChangedHandler);

    const _cleanupTimeout = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };
    const _cleanupPollingInterval = () => {
      if (pollingIntervalId !== null) {
        window.clearInterval(pollingIntervalId);
        pollingIntervalId = null;
      }
    };
    const _cleanupListeners = () => {
      socket.off('queue_item_status_changed', queueItemStatusChangedHandler);
    };

    const cleanup = () => {
      _cleanupTimeout();
      _cleanupPollingInterval();
      _cleanupListeners();
    };

    if (timeout) {
      timeoutId = window.setTimeout(() => {
        if (didSuceed) {
          // If we already succeeded, we don't need to do anything
          return;
        }
        log.trace('Graph canceled by timeout');
        cleanup();
        if (queueItemId !== null) {
          cancelQueueItem(queueItemId, store);
        }
        reject(new Error('Graph timed out'));
      }, timeout);
    }

    if (signal) {
      signal.addEventListener('abort', () => {
        if (didSuceed) {
          // If we already succeeded, we don't need to do anything
          return;
        }
        log.trace('Graph canceled by signal');
        cleanup();
        if (queueItemId !== null) {
          cancelQueueItem(queueItemId, store);
        }
        reject(new Error('Graph canceled'));
      });
    }
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
