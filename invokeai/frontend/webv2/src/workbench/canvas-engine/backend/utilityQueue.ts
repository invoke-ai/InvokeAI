/**
 * The utility queue: fire-and-await small graphs OUTSIDE any project's queue.
 *
 * Filter previews and (later) Segment-Anything need to run a one-shot graph and
 * read back its single output image WITHOUT the result ever touching project
 * staging or the gallery. {@link runUtilityGraph} does exactly that:
 *
 * 1. Mints a fresh `webv2:util:<uuid>` origin (see `backend/events.ts`). That
 *    origin is deliberately invisible to project routing — `parseQueueItemOrigin`
 *    returns `null` for it, so `queueCoordinator.reconcile` /
 *    `isQueueServerItemInProject` never adopt it and `routeQueueItemResults`
 *    (only invoked for coordinator-tracked project runs, which a utility item is
 *    never registered as) never sees it. This is the plan's Risk-4 guard.
 * 2. Attaches raw `socketHub.on` listeners (they survive socket recreation) for
 *    `invocation_complete` (to capture the output image name) and
 *    `queue_item_status_changed` (to settle on the terminal status), matching
 *    events by our unique origin.
 * 3. Enqueues the graph (listeners are attached first, closing the fast-finish
 *    race), then resolves with the output `imageName` on completion, or rejects
 *    on failure/cancellation/timeout/abort.
 *
 * Zero React, zero DOM: `hub` and `enqueue` are injected, so this runs in node
 * tests against fakes. Every side-effecting dependency is a parameter.
 */

import type { InvocationCompleteEvent, QueueItemStatusChangedEvent } from '@workbench/backend/events';
import type { SocketHub } from '@workbench/backend/socketHub';
import type { BackendGraphContract } from '@workbench/types';

import { buildUtilityQueueItemOrigin } from '@workbench/backend/events';
import { enqueueUtilityGraph } from '@workbench/generation/api';

/** The default time a utility graph may run before it is abandoned. */
export const DEFAULT_UTILITY_QUEUE_TIMEOUT_MS = 120_000;

/** Thrown when a utility graph fails, is canceled, times out, or is aborted. */
export class UtilityQueueError extends Error {
  readonly reason: 'failed' | 'canceled' | 'timeout' | 'aborted' | 'no-output' | 'enqueue';

  constructor(reason: UtilityQueueError['reason'], message: string) {
    super(message);
    this.name = 'UtilityQueueError';
    this.reason = reason;
  }
}

/** The enqueue seam: posts the graph under `origin`, resolving to its backend item ids. */
export type UtilityEnqueue = (request: {
  graph: BackendGraphContract;
  origin: string;
}) => Promise<{ itemIds: number[]; enqueued: number }>;

/** Dependencies for {@link runUtilityGraph} (all injectable for tests). */
export interface RunUtilityGraphOptions {
  /** The graph to enqueue and await. */
  graph: BackendGraphContract;
  /**
   * The source node id whose `invocation_complete` output image is the result.
   * When omitted, the first image-bearing completion for our origin is used.
   */
  outputNodeId?: string;
  /** The socket hub (only `on` is used). Raw listeners survive socket recreation. */
  hub: Pick<SocketHub, 'on'>;
  /** Enqueue seam (defaults to the real utility enqueue API). */
  enqueue?: UtilityEnqueue;
  /** Abandon after this many ms (default {@link DEFAULT_UTILITY_QUEUE_TIMEOUT_MS}). `0` disables. */
  timeoutMs?: number;
  /** Cancels the await (does not itself cancel the backend item). */
  signal?: AbortSignal;
  /** Injectable id source (defaults to `crypto.randomUUID`). */
  createId?: () => string;
}

/** The resolved result of a utility graph: its single output image. */
export interface UtilityGraphResult {
  imageName: string;
  /** The origin used, for diagnostics/tests. */
  origin: string;
}

/** Extracts an image name from an `invocation_complete` result payload, if present. */
const extractImageName = (result: InvocationCompleteEvent['result'] | undefined): string | null => {
  const image = (result as { image?: { image_name?: unknown } } | undefined)?.image;
  return typeof image?.image_name === 'string' ? image.image_name : null;
};

/**
 * Runs `graph` on the utility queue and resolves with its output image name.
 * Never routes into project state (isolated origin). Rejects with a
 * {@link UtilityQueueError} on failure/cancel/timeout/abort/enqueue error.
 */
export const runUtilityGraph = (options: RunUtilityGraphOptions): Promise<UtilityGraphResult> => {
  const { graph, hub, outputNodeId, signal } = options;
  const enqueue = options.enqueue ?? enqueueUtilityGraph;
  const timeoutMs = options.timeoutMs ?? DEFAULT_UTILITY_QUEUE_TIMEOUT_MS;
  const utilityId = (options.createId ?? (() => crypto.randomUUID()))();
  const origin = buildUtilityQueueItemOrigin(utilityId);

  return new Promise<UtilityGraphResult>((resolve, reject) => {
    let settled = false;
    let capturedImageName: string | null = null;
    const detachers: Array<() => void> = [];
    let timer: ReturnType<typeof setTimeout> | null = null;

    const cleanup = (): void => {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      for (const detach of detachers) {
        detach();
      }
      detachers.length = 0;
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
    };

    const settleResolve = (imageName: string): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve({ imageName, origin });
    };

    const settleReject = (error: UtilityQueueError): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
    };

    function onAbort(): void {
      settleReject(new UtilityQueueError('aborted', 'The utility graph was aborted.'));
    }

    // Attach listeners BEFORE enqueue so a very fast completion is not missed.
    detachers.push(
      hub.on('invocation_complete', (event: InvocationCompleteEvent) => {
        if (event.origin !== origin) {
          return;
        }
        // Only take the target node's output (or, when unspecified, any image).
        if (outputNodeId && event.invocation_source_id !== outputNodeId) {
          return;
        }
        const imageName = extractImageName(event.result);
        if (imageName) {
          capturedImageName = imageName;
        }
      })
    );

    detachers.push(
      hub.on('queue_item_status_changed', (event: QueueItemStatusChangedEvent) => {
        if (event.origin !== origin) {
          return;
        }
        if (event.status === 'completed') {
          if (capturedImageName) {
            settleResolve(capturedImageName);
          } else {
            settleReject(new UtilityQueueError('no-output', 'The utility graph produced no output image.'));
          }
        } else if (event.status === 'failed') {
          settleReject(new UtilityQueueError('failed', event.error_message ?? 'The utility graph failed.'));
        } else if (event.status === 'canceled') {
          settleReject(new UtilityQueueError('canceled', 'The utility graph was canceled.'));
        }
      })
    );

    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort);
    }

    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        settleReject(new UtilityQueueError('timeout', `The utility graph timed out after ${timeoutMs}ms.`));
      }, timeoutMs);
    }

    // Enqueue last. A synchronous listener callback could already have fired by
    // the time this resolves; `settled` guards against a late enqueue result.
    enqueue({ graph, origin })
      .then((result) => {
        if (!settled && result.enqueued === 0) {
          settleReject(new UtilityQueueError('enqueue', 'The backend queue did not accept the utility graph.'));
        }
      })
      .catch((error: unknown) => {
        settleReject(
          new UtilityQueueError('enqueue', error instanceof Error ? error.message : 'Failed to enqueue utility graph.')
        );
      });
  });
};
