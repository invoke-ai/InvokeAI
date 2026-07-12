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
 *    `invocation_complete` (to capture the output image name and dimensions) and
 *    `queue_item_status_changed` (to settle on the terminal status), matching
 *    events by our unique origin.
 * 3. Enqueues the graph (listeners are attached first, closing the fast-finish
 *    race), then resolves with the output image metadata on completion, or rejects
 *    on failure/cancellation/timeout/abort. Abort and timeout also best-effort
 *    cancel every backend item accepted for this run, including when enqueue
 *    resolves after the local await has already rejected.
 *
 * Zero React, zero DOM: `hub` and `enqueue` are injected, so this runs in node
 * tests against fakes. Every side-effecting dependency is a parameter.
 */

import type { InvocationCompleteEvent, QueueItemStatusChangedEvent } from '@workbench/backend/events';
import type { SocketHub } from '@workbench/backend/socketHub';
import type { BackendGraphContract } from '@workbench/types';

import { buildUtilityQueueItemOrigin } from '@workbench/backend/events';
import { ApiError } from '@workbench/backend/http';
import { cancelQueueItems, enqueueUtilityGraph, getQueueItem } from '@workbench/generation/api';

/** The default time a utility graph may run before it is abandoned. */
export const DEFAULT_UTILITY_QUEUE_TIMEOUT_MS = 120_000;

/** Bounded retries for transient completed-item reconciliation failures. */
export const DEFAULT_UTILITY_RECONCILE_RETRY_POLICY = { delayMs: 100, maxAttempts: 3 } as const;

/** Thrown when a utility graph fails, is canceled, times out, or is aborted. */
export class UtilityQueueError extends Error {
  readonly reason: 'failed' | 'canceled' | 'timeout' | 'aborted' | 'no-output' | 'enqueue' | 'reconcile';

  constructor(reason: UtilityQueueError['reason'], message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'UtilityQueueError';
    this.reason = reason;
  }
}

/** The enqueue seam: posts the graph under `origin`, resolving to its backend item ids. */
export type UtilityEnqueue = (request: {
  graph: BackendGraphContract;
  origin: string;
}) => Promise<{ itemIds: number[]; enqueued: number }>;

/** The cancellation seam: best-effort cancels accepted backend item ids. */
export type UtilityCancel = (itemIds: number[]) => Promise<void>;

type UtilityImageOutput = { height: number; imageName: string; width: number };

/** Deterministically reads a completed queue item's target output. */
export type UtilityCompletedOutputReconciler = (
  itemIds: number[],
  outputNodeId?: string
) => Promise<UtilityImageOutput | null>;

export interface UtilityReconcileRetryPolicy {
  delayMs: number;
  maxAttempts: number;
}

/** Returns a cancellation function for one scheduled reconciliation retry. */
export type UtilityReconcileRetryScheduler = (callback: () => void, delayMs: number) => () => void;

export type UtilityReconcileErrorClassifier = (cause: unknown) => 'retry' | 'fail';

export const classifyUtilityReconcileError: UtilityReconcileErrorClassifier = (cause) => {
  if (cause instanceof TypeError) {
    return 'retry';
  }
  if (cause instanceof ApiError) {
    return cause.status === 408 || cause.status === 429 || (cause.status >= 500 && cause.status <= 599)
      ? 'retry'
      : 'fail';
  }
  return 'fail';
};

const scheduleUtilityReconcileRetry: UtilityReconcileRetryScheduler = (callback, delayMs) => {
  const timer = setTimeout(callback, delayMs);
  return () => clearTimeout(timer);
};

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
  /** Cancellation seam (defaults to the real queue cancellation API). */
  cancel?: UtilityCancel;
  /** Abandon after this many ms (default {@link DEFAULT_UTILITY_QUEUE_TIMEOUT_MS}). `0` disables. */
  timeoutMs?: number;
  /** Cancels the await and best-effort cancels accepted backend items. */
  signal?: AbortSignal;
  /** Injectable id source (defaults to `crypto.randomUUID`). */
  createId?: () => string;
  /** Injectable completed-item reconciliation seam. */
  reconcileCompletedOutput?: UtilityCompletedOutputReconciler;
  /** Bounded reconciliation retry policy. */
  reconcileRetryPolicy?: UtilityReconcileRetryPolicy;
  /** Injectable reconciliation retry timer seam. */
  scheduleReconcileRetry?: UtilityReconcileRetryScheduler;
  /** Injectable reconciliation failure classifier. */
  classifyReconcileError?: UtilityReconcileErrorClassifier;
}

/** The resolved result of a utility graph: its single output image. */
export interface UtilityGraphResult {
  height: number;
  imageName: string;
  /** The origin used, for diagnostics/tests. */
  origin: string;
  width: number;
}

/** Extracts the current backend ImageOutput fields from a completion payload. */
const extractImageOutput = (result: InvocationCompleteEvent['result'] | undefined): UtilityImageOutput | null => {
  const output = result as { height?: unknown; image?: { image_name?: unknown }; width?: unknown } | undefined;
  if (
    typeof output?.image?.image_name !== 'string' ||
    typeof output.width !== 'number' ||
    !Number.isFinite(output.width) ||
    !Number.isInteger(output.width) ||
    output.width <= 0 ||
    typeof output.height !== 'number' ||
    !Number.isFinite(output.height) ||
    !Number.isInteger(output.height) ||
    output.height <= 0
  ) {
    return null;
  }
  return { height: output.height, imageName: output.image.image_name, width: output.width };
};

export const reconcileUtilityCompletedOutput: UtilityCompletedOutputReconciler = async (itemIds, outputNodeId) => {
  for (const itemId of itemIds) {
    const item = await getQueueItem(itemId);
    if (
      !item ||
      typeof item !== 'object' ||
      item.item_id !== itemId ||
      !item.session ||
      typeof item.session !== 'object' ||
      !item.session.results ||
      typeof item.session.results !== 'object' ||
      Array.isArray(item.session.results) ||
      (item.session.prepared_source_mapping !== undefined &&
        (typeof item.session.prepared_source_mapping !== 'object' ||
          item.session.prepared_source_mapping === null ||
          Array.isArray(item.session.prepared_source_mapping)))
    ) {
      throw new Error(`Queue item ${itemId} returned malformed reconciliation data.`);
    }
    const results = item.session?.results ?? {};
    const preparedSourceMapping = item.session?.prepared_source_mapping ?? {};
    for (const [preparedNodeId, result] of Object.entries(results)) {
      if (outputNodeId && (preparedSourceMapping[preparedNodeId] ?? preparedNodeId) !== outputNodeId) {
        continue;
      }
      const output = extractImageOutput(result as InvocationCompleteEvent['result']);
      if (output) {
        return output;
      }
      if (outputNodeId) {
        throw new Error(`Queue item ${itemId} returned malformed output for node ${outputNodeId}.`);
      }
    }
  }
  return null;
};

/**
 * Runs `graph` on the utility queue and resolves with its output image metadata.
 * Never routes into project state (isolated origin). Rejects with a
 * {@link UtilityQueueError} on failure/cancel/timeout/abort/enqueue error.
 */
export const runUtilityGraph = (options: RunUtilityGraphOptions): Promise<UtilityGraphResult> => {
  const { graph, hub, outputNodeId, signal } = options;
  const enqueue = options.enqueue ?? enqueueUtilityGraph;
  const cancel = options.cancel ?? cancelQueueItems;
  const timeoutMs = options.timeoutMs ?? DEFAULT_UTILITY_QUEUE_TIMEOUT_MS;
  const reconcileCompletedOutput = options.reconcileCompletedOutput ?? reconcileUtilityCompletedOutput;
  const reconcileRetryPolicy = options.reconcileRetryPolicy ?? DEFAULT_UTILITY_RECONCILE_RETRY_POLICY;
  const scheduleReconcileRetry = options.scheduleReconcileRetry ?? scheduleUtilityReconcileRetry;
  const classifyReconcileError = options.classifyReconcileError ?? classifyUtilityReconcileError;
  const reconcileMaxAttempts = Number.isFinite(reconcileRetryPolicy.maxAttempts)
    ? Math.max(1, Math.floor(reconcileRetryPolicy.maxAttempts))
    : DEFAULT_UTILITY_RECONCILE_RETRY_POLICY.maxAttempts;
  const reconcileDelayMs = Number.isFinite(reconcileRetryPolicy.delayMs)
    ? Math.max(0, reconcileRetryPolicy.delayMs)
    : DEFAULT_UTILITY_RECONCILE_RETRY_POLICY.delayMs;
  const utilityId = (options.createId ?? (() => crypto.randomUUID()))();
  const origin = buildUtilityQueueItemOrigin(utilityId);

  return new Promise<UtilityGraphResult>((resolve, reject) => {
    let settled = false;
    let capturedOutput: { height: number; imageName: string; width: number } | null = null;
    let cancellationRequested = false;
    let cancellationStarted = false;
    let enqueueResult: { itemIds: number[]; enqueued: number } | null = null;
    let completionReceived = false;
    let reconciliationAttempts = 0;
    let reconciliationInFlight = false;
    let cancelReconciliationRetry: (() => void) | null = null;
    const detachers: Array<() => void> = [];
    let timer: ReturnType<typeof setTimeout> | null = null;

    const cancelAcceptedItems = (): void => {
      if (
        !cancellationRequested ||
        cancellationStarted ||
        enqueueResult === null ||
        enqueueResult.enqueued === 0 ||
        enqueueResult.itemIds.length === 0
      ) {
        return;
      }

      cancellationStarted = true;
      const itemIds = [...enqueueResult.itemIds];
      void Promise.resolve()
        .then(() => cancel(itemIds))
        .catch(() => {
          // Cancellation is best-effort and must never mask the original settle reason.
        });
    };

    const requestBackendCancellation = (): void => {
      cancellationRequested = true;
      cancelAcceptedItems();
    };

    const cleanup = (): void => {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      cancelReconciliationRetry?.();
      cancelReconciliationRetry = null;
      for (const detach of detachers) {
        detach();
      }
      detachers.length = 0;
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
    };

    const settleResolve = (output: { height: number; imageName: string; width: number }): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve({ ...output, origin });
    };

    const settleReject = (error: UtilityQueueError): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
    };

    const reconcileCompletion = (): void => {
      if (
        settled ||
        !completionReceived ||
        reconciliationInFlight ||
        cancelReconciliationRetry !== null ||
        enqueueResult === null
      ) {
        return;
      }
      if (capturedOutput) {
        settleResolve(capturedOutput);
        return;
      }
      reconciliationAttempts += 1;
      reconciliationInFlight = true;
      const reconciliationItemIds = [...enqueueResult.itemIds];
      const scheduleNextReconciliation = (): void => {
        cancelReconciliationRetry = scheduleReconcileRetry(() => {
          cancelReconciliationRetry = null;
          reconcileCompletion();
        }, reconcileDelayMs);
      };
      void Promise.resolve()
        .then(() => reconcileCompletedOutput(reconciliationItemIds, outputNodeId))
        .then((output) => {
          reconciliationInFlight = false;
          if (settled) {
            return;
          }
          if (capturedOutput) {
            settleResolve(capturedOutput);
          } else if (output) {
            settleResolve(output);
          } else if (reconciliationAttempts >= reconcileMaxAttempts) {
            settleReject(new UtilityQueueError('no-output', 'The utility graph produced no output image.'));
          } else {
            scheduleNextReconciliation();
          }
        })
        .catch((cause: unknown) => {
          reconciliationInFlight = false;
          if (settled || capturedOutput) {
            if (capturedOutput) {
              settleResolve(capturedOutput);
            }
            return;
          }
          if (classifyReconcileError(cause) === 'fail' || reconciliationAttempts >= reconcileMaxAttempts) {
            const detail = cause instanceof Error ? cause.message : String(cause);
            settleReject(
              new UtilityQueueError(
                'reconcile',
                `Failed to reconcile utility graph output after ${reconciliationAttempts} attempts: ${detail}`,
                cause
              )
            );
            return;
          }
          scheduleNextReconciliation();
        });
    };

    function onAbort(): void {
      if (settled) {
        return;
      }
      requestBackendCancellation();
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
        const output = extractImageOutput(event.result);
        if (output) {
          capturedOutput = output;
          if (completionReceived) {
            settleResolve(output);
          }
        }
      })
    );

    detachers.push(
      hub.on('queue_item_status_changed', (event: QueueItemStatusChangedEvent) => {
        if (event.origin !== origin) {
          return;
        }
        if (event.status === 'completed') {
          completionReceived = true;
          reconcileCompletion();
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
        if (settled) {
          return;
        }
        requestBackendCancellation();
        settleReject(new UtilityQueueError('timeout', `The utility graph timed out after ${timeoutMs}ms.`));
      }, timeoutMs);
    }

    // Enqueue last. A synchronous listener callback could already have fired by
    // the time this resolves; `settled` guards against a late enqueue result.
    enqueue({ graph, origin })
      .then((result) => {
        enqueueResult = result;
        cancelAcceptedItems();
        if (!settled && result.enqueued === 0) {
          settleReject(new UtilityQueueError('enqueue', 'The backend queue did not accept the utility graph.'));
        } else {
          reconcileCompletion();
        }
      })
      .catch((error: unknown) => {
        settleReject(
          new UtilityQueueError('enqueue', error instanceof Error ? error.message : 'Failed to enqueue utility graph.')
        );
      });
  });
};
