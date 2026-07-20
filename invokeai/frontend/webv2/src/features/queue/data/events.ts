import type { QueueItemStatus, TerminalQueueItemStatus } from '@features/queue/core/types';
import type { QueueStatusCountsDTO } from '@features/queue/data/serverTypes';

/**
 * Typed contracts for the backend Socket.IO events webv2 consumes. Payload
 * shapes mirror the pydantic models in
 * `invokeai/app/services/events/events_common.py` (serialized as snake_case).
 */

export interface QueueItemEventBase {
  queue_id: string;
  item_id: number;
  batch_id: string;
  origin: string | null;
  destination: string | null;
  timestamp: number;
  user_id: string;
}

export interface BatchStatusDTO {
  batch_id: string;
  canceled: number;
  completed: number;
  destination: string | null;
  failed: number;
  in_progress: number;
  origin: string | null;
  pending: number;
  queue_id: string;
  total: number;
  waiting: number;
}

export interface QueueItemStatusChangedEvent extends QueueItemEventBase {
  status: QueueItemStatus;
  error_type: string | null;
  error_message: string | null;
  error_traceback: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  session_id: string;
  status_sequence: number | null;
  batch_status: BatchStatusDTO;
  queue_status: QueueStatusCountsDTO;
}

export interface BatchEnqueuedEvent {
  queue_id: string;
  batch_id: string;
  enqueued: number;
  requested: number;
  priority: number;
  origin: string | null;
  user_id: string;
  timestamp: number;
}

export interface QueueClearedEvent {
  queue_id: string;
  timestamp: number;
  user_id: string | null;
}

export interface QueueItemsRetriedEvent {
  queue_id: string;
  retried_item_ids: number[];
  retried_item_ids_by_user: Record<string, number[]>;
  timestamp: number;
  user_ids: string[];
}

export interface QueueItemsCanceledEvent {
  canceled_item_ids: number[];
  canceled_item_ids_by_user: Record<string, number[]>;
  queue_id: string;
  timestamp: number;
  user_ids: string[];
}

/** Shared shape of per-invocation lifecycle events (`InvocationEventBase` on the backend). */
export interface InvocationEventBase extends QueueItemEventBase {
  session_id: string;
  /** The id of the executing invocation's source node — the editor's node id. */
  invocation_source_id: string;
}

export interface InvocationStartedEvent extends InvocationEventBase {}

export interface InvocationProgressEvent extends InvocationEventBase {
  message: string;
  /** 0..1, or null for indeterminate progress. */
  percentage: number | null;
  /** Intermittent denoising preview, when the invocation produces one. */
  image?: { width: number; height: number; dataURL: string } | null;
}

export interface InvocationCompleteEvent extends InvocationEventBase {
  /** The invocation's output, discriminated by its `type` (e.g. `image_output`). */
  result: { type: string } & Record<string, unknown>;
}

export interface InvocationErrorEvent extends InvocationEventBase {
  error_type: string;
  error_message: string;
}

export interface BackendSocketEvents {
  queue_item_status_changed: QueueItemStatusChangedEvent;
  batch_enqueued: BatchEnqueuedEvent;
  queue_cleared: QueueClearedEvent;
  queue_items_retried: QueueItemsRetriedEvent;
  queue_items_canceled: QueueItemsCanceledEvent;
  invocation_started: InvocationStartedEvent;
  invocation_progress: InvocationProgressEvent;
  invocation_complete: InvocationCompleteEvent;
  invocation_error: InvocationErrorEvent;
}

export const isTerminalBackendStatus = (status: QueueItemStatus): status is TerminalQueueItemStatus =>
  status === 'completed' || status === 'failed' || status === 'canceled';

/**
 * Queue items enqueued by webv2 carry the local queue item id in their origin
 * so that submissions survive a reload: on startup the backend queue is listed
 * and items are re-adopted by decoding their origin.
 */
const QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:';
const PROJECT_QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:p:';

/**
 * The origin prefix for utility-queue items — small graphs (filter previews,
 * SAM, …) enqueued OUTSIDE any project's queue and awaited directly via
 * `socketHub.on` (see `canvas-engine/backend/utilityQueue.ts`).
 *
 * The whole point of a distinct prefix is result isolation (plan Risk 4): a
 * utility item must never be mistaken for a project queue item and routed into
 * staging or the gallery. `parseQueueItemOrigin` therefore returns `null` for it
 * — so `queueCoordinator.reconcile` and `isQueueItemReadModelInProject` never map a
 * utility backend item to a local project item, and `routeQueueItemResults`
 * (only ever invoked for coordinator-tracked project runs, which utility items
 * are never registered as) never sees it. The `util:` segment sits under the
 * shared `webv2:` namespace but is checked BEFORE the generic branch below, so
 * it is not misparsed as a bare (non-project) local queue item id.
 */
const UTILITY_QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:util:';

export const buildProjectQueueItemOriginPrefix = (projectId: string): string =>
  `${PROJECT_QUEUE_ITEM_ORIGIN_PREFIX}${projectId}:q:`;

export const buildQueueItemOrigin = (localQueueItemId: string, projectId?: string): string =>
  projectId
    ? `${buildProjectQueueItemOriginPrefix(projectId)}${localQueueItemId}`
    : `${QUEUE_ITEM_ORIGIN_PREFIX}${localQueueItemId}`;

/** Builds the isolated origin for a utility-queue item (`webv2:util:<id>`). */
export const buildUtilityQueueItemOrigin = (utilityId: string): string =>
  `${UTILITY_QUEUE_ITEM_ORIGIN_PREFIX}${utilityId}`;

/** True when `origin` belongs to the utility queue (never a project/local queue item). */
export const isUtilityQueueItemOrigin = (origin: string | null | undefined): boolean =>
  origin?.startsWith(UTILITY_QUEUE_ITEM_ORIGIN_PREFIX) ?? false;

export const parseQueueItemOrigin = (origin: string | null | undefined): string | null => {
  // Utility items are intentionally invisible to project routing (Risk 4): they
  // resolve to no local queue item, so nothing adopts them or routes their
  // results. Checked first because `webv2:util:` also matches the generic
  // `webv2:` branch below.
  if (isUtilityQueueItemOrigin(origin)) {
    return null;
  }

  return origin?.startsWith(PROJECT_QUEUE_ITEM_ORIGIN_PREFIX)
    ? origin.slice(origin.lastIndexOf(':q:') + 3)
    : origin?.startsWith(QUEUE_ITEM_ORIGIN_PREFIX)
      ? origin.slice(QUEUE_ITEM_ORIGIN_PREFIX.length)
      : null;
};

export const parseQueueItemOriginProjectId = (origin: string | null | undefined): string | null => {
  if (!origin?.startsWith(PROJECT_QUEUE_ITEM_ORIGIN_PREFIX)) {
    return null;
  }

  const rest = origin.slice(PROJECT_QUEUE_ITEM_ORIGIN_PREFIX.length);
  const separatorIndex = rest.indexOf(':q:');

  return separatorIndex === -1 ? null : rest.slice(0, separatorIndex);
};
