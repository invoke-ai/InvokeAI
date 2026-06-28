/**
 * Typed contracts for the backend Socket.IO events webv2 consumes. Payload
 * shapes mirror the pydantic models in
 * `invokeai/app/services/events/events_common.py` (serialized as snake_case).
 */

/** Backend queue item lifecycle status. Note the backend spells it `canceled`. */
export type BackendQueueItemStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'canceled';

export type TerminalBackendQueueItemStatus = Extract<BackendQueueItemStatus, 'completed' | 'failed' | 'canceled'>;

export interface QueueItemEventBase {
  queue_id: string;
  item_id: number;
  batch_id: string;
  origin: string | null;
  destination: string | null;
}

export interface QueueItemStatusChangedEvent extends QueueItemEventBase {
  status: BackendQueueItemStatus;
  error_type: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  session_id: string;
}

export interface BatchEnqueuedEvent {
  queue_id: string;
  batch_id: string;
  enqueued: number;
  requested: number;
  priority: number;
  origin: string | null;
  user_id: string;
}

export interface QueueClearedEvent {
  queue_id: string;
}

export interface QueueItemsRetriedEvent {
  queue_id: string;
  retried_item_ids: number[];
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
  invocation_started: InvocationStartedEvent;
  invocation_progress: InvocationProgressEvent;
  invocation_complete: InvocationCompleteEvent;
  invocation_error: InvocationErrorEvent;
}

export const isTerminalBackendStatus = (status: BackendQueueItemStatus): status is TerminalBackendQueueItemStatus =>
  status === 'completed' || status === 'failed' || status === 'canceled';

/**
 * Queue items enqueued by webv2 carry the local queue item id in their origin
 * so that submissions survive a reload: on startup the backend queue is listed
 * and items are re-adopted by decoding their origin.
 */
const QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:';
const PROJECT_QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:p:';

export const buildProjectQueueItemOriginPrefix = (projectId: string): string =>
  `${PROJECT_QUEUE_ITEM_ORIGIN_PREFIX}${projectId}:q:`;

export const buildQueueItemOrigin = (localQueueItemId: string, projectId?: string): string =>
  projectId
    ? `${buildProjectQueueItemOriginPrefix(projectId)}${localQueueItemId}`
    : `${QUEUE_ITEM_ORIGIN_PREFIX}${localQueueItemId}`;

export const parseQueueItemOrigin = (origin: string | null | undefined): string | null =>
  origin?.startsWith(PROJECT_QUEUE_ITEM_ORIGIN_PREFIX)
    ? origin.slice(origin.lastIndexOf(':q:') + 3)
    : origin?.startsWith(QUEUE_ITEM_ORIGIN_PREFIX)
      ? origin.slice(QUEUE_ITEM_ORIGIN_PREFIX.length)
      : null;

export const parseQueueItemOriginProjectId = (origin: string | null | undefined): string | null => {
  if (!origin?.startsWith(PROJECT_QUEUE_ITEM_ORIGIN_PREFIX)) {
    return null;
  }

  const rest = origin.slice(PROJECT_QUEUE_ITEM_ORIGIN_PREFIX.length);
  const separatorIndex = rest.indexOf(':q:');

  return separatorIndex === -1 ? null : rest.slice(0, separatorIndex);
};
