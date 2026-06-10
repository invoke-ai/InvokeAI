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

export interface InvocationProgressEvent extends QueueItemEventBase {
  session_id: string;
  message: string;
  /** 0..1, or null for indeterminate progress. */
  percentage: number | null;
}

export interface BackendSocketEvents {
  queue_item_status_changed: QueueItemStatusChangedEvent;
  invocation_progress: InvocationProgressEvent;
}

export const isTerminalBackendStatus = (status: BackendQueueItemStatus): status is TerminalBackendQueueItemStatus =>
  status === 'completed' || status === 'failed' || status === 'canceled';

/**
 * Queue items enqueued by webv2 carry the local queue item id in their origin
 * so that submissions survive a reload: on startup the backend queue is listed
 * and items are re-adopted by decoding their origin.
 */
const QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:';

export const buildQueueItemOrigin = (localQueueItemId: string): string =>
  `${QUEUE_ITEM_ORIGIN_PREFIX}${localQueueItemId}`;

export const parseQueueItemOrigin = (origin: string | null | undefined): string | null =>
  origin?.startsWith(QUEUE_ITEM_ORIGIN_PREFIX) ? origin.slice(QUEUE_ITEM_ORIGIN_PREFIX.length) : null;
