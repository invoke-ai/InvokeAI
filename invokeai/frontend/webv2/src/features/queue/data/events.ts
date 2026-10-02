import type { QueueItemStatus, TerminalQueueItemStatus } from '@features/queue/core/types';
import type { QueueStatusCountsDTO } from '@features/queue/data/serverTypes';

/** Mirror snake_case backend Socket.IO payload contracts from events_common.py. */

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
  parent_item_id?: number | null;
  root_item_id?: number | null;
  /** The visible parent Call Saved Workflow node for child execution events. */
  workflow_call_parent_source_id?: string | null;
}

export interface InvocationStartedEvent extends InvocationEventBase {}

export interface InvocationProgressEvent extends InvocationEventBase {
  message: string;
  /** 0..1, or null for indeterminate progress. */
  percentage: number | null;
  /** Intermittent denoising preview, when the invocation produces one. */
  image?: { width: number; height: number; dataURL: string } | null;
  /**
   * Monotonic per queue item and session on image-bearing frames. Drop revisions at or below the latest accepted
   * frame so live and reconnect replay cannot move preview backward.
   */
  revision?: number | null;
  /** Device identifies concurrent accelerator sessions; null denotes CPU/MPS or single-device mode. */
  device?: string | null;
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

/** Encode local IDs in origins so reload reconciliation can re-adopt submitted backend items. */
const QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:';
const PROJECT_QUEUE_ITEM_ORIGIN_PREFIX = 'webv2:p:';

/**
 * Utility origins isolate one-shot graphs from project adoption and result routing. Parse util before generic
 * webv2 origins to prevent accidental local IDs.
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
  // Reject utility origins before the generic webv2 branch so project reconciliation cannot adopt them.
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
