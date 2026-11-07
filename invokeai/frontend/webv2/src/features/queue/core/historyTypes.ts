import type {
  QueueCompiledSubmission,
  QueueGraphSnapshot,
  QueueResultImage,
  QueueResultDestination,
  QueueSourceId,
  QueueSubmissionPresentation,
} from './types';

export type QueueHistoryItemStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Why a run failed, for failures the queue runtime words itself, so the notice can show translated text. The English
 * `error` stays alongside as diagnostic detail; server and validation text is shown as the notice's reason.
 * - `invalid-submission`: the queued request is missing or malformed, so it was never sent.
 * - `not-accepted`: the server queue took none of the run's items.
 * - `project-not-saved`: the run's project cannot be saved, so it was not sent.
 * - `removed`: the run's backend items vanished from the server queue.
 * - `server-error`: enqueue kept failing with a server error until the runtime gave up.
 * - `unexplained`: the backend failed the run without a reason.
 */
export type QueueRunErrorCode =
  | 'invalid-submission'
  | 'not-accepted'
  | 'project-not-saved'
  | 'removed'
  | 'server-error'
  | 'unexplained';

export interface QueueSubmissionSnapshot {
  backendSubmission: QueueCompiledSubmission;
  sourceId: QueueSourceId;
  destination: QueueResultDestination;
  graph: QueueGraphSnapshot;
  /** Gallery board captured at enqueue time; null means uncategorized. */
  galleryBoardId: string | null;
  /** Workflow outputs omit intermediate node images. */
  filterIntermediateResults: boolean;
  resultNodeIds?: readonly string[];
  presentation: QueueSubmissionPresentation;
  /** Set by Invoke Front: the backend runs the batch ahead of the user's other pending work. */
  prepend?: boolean;
  submittedAt: string;
}

/** The latest failed attempt to submit a run the backend has not accepted yet; cleared once the backend accepts it. */
export interface QueueSubmissionRetry {
  /**
   * 'server-error': a 5xx other than 502/503/504, counted toward the failure budget. 'unavailable': 408, 429, 502,
   * 503 or 504. 'unreachable': the request never got an answer (network failure or timeout). 'error': anything else,
   * such as an answer that could not be read; the batch may have been accepted, so the run is reconciled and retried.
   */
  reason: 'error' | 'server-error' | 'unavailable' | 'unreachable';
  /** Consecutive failed submissions so far, whatever their reason; at least 1. Paces the next attempt. */
  failureCount: number;
  /** Consecutive server-error answers so far; 0 when the latest failure was another reason. */
  serverErrorCount: number;
  /** Epoch ms of the first of those consecutive server errors; null when serverErrorCount is 0. */
  serverErrorSince: number | null;
  /** HTTP status of the latest failure; null without a response. */
  status: number | null;
  /** Message of the latest failure, as shown to the user. */
  message: string;
  /** Epoch ms of the latest failure. */
  failedAt: number;
}

export interface QueueItem {
  id: string;
  status: QueueHistoryItemStatus;
  /** Whether this run can still exist in durable browser recovery storage. */
  localRecoveryState?: 'durable' | 'local-only' | 'uncertain';
  cancellationPending?: boolean;
  cancellable: boolean;
  snapshot: QueueSubmissionSnapshot;
  backendItemIds?: number[];
  completedBackendItemIds?: number[];
  cancelledBackendItemIds?: number[];
  /** Backend items a retry reopened this run away from; retrying another of them later still returns here. */
  retriedBackendItemIds?: number[];
  backendBatchId?: string;
  error?: string;
  /** Set while the backend has not accepted the run and its latest submission failed. */
  submissionRetry?: QueueSubmissionRetry;
  resultImages?: QueueResultImage[];
  /** Videos already routed to the Gallery; a repeat routing pass must not reselect them. */
  resultVideoNames?: string[];
}

export interface QueueState {
  items: QueueItem[];
}

export interface RunRecord {
  id: string;
  projectId: string;
  queueItemId: string;
  sourceId: QueueSourceId;
  destination: QueueResultDestination;
  graphSnapshotId: string;
  status: QueueHistoryItemStatus;
  submittedAt: string;
  completedAt?: string;
}
