import type {
  QueueCompiledSubmission,
  QueueGraphSnapshot,
  QueueResultImage,
  QueueResultDestination,
  QueueSourceId,
  QueueSubmissionPresentation,
} from './types';

export type QueueHistoryItemStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/** Failures the runtime words itself, so the notice can be translated; `error` keeps the English detail. */
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
  filterIntermediateResults: boolean;
  resultNodeIds?: readonly string[];
  presentation: QueueSubmissionPresentation;
  prepend?: boolean;
  submittedAt: string;
}

/** The latest failed attempt to submit a run the backend has not accepted yet; cleared once the backend accepts it. */
export interface QueueSubmissionRetry {
  /** `server-error`: 5xx but 502-504, spends the budget; `unavailable`: 408/429/502-504; `unreachable`: no answer. */
  reason: 'error' | 'server-error' | 'unavailable' | 'unreachable';
  /** Consecutive failed submissions so far, whatever their reason; at least 1. Paces the next attempt. */
  failureCount: number;
  serverErrorCount: number;
  /** Epoch ms of the first of those consecutive server errors; null when serverErrorCount is 0. */
  serverErrorSince: number | null;
  status: number | null;
  message: string;
  failedAt: number;
}

export interface QueueItem {
  id: string;
  status: QueueHistoryItemStatus;
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
