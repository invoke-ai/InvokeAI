import type {
  QueueCompiledSubmission,
  QueueGraphSnapshot,
  QueueResultImage,
  QueueResultDestination,
  QueueSourceId,
  QueueSubmissionPresentation,
} from './types';

export type QueueHistoryItemStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

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
