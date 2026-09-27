import type {
  QueuePromptSeedBehaviour,
  QueueSeedStep,
  QueueWorkflowBatchDatum,
  QueueWorkflowSeed,
} from '@features/queue/core/promptBatch';

export type { QueueSeedStep, QueueWorkflowBatchDatum, QueueWorkflowSeed };

export interface QueueBackendInvocation {
  id: string;
  type: string;
  [key: string]: unknown;
}

export interface QueueBackendGraphEdge {
  destination: { field: string; node_id: string };
  source: { field: string; node_id: string };
}

export interface QueueBackendGraph {
  edges: QueueBackendGraphEdge[];
  id: string;
  nodes: Record<string, QueueBackendInvocation>;
}

export type QueueResultDestination = 'canvas' | 'gallery';
export type QueueSourceId = 'canvas' | 'generate' | 'upscale' | 'video' | 'workflow';

export interface QueueSubmissionPresentation {
  batchCount: number;
  height: number;
  positivePrompt?: string;
  width: number;
}

export interface QueueGraphSnapshot {
  id: string;
  label: string;
}

interface QueueEnqueueRequestBase {
  batchCount: number;
  destination: QueueResultDestination;
  graph: QueueBackendGraph;
  projectId: string;
  sourceQueueItemId: string;
}

export interface QueueEnqueueWorkflowRequest extends QueueEnqueueRequestBase {
  /** Serialized parent workflow, compatible with the backend's `WorkflowWithoutID` schema. */
  workflow?: Record<string, unknown>;
  /** The seed inputs that vary between the `batchCount` runs; the graph carries each one's first seed. */
  seeds?: QueueWorkflowSeed[];
  /** Batch-node value lists: outer groups multiply, inner datums zip. */
  batchData?: QueueWorkflowBatchDatum[][];
}

export interface QueueEnqueueGenerateRequest extends QueueEnqueueRequestBase {
  negativePrompt: string;
  negativePromptNodeId: string;
  positivePrompt: string;
  positivePromptNodeId: string;
  /**
   * Prompts to submit as a batch dimension, already expanded by the caller.
   * Absent or empty submits `positivePrompt` verbatim.
   */
  positivePrompts?: string[];
  seed: number;
  seedBehaviour?: QueuePromptSeedBehaviour;
  seedNodeId: string;
  seedStep: QueueSeedStep;
  /** Legacy recovery maps the old random toggle to seedStep and preserves its expansion rules. */
  legacySeedPlan?: true;
}

export interface QueueEnqueueResult {
  batchId?: string;
  enqueued: number;
  itemIds: number[];
  requested: number;
}

/** Immutable source-compiled payload. Queue submits it without reading source widget state. */
export type QueueCompiledSubmission =
  | {
      batchCount: number;
      graph: QueueBackendGraph;
      kind: 'workflow';
      /** Serialized parent workflow, compatible with the backend's `WorkflowWithoutID` schema. */
      workflow?: Record<string, unknown>;
      /** The seed inputs that vary between runs, expanded into one zipped batch group at send time. */
      seeds?: QueueWorkflowSeed[];
      /** Batch-node value lists resolved at compile time: outer groups multiply, inner datums zip. */
      batchData?: QueueWorkflowBatchDatum[][];
      /** Capture the bound library ID at compile time so completion attribution survives later editor changes. */
      libraryWorkflowId?: string;
    }
  | {
      batchCount: number;
      graph: QueueBackendGraph;
      kind: 'generate';
      negativePrompt: string;
      negativePromptNodeId: string;
      positivePrompt: string;
      positivePromptNodeId: string;
      /** Absent on items compiled before dynamic prompting shipped. */
      positivePrompts?: string[];
      seed: number;
      seedBehaviour?: QueuePromptSeedBehaviour;
      seedNodeId: string;
      /** Items compiled before seed modes carry `shouldRandomizeSeed` instead; the runtime maps it on read. */
      seedStep: QueueSeedStep;
    }
  | { error: string; kind: 'invalid' };

export interface QueueBackendItem {
  batchId?: string;
  destination?: string | null;
  errorMessage?: string | null;
  errorType?: string | null;
  id: number;
  origin?: string | null;
  status: QueueItemStatus;
}

export interface QueueResultImage {
  /** Board the backend saved the image to (a node's explicit board); unset when uncategorized. */
  boardId?: string;
  /** Backend creation timestamp; `queuedAt` is the submission instant. */
  createdAt?: string;
  height: number;
  imageName: string;
  imageUrl: string;
  isIntermediate?: boolean;
  queuedAt: string;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  width: number;
}

/** A settled run of a library-bound workflow, reported once its results are routed. */
export interface QueueWorkflowRunCompletedEvent {
  /** Result images in run order; the last one is the run's final output. */
  imageNames: readonly string[];
  libraryWorkflowId: string;
  projectId: string;
  queueItemId: string;
}

/** App injects optional completed-library-run capture; Queue owns neither library nor gallery. */
export interface QueueWorkflowRunSink {
  onWorkflowRunCompleted(event: QueueWorkflowRunCompletedEvent): void;
}

export type QueueItemStatus = 'pending' | 'in_progress' | 'waiting' | 'completed' | 'failed' | 'canceled';
export type TerminalQueueItemStatus = Extract<QueueItemStatus, 'completed' | 'failed' | 'canceled'>;
export type QueueConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export interface QueueQueryScope {
  originPrefix?: string;
}

export interface QueueNodeFieldValue {
  fieldName: string;
  nodePath: string;
  value: string | number | { imageName?: string } | null;
}

/** Live, UI-facing representation of one backend queue item. */
export interface QueueItemReadModel {
  batchId: string;
  completedAt?: string | null;
  createdAt: string;
  destination?: string | null;
  /** The accelerator that processed this item, e.g. `cuda:1` or `xpu:1`. Null on unindexed and single-device installs. */
  device?: string | null;
  errorMessage?: string | null;
  errorTraceback?: string | null;
  errorType?: string | null;
  fieldValues?: QueueNodeFieldValue[] | null;
  id: number;
  origin?: string | null;
  resultImageNames: string[];
  retriedFromItemId?: number | null;
  sessionId: string;
  startedAt?: string | null;
  status: QueueItemStatus;
  updatedAt: string;
  userDisplayName?: string | null;
  userEmail?: string | null;
  userId?: string;
}

export interface QueueCounts {
  batchId?: string | null;
  canceled: number;
  completed: number;
  failed: number;
  inProgress: number;
  itemId?: number | null;
  pending: number;
  queueId: string;
  sessionId?: string | null;
  total: number;
  userInProgress?: number | null;
  userPending?: number | null;
  waiting: number;
}

export interface PersonalQueueActivity {
  inProgress: number;
  pending: number;
}

/** Active counts for the signed-in user, falling back to global counts in single-user mode. */
export const getPersonalQueueActivity = (counts: QueueCounts): PersonalQueueActivity => ({
  inProgress: counts.userInProgress ?? counts.inProgress,
  pending: counts.userPending ?? counts.pending,
});

export interface QueueProcessorReadModel {
  isProcessing: boolean;
  isStarted: boolean;
}

export interface QueueStatusReadModel {
  processor: QueueProcessorReadModel;
  queue: QueueCounts;
}

export interface QueueItemIdsReadModel {
  itemIds: number[];
  totalCount: number;
}

export interface QueueItemProgress {
  /** 1-based image slot currently executing inside this local batch. */
  activeItemIndex?: number;
  completedItemCount?: number;
  message: string;
  /** 0..1, or null while indeterminate. */
  percentage: number | null;
  totalItemCount?: number;
}

export interface QueueItemProgressTarget {
  queueItemId: string;
  itemIndex: number;
}

export interface QueueReadModel {
  current: QueueItemReadModel | null;
  items: QueueItemReadModel[];
  next: QueueItemReadModel | null;
  scope: QueueQueryScope;
  status: QueueStatusReadModel;
}

export interface QueueResultImageOptions {
  resultNodeIds?: readonly string[];
}

export interface QueueResultVideoOptions extends QueueResultImageOptions {
  /** Drop videos whose DTO reports is_intermediate — the video analogue of filterIntermediateResults. */
  excludeIntermediate?: boolean;
}

/** User-facing Queue commands; transport and adapter details stay private. */
export interface QueueFeatureCommands {
  cancelCurrentItem(): Promise<void>;
  cancelItem(itemId: number): Promise<void>;
  cancelScopedItems(scope?: QueueQueryScope, options?: { keepCurrent?: boolean }): Promise<void>;
  clearFailedItems(scope?: QueueQueryScope): Promise<void>;
  clearItems(scope?: QueueQueryScope): Promise<void>;
  pauseProcessor(): Promise<void>;
  resumeProcessor(): Promise<void>;
}

/** Own command transport and realtime events together so runtimes do not assemble HTTP/socket plumbing. */
/** Feed preview snapshots through the socket handler's revision gate to reject frames already delivered live. */
export interface QueueProgressPreviewPayload {
  queue_id: string;
  item_id: number;
  session_id: string;
  invocation_source_id: string;
  parent_item_id?: number | null;
  root_item_id?: number | null;
  workflow_call_parent_source_id?: string | null;
  revision: number | null;
  message: string;
  percentage: number | null;
  image: { width: number; height: number; dataURL: string } | null;
}

export interface QueueBackendPort extends QueueFeatureCommands {
  acknowledgeEnqueue?(projectId: string, sourceQueueItemId: string): Promise<void>;
  getEnqueueReceipt?(projectId: string, sourceQueueItemId: string): Promise<QueueEnqueueResult | null>;
  /** The latest preview frame of each of the user's running items (`GET queue/{id}/previews`). */
  readProgressPreviews?(signal?: AbortSignal): Promise<QueueProgressPreviewPayload[]>;
  cancelQueueItems(itemIds: number[]): Promise<void>;
  cancelQueueItemsByBatchIds(batchIds: string[]): Promise<void>;
  enqueueGenerate(request: QueueEnqueueGenerateRequest): Promise<QueueEnqueueResult>;
  enqueueWorkflow(request: QueueEnqueueWorkflowRequest): Promise<QueueEnqueueResult>;
  getItem(itemId: number): Promise<QueueBackendItem>;
  getResultImages(
    itemId: number,
    sourceQueueItemId: string,
    queuedAt: string,
    options?: QueueResultImageOptions
  ): Promise<QueueResultImage[]>;
  /** Names of the videos a completed backend item produced (no DTO hydration needed). */
  getResultVideoNames(itemId: number, options?: QueueResultVideoOptions): Promise<string[]>;
  listItems(): Promise<QueueBackendItem[]>;
  readCurrent(scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueItemReadModel | null>;
  readItemIds(order: 'asc' | 'desc', scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueItemIdsReadModel>;
  readItemsById(itemIds: number[], signal?: AbortSignal): Promise<QueueItemReadModel[]>;
  readNext(scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueItemReadModel | null>;
  readStatus(scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueStatusReadModel>;
  retryItems(itemIds: number[]): Promise<unknown>;
  emit(event: string, payload: unknown): void;
  on(event: string, handler: (payload: never) => void): () => void;
  onConnectionChange(handler: (status: QueueConnectionStatus, error?: string) => void): () => void;
}
