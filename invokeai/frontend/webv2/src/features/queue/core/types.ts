import type {
  GeneratePromptSubmission,
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
  /** Ask the backend to run the batch ahead of the user's other pending work (Invoke Front). */
  prepend?: boolean;
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

export interface QueueEnqueueGenerateRequest extends QueueEnqueueRequestBase, GeneratePromptSubmission {}

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
      workflow?: Record<string, unknown>;
      /** The seed inputs that vary between runs, expanded into one zipped batch group at send time. */
      seeds?: QueueWorkflowSeed[];
      batchData?: QueueWorkflowBatchDatum[][];
      /** Absent on records queued before projects owned several workflows. */
      projectWorkflowId?: string;
      /** Library binding captured at enqueue, independent of later relinking or removing the project copy. */
      libraryWorkflowSource?: { libraryWorkflowId: string; ownerUserId?: string };
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
  /** Retained across result flattening so Canvas delivery acknowledges the item that produced each image. */
  sourceBackendItemId?: number;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  width: number;
}

export interface QueueResultVideo {
  /** Board the video is on when read (after any destination attach); unset when uncategorized. */
  boardId?: string;
  category: 'general' | 'control' | 'mask' | 'user' | 'other';
  createdAt?: string;
  durationSeconds: number;
  fps?: number;
  height: number;
  isIntermediate: boolean;
  mediaOrigin?: string;
  queuedAt: string;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  videoName: string;
  videoUrl: string;
  width: number;
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
  /** The workflow-call parent this item runs for; null or absent on root items. */
  parentItemId?: number | null;
  /** Pending items run highest priority first, then oldest first; absent when the source did not say. */
  priority?: number | null;
  resultImageNames: string[];
  retriedFromItemId?: number | null;
  sessionId: string;
  startedAt?: string | null;
  status: QueueItemStatus;
  /** Rises with every status change of the item; a reading with a lower one is older. Absent when unknown. */
  statusSequence?: number | null;
  updatedAt: string;
  userDisplayName?: string | null;
  userEmail?: string | null;
  userId?: string;
  /** The name of the workflow submitted with the item; absent for Generate runs, which submit none. */
  workflowName?: string | null;
}

/** Read on demand for diagnostics; carries the executed graph, results and workflow the live read model omits. */
export interface QueueItemDetailReadModel {
  errorMessage: string | null;
  errorTraceback: string | null;
  errorType: string | null;
  /** The compiled graph the session executed; empty when the backend redacted another user's item. */
  graph: Record<string, unknown>;
  id: number;
  priority: number | null;
  results: Record<string, unknown>;
  sessionId: string;
  status: QueueItemStatus;
  workflow: Record<string, unknown> | null;
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

/** The project workflow a run was compiled from; node progress is shown only in that editor. */
export interface QueueRunOrigin {
  projectId: string;
  workflowId: string;
}

export interface QueueResultVideoOptions extends QueueResultImageOptions {
  excludeIntermediate?: boolean;
}

export interface QueueFeatureCommands {
  cancelBatch(batchId: string): Promise<number>;
  /** Resolves with the item as the cancel left it: unchanged, so not canceled, when it had already finished. */
  cancelItem(itemId: number): Promise<QueueItemReadModel>;
  cancelScopedItems(scope?: QueueQueryScope, options?: { keepCurrent?: boolean }): Promise<void>;
  clearFailedItems(scope?: QueueQueryScope): Promise<void>;
  clearFinishedItems(scope?: QueueQueryScope): Promise<void>;
  clearItems(scope?: QueueQueryScope): Promise<void>;
  pauseProcessor(): Promise<void>;
  resumeProcessor(): Promise<void>;
  /** Re-queues failed or canceled root items from their saved sessions; resolves with the ids actually retried. */
  retryItems(itemIds: number[]): Promise<number[]>;
}

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

/** Who stages a retried item's results, as the server agreed it across every browser and tab of the account. */
export type QueueItemDeliveryClaim =
  | { kind: 'claimed' | 'delivered' | 'released' }
  | { kind: 'held'; retryInMs: number };

export interface QueueBackendPort extends QueueFeatureCommands {
  acknowledgeEnqueue?(projectId: string, sourceQueueItemId: string): Promise<void>;
  /** A claim holds until confirmed, released or expired; a confirmed item is delivered for good. */
  claimItemDelivery(
    itemId: number,
    action: 'claim' | 'confirm' | 'release',
    claimant: string
  ): Promise<QueueItemDeliveryClaim>;
  getEnqueueReceipt?(projectId: string, sourceQueueItemId: string): Promise<QueueEnqueueResult | null>;
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
  getResultVideoNames(itemId: number, options?: QueueResultVideoOptions): Promise<string[]>;
  /** Hydrate result videos for display; videos that are gone or unreadable are omitted. */
  getResultVideos(videoNames: string[], sourceQueueItemId: string, queuedAt: string): Promise<QueueResultVideo[]>;
  listItems(): Promise<QueueBackendItem[]>;
  readCurrent(scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueItemReadModel | null>;
  /** With `limit`, the first ids of that order; a server that predates `limit` returns every id. */
  readItemIds(
    order: 'asc' | 'desc',
    scope?: QueueQueryScope,
    signal?: AbortSignal,
    limit?: number
  ): Promise<QueueItemIdsReadModel>;
  readItemsById(itemIds: number[], signal?: AbortSignal): Promise<QueueItemReadModel[]>;
  readNext(scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueItemReadModel | null>;
  readStatus(scope?: QueueQueryScope, signal?: AbortSignal): Promise<QueueStatusReadModel>;
  readItemDetail(itemId: number, signal?: AbortSignal): Promise<QueueItemDetailReadModel>;
  retryItems(itemIds: number[]): Promise<number[]>;
  emit(event: string, payload: unknown): void;
  on(event: string, handler: (payload: never) => void): () => void;
  onConnectionChange(handler: (status: QueueConnectionStatus, error?: string) => void): () => void;
}
