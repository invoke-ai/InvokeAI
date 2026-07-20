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
export type QueueSourceId = 'canvas' | 'generate' | 'upscale' | 'workflow';

export interface QueueSubmissionPresentation {
  batchCount: number;
  height: number;
  positivePrompt?: string;
  width: number;
}

export interface QueueGraphSnapshot {
  backendGraph?: QueueBackendGraph;
  edges?: unknown[];
  id: string;
  label: string;
  nodes?: unknown[];
  updatedAt?: string;
  version?: number;
}

export interface QueueEnqueueWorkflowRequest {
  batchCount: number;
  destination: QueueResultDestination;
  graph: QueueBackendGraph;
  projectId: string;
  sourceQueueItemId: string;
}

export interface QueueEnqueueGenerateRequest extends QueueEnqueueWorkflowRequest {
  negativePrompt: string;
  negativePromptNodeId: string;
  positivePrompt: string;
  positivePromptNodeId: string;
  seed: number;
  seedNodeId: string;
  shouldRandomizeSeed: boolean;
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
    }
  | {
      batchCount: number;
      graph: QueueBackendGraph;
      kind: 'generate';
      negativePrompt: string;
      negativePromptNodeId: string;
      positivePrompt: string;
      positivePromptNodeId: string;
      seed: number;
      seedNodeId: string;
      shouldRandomizeSeed: boolean;
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
  height: number;
  imageName: string;
  imageUrl: string;
  isIntermediate?: boolean;
  queuedAt: string;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  width: number;
}

export type QueueItemStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'canceled';
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
}

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

/** User-facing Queue commands; transport and adapter details stay private. */
export interface QueueFeatureCommands {
  cancelCurrentItem(): Promise<void>;
  cancelItem(itemId: number): Promise<void>;
  cancelScopedItems(scope?: QueueQueryScope, currentItemId?: number | null): Promise<void>;
  clearFailedItems(scope?: QueueQueryScope): Promise<void>;
  clearItems(scope?: QueueQueryScope): Promise<void>;
  pauseProcessor(): Promise<void>;
  resumeProcessor(): Promise<void>;
}

/**
 * The queue feature's backend seam. It owns both command transport and realtime
 * events so runtimes never assemble HTTP calls and socket subscriptions.
 */
export interface QueueBackendPort extends QueueFeatureCommands {
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
  listItems(): Promise<QueueBackendItem[]>;
  readCurrent(scope?: QueueQueryScope): Promise<QueueItemReadModel | null>;
  readItemIds(order: 'asc' | 'desc', scope?: QueueQueryScope): Promise<QueueItemIdsReadModel>;
  readItemsById(itemIds: number[]): Promise<QueueItemReadModel[]>;
  readNext(scope?: QueueQueryScope): Promise<QueueItemReadModel | null>;
  readStatus(scope?: QueueQueryScope): Promise<QueueStatusReadModel>;
  retryItems(itemIds: number[]): Promise<unknown>;
  emit(event: string, payload: unknown): void;
  on(event: string, handler: (payload: never) => void): () => void;
  onConnectionChange(handler: (status: QueueConnectionStatus, error?: string) => void): () => void;
}
