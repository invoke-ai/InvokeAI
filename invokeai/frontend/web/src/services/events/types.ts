import type { components } from 'services/api/schema';
import type {
  BaseModelType,
  Graph,
  GraphExecutionState,
  ModelType,
  SubModelType,
} from 'services/api/types';

/**
 * A progress image, we get one for each step in the generation
 */
export type ProgressImage = {
  dataURL: string;
  width: number;
  height: number;
};

export type AnyInvocation = NonNullable<NonNullable<Graph['nodes']>[string]>;

export type AnyResult = NonNullable<GraphExecutionState['results'][string]>;

export type BaseNode = {
  id: string;
  type: string;
  [key: string]: AnyInvocation[keyof AnyInvocation];
};

export type ModelLoadStartedEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  model_name: string;
  base_model: BaseModelType;
  model_type: ModelType;
  submodel: SubModelType;
};

export type ModelLoadCompletedEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  model_name: string;
  base_model: BaseModelType;
  model_type: ModelType;
  submodel: SubModelType;
  hash?: string;
  location: string;
  precision: string;
};

/**
 * A `generator_progress` socket.io event.
 *
 * @example socket.on('generator_progress', (data: GeneratorProgressEvent) => { ... }
 */
export type GeneratorProgressEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  node_id: string;
  source_node_id: string;
  progress_image?: ProgressImage;
  step: number;
  order: number;
  total_steps: number;
};

/**
 * A `invocation_complete` socket.io event.
 *
 * `result` is a discriminated union with a `type` property as the discriminant.
 *
 * @example socket.on('invocation_complete', (data: InvocationCompleteEvent) => { ... }
 */
export type InvocationCompleteEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  node: BaseNode;
  source_node_id: string;
  result: AnyResult;
};

/**
 * A `invocation_error` socket.io event.
 *
 * @example socket.on('invocation_error', (data: InvocationErrorEvent) => { ... }
 */
export type InvocationErrorEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  node: BaseNode;
  source_node_id: string;
  error_type: string;
  error: string;
};

/**
 * A `invocation_started` socket.io event.
 *
 * @example socket.on('invocation_started', (data: InvocationStartedEvent) => { ... }
 */
export type InvocationStartedEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  node: BaseNode;
  source_node_id: string;
};

/**
 * A `graph_execution_state_complete` socket.io event.
 *
 * @example socket.on('graph_execution_state_complete', (data: GraphExecutionStateCompleteEvent) => { ... }
 */
export type GraphExecutionStateCompleteEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
};

/**
 * A `session_retrieval_error` socket.io event.
 *
 * @example socket.on('session_retrieval_error', (data: SessionRetrievalErrorEvent) => { ... }
 */
export type SessionRetrievalErrorEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  error_type: string;
  error: string;
};

/**
 * A `invocation_retrieval_error` socket.io event.
 *
 * @example socket.on('invocation_retrieval_error', (data: InvocationRetrievalErrorEvent) => { ... }
 */
export type InvocationRetrievalErrorEvent = {
  queue_id: string;
  queue_item_id: number;
  queue_batch_id: string;
  graph_execution_state_id: string;
  node_id: string;
  error_type: string;
  error: string;
};

/**
 * A `queue_item_status_changed` socket.io event.
 *
 * @example socket.on('queue_item_status_changed', (data: QueueItemStatusChangedEvent) => { ... }
 */
export type QueueItemStatusChangedEvent = {
  queue_id: string;
  queue_item: {
    queue_id: string;
    item_id: number;
    batch_id: string;
    session_id: string;
    status: components['schemas']['SessionQueueItemDTO']['status'];
    error: string | undefined;
    created_at: string;
    updated_at: string;
    started_at: string | undefined;
    completed_at: string | undefined;
  };
  batch_status: {
    queue_id: string;
    batch_id: string;
    pending: number;
    in_progress: number;
    completed: number;
    failed: number;
    canceled: number;
    total: number;
  };
  queue_status: {
    queue_id: string;
    item_id?: number;
    batch_id?: string;
    session_id?: string;
    pending: number;
    in_progress: number;
    completed: number;
    failed: number;
    canceled: number;
    total: number;
  };
};

export type ClientEmitSubscribeQueue = {
  queue_id: string;
};

export type ClientEmitUnsubscribeQueue = {
  queue_id: string;
};

export type UploadImagesEvent = {
  status: 'started' | 'processing' | 'done' | 'error';
  message?: string;
  progress?: number;
  processed?: number;
  total?: number;
  errors?: string[];
  images_uploading?: string[];
  images_DTOs?: any[];
};

export type ServerToClientEvents = {
  generator_progress: (payload: GeneratorProgressEvent) => void;
  invocation_complete: (payload: InvocationCompleteEvent) => void;
  invocation_error: (payload: InvocationErrorEvent) => void;
  invocation_started: (payload: InvocationStartedEvent) => void;
  graph_execution_state_complete: (
    payload: GraphExecutionStateCompleteEvent
  ) => void;
  model_load_started: (payload: ModelLoadStartedEvent) => void;
  model_load_completed: (payload: ModelLoadCompletedEvent) => void;
  session_retrieval_error: (payload: SessionRetrievalErrorEvent) => void;
  invocation_retrieval_error: (payload: InvocationRetrievalErrorEvent) => void;
  queue_item_status_changed: (payload: QueueItemStatusChangedEvent) => void;
  upload_images: (payload: UploadImagesEvent) => void;
};

export type ClientToServerEvents = {
  connect: () => void;
  disconnect: () => void;
  subscribe_queue: (payload: ClientEmitSubscribeQueue) => void;
  unsubscribe_queue: (payload: ClientEmitUnsubscribeQueue) => void;
};
