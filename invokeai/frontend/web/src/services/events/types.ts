import { Graph, GraphExecutionState } from '../api';

/**
 * A progress image, we get one for each step in the generation
 */
export type ProgressImage = {
  dataURL: string;
  width: number;
  height: number;
};

export type AnyInvocationType = NonNullable<
  NonNullable<Graph['nodes']>[string]['type']
>;

export type AnyInvocation = NonNullable<Graph['nodes']>[string];

export type AnyResult = GraphExecutionState['results'][string];

/**
 * A `generator_progress` socket.io event.
 *
 * @example socket.on('generator_progress', (data: GeneratorProgressEvent) => { ... }
 */
export type GeneratorProgressEvent = {
  graph_execution_state_id: string;
  node: AnyInvocation;
  source_node_id: string;
  progress_image?: ProgressImage;
  step: number;
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
  graph_execution_state_id: string;
  node: AnyInvocation;
  source_node_id: string;
  result: AnyResult;
};

/**
 * A `invocation_error` socket.io event.
 *
 * @example socket.on('invocation_error', (data: InvocationErrorEvent) => { ... }
 */
export type InvocationErrorEvent = {
  graph_execution_state_id: string;
  node: AnyInvocation;
  source_node_id: string;
  error: string;
};

/**
 * A `invocation_started` socket.io event.
 *
 * @example socket.on('invocation_started', (data: InvocationStartedEvent) => { ... }
 */
export type InvocationStartedEvent = {
  graph_execution_state_id: string;
  node: AnyInvocation;
  source_node_id: string;
};

/**
 * A `graph_execution_state_complete` socket.io event.
 *
 * @example socket.on('graph_execution_state_complete', (data: GraphExecutionStateCompleteEvent) => { ... }
 */
export type GraphExecutionStateCompleteEvent = {
  graph_execution_state_id: string;
};

export type ClientEmitSubscribe = {
  session: string;
};

export type ClientEmitUnsubscribe = {
  session: string;
};

export type ServerToClientEvents = {
  generator_progress: (payload: GeneratorProgressEvent) => void;
  invocation_complete: (payload: InvocationCompleteEvent) => void;
  invocation_error: (payload: InvocationErrorEvent) => void;
  invocation_started: (payload: InvocationStartedEvent) => void;
  graph_execution_state_complete: (
    payload: GraphExecutionStateCompleteEvent
  ) => void;
};

export type ClientToServerEvents = {
  connect: () => void;
  disconnect: () => void;
  subscribe: (payload: ClientEmitSubscribe) => void;
  unsubscribe: (payload: ClientEmitUnsubscribe) => void;
};
