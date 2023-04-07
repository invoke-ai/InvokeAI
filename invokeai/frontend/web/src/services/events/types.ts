import { Graph, GraphExecutionState } from '../api';

/**
 * A progress image, we get one for each step in the generation
 */
export type ProgressImage = {
  dataURL: string;
  width: number;
  height: number;
};

export type AnyInvocation = NonNullable<
  NonNullable<Graph['nodes']>[string]['type']
>;

export type AnyResult = GraphExecutionState['results'][string];

/**
 * A `generator_progress` socket.io event.
 *
 * @example socket.on('generator_progress', (data: GeneratorProgressEvent) => { ... }
 */
export type GeneratorProgressEvent = {
  graph_execution_state_id: string;
  invocation: AnyInvocation;
  source_id: string;
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
  source_id: string;
  result: AnyResult;
};

/**
 * A `invocation_error` socket.io event.
 *
 * @example socket.on('invocation_error', (data: InvocationErrorEvent) => { ... }
 */
export type InvocationErrorEvent = {
  graph_execution_state_id: string;
  invocation: AnyInvocation;
  source_id: string;
  error: string;
};

/**
 * A `invocation_started` socket.io event.
 *
 * @example socket.on('invocation_started', (data: InvocationStartedEvent) => { ... }
 */
export type InvocationStartedEvent = {
  graph_execution_state_id: string;
  invocation: AnyInvocation;
  source_id: string;
};

/**
 * A `graph_execution_state_complete` socket.io event.
 *
 * @example socket.on('graph_execution_state_complete', (data: GraphExecutionStateCompleteEvent) => { ... }
 */
export type GraphExecutionStateCompleteEvent = {
  graph_execution_state_id: string;
};
