/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { CollectInvocationOutput } from './CollectInvocationOutput';
import type { CompelOutput } from './CompelOutput';
import type { ControlOutput } from './ControlOutput';
import type { FloatCollectionOutput } from './FloatCollectionOutput';
import type { FloatOutput } from './FloatOutput';
import type { Graph } from './Graph';
import type { GraphInvocationOutput } from './GraphInvocationOutput';
import type { ImageOutput } from './ImageOutput';
import type { IntCollectionOutput } from './IntCollectionOutput';
import type { IntOutput } from './IntOutput';
import type { IterateInvocationOutput } from './IterateInvocationOutput';
import type { LatentsOutput } from './LatentsOutput';
import type { MaskOutput } from './MaskOutput';
import type { NoiseOutput } from './NoiseOutput';
import type { PromptOutput } from './PromptOutput';

/**
 * Tracks the state of a graph execution
 */
export type GraphExecutionState = {
  /**
   * The id of the execution state
   */
  id: string;
  /**
   * The graph being executed
   */
  graph: Graph;
  /**
   * The expanded graph of activated and executed nodes
   */
  execution_graph: Graph;
  /**
   * The set of node ids that have been executed
   */
  executed: Array<string>;
  /**
   * The list of node ids that have been executed, in order of execution
   */
  executed_history: Array<string>;
  /**
   * The results of node executions
   */
  results: Record<string, (ImageOutput | MaskOutput | ControlOutput | PromptOutput | CompelOutput | IntOutput | FloatOutput | LatentsOutput | NoiseOutput | IntCollectionOutput | FloatCollectionOutput | GraphInvocationOutput | IterateInvocationOutput | CollectInvocationOutput)>;
  /**
   * Errors raised when executing nodes
   */
  errors: Record<string, string>;
  /**
   * The map of prepared nodes to original graph nodes
   */
  prepared_source_mapping: Record<string, string>;
  /**
   * The map of original graph nodes to prepared nodes
   */
  source_prepared_mapping: Record<string, Array<string>>;
};

