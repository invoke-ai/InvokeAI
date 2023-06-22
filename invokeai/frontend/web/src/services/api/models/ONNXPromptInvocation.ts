/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ClipField } from './ClipField';

/**
 * A node to process inputs and produce outputs.
 * May use dependency injection in __init__ to receive providers.
 */
export type ONNXPromptInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'prompt_onnx';
  /**
   * Prompt
   */
  prompt?: string;
  /**
   * Clip to use
   */
  clip?: ClipField;
};
