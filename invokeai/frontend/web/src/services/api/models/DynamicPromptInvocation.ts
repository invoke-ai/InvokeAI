/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator
 */
export type DynamicPromptInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'dynamic_prompt';
  /**
   * The prompt to parse with dynamicprompts
   */
  prompt: string;
  /**
   * The number of prompts to generate
   */
  max_prompts?: number;
  /**
   * Whether to use the combinatorial generator
   */
  combinatorial?: boolean;
};
