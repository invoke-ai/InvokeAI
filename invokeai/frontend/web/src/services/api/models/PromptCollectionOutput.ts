/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Base class for invocations that output a collection of prompts
 */
export type PromptCollectionOutput = {
  type: 'prompt_collection_output';
  /**
   * The output prompt collection
   */
  prompt_collection: Array<string>;
  /**
   * The size of the prompt collection
   */
  count: number;
};
