/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Base class for invocations that output a prompt
 */
export type PromptOutput = {
  type: 'prompt';
  /**
   * The output prompt
   */
  prompt: string;
};

