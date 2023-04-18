/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * An image's InvokeAI-specific metadata
 */
export type InvokeAIMetadata = {
  /**
   * The session that generated this image
   */
  session_id: string;
  /**
   * The prepared invocation that generated this image
   */
  invocation?: any;
};

