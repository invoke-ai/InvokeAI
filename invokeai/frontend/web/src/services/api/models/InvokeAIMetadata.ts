/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { NodeMetadata } from './NodeMetadata';

export type InvokeAIMetadata = {
  /**
   * The session in which this image was created
   */
  session_id?: string;
  /**
   * The node that created this image
   */
  node?: NodeMetadata;
};

