/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ClipField } from './ClipField';

/**
 * Parse prompt using compel package to conditioning.
 */
export type CompelInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'compel';
  /**
   * Prompt
   */
  prompt?: string;
  /**
   * Clip to use
   */
  clip?: ClipField;
};

