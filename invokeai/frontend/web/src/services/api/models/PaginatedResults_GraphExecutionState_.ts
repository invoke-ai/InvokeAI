/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { GraphExecutionState } from './GraphExecutionState';

/**
 * Paginated results
 */
export type PaginatedResults_GraphExecutionState_ = {
  /**
   * Items
   */
  items: Array<GraphExecutionState>;
  /**
   * Current Page
   */
  page: number;
  /**
   * Total number of pages
   */
  pages: number;
  /**
   * Number of items per page
   */
  per_page: number;
  /**
   * Total number of items in result
   */
  total: number;
};

