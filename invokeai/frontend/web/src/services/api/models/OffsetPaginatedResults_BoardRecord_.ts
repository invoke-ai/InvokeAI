/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BoardRecord } from './BoardRecord';

/**
 * Offset-paginated results
 */
export type OffsetPaginatedResults_BoardRecord_ = {
  /**
   * Items
   */
  items: Array<BoardRecord>;
  /**
   * Offset from which to retrieve items
   */
  offset: number;
  /**
   * Limit of items to get
   */
  limit: number;
  /**
   * Total number of items in result
   */
  total: number;
};

