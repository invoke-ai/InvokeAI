/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BoardDTO } from './BoardDTO';

/**
 * Offset-paginated results
 */
export type OffsetPaginatedResults_BoardDTO_ = {
  /**
   * Items
   */
  items: Array<BoardDTO>;
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

