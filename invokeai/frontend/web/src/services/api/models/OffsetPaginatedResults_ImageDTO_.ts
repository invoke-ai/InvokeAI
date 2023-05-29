/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageDTO } from './ImageDTO';

/**
 * Offset-paginated results
 */
export type OffsetPaginatedResults_ImageDTO_ = {
  /**
   * Items
   */
  items: Array<ImageDTO>;
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

