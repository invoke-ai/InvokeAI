/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageDTO } from './ImageDTO';

/**
 * Paginated results
 */
export type PaginatedResults_ImageDTO_ = {
  /**
   * Items
   */
  items: Array<ImageDTO>;
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

