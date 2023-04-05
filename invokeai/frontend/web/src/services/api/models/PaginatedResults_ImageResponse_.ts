/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageResponse } from './ImageResponse';

/**
 * Paginated results
 */
export type PaginatedResults_ImageResponse_ = {
  /**
   * Items
   */
  items: Array<ImageResponse>;
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

