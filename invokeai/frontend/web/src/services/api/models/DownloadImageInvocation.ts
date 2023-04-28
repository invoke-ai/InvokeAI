/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Download an image from a URL
 */
export type DownloadImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'download_image';
  /**
   * The URL to download
   */
  image_url: string;
};

