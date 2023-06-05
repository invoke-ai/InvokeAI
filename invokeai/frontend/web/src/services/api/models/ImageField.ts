/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ResourceOrigin } from './ResourceOrigin';

/**
 * An image field used for passing image objects between invocations
 */
export type ImageField = {
  /**
   * The type of the image
   */
  image_origin: ResourceOrigin;
  /**
   * The name of the image
   */
  image_name: string;
};

