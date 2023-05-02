/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Outputs an image from a base 64 data URL.
 */
export type DataURLToImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'dataURL_image';
  /**
   * The b64 data URL
   */
  dataURL: string;
};

