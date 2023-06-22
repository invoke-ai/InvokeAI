/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Gets a channel from an image.
 */
export type ImageChannelInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'img_chan';
  /**
   * The image to get the channel from
   */
  image?: ImageField;
  /**
   * The channel to get
   */
  channel?: 'A' | 'R' | 'G' | 'B';
};

