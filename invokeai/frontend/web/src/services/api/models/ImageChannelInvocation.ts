/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';
import type { InvocationMeta } from './InvocationMeta';

/**
 * Gets a channel from an image.
 */
export type ImageChannelInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img_chan';
  /**
   * The meta properties of this node.
   */
  meta?: InvocationMeta;
  /**
   * The image to get the channel from
   */
  image?: ImageField;
  /**
   * The channel to get
   */
  channel?: 'A' | 'R' | 'G' | 'B';
};

