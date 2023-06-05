/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Applies mediapipe face processing to image
 */
export type MediapipeFaceProcessorInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'mediapipe_face_processor';
  /**
   * The image to process
   */
  image?: ImageField;
  /**
   * Maximum number of faces to detect
   */
  max_faces?: number;
  /**
   * Minimum confidence for face detection
   */
  min_confidence?: number;
};

