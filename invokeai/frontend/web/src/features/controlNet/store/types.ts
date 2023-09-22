import { isObject } from 'lodash-es';
import {
  CannyImageProcessorInvocation,
  ContentShuffleImageProcessorInvocation,
  HedImageProcessorInvocation,
  LineartAnimeImageProcessorInvocation,
  LineartImageProcessorInvocation,
  MediapipeFaceProcessorInvocation,
  MidasDepthImageProcessorInvocation,
  MlsdImageProcessorInvocation,
  NormalbaeImageProcessorInvocation,
  OpenposeImageProcessorInvocation,
  PidiImageProcessorInvocation,
  ZoeDepthImageProcessorInvocation,
} from 'services/api/types';
import { O } from 'ts-toolbelt';

/**
 * Any ControlNet processor node
 */
export type ControlNetProcessorNode =
  | CannyImageProcessorInvocation
  | ContentShuffleImageProcessorInvocation
  | HedImageProcessorInvocation
  | LineartAnimeImageProcessorInvocation
  | LineartImageProcessorInvocation
  | MediapipeFaceProcessorInvocation
  | MidasDepthImageProcessorInvocation
  | MlsdImageProcessorInvocation
  | NormalbaeImageProcessorInvocation
  | OpenposeImageProcessorInvocation
  | PidiImageProcessorInvocation
  | ZoeDepthImageProcessorInvocation;

/**
 * Any ControlNet processor type
 */
export type ControlNetProcessorType = NonNullable<
  ControlNetProcessorNode['type'] | 'none'
>;

/**
 * The Canny processor node, with parameters flagged as required
 */
export type RequiredCannyImageProcessorInvocation = O.Required<
  CannyImageProcessorInvocation,
  'type' | 'low_threshold' | 'high_threshold'
>;

/**
 * The ContentShuffle processor node, with parameters flagged as required
 */
export type RequiredContentShuffleImageProcessorInvocation = O.Required<
  ContentShuffleImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'w' | 'h' | 'f'
>;

/**
 * The HED processor node, with parameters flagged as required
 */
export type RequiredHedImageProcessorInvocation = O.Required<
  HedImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'scribble'
>;

/**
 * The Lineart Anime processor node, with parameters flagged as required
 */
export type RequiredLineartAnimeImageProcessorInvocation = O.Required<
  LineartAnimeImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution'
>;

/**
 * The Lineart processor node, with parameters flagged as required
 */
export type RequiredLineartImageProcessorInvocation = O.Required<
  LineartImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'coarse'
>;

/**
 * The MediapipeFace processor node, with parameters flagged as required
 */
export type RequiredMediapipeFaceProcessorInvocation = O.Required<
  MediapipeFaceProcessorInvocation,
  'type' | 'max_faces' | 'min_confidence'
>;

/**
 * The MidasDepth processor node, with parameters flagged as required
 */
export type RequiredMidasDepthImageProcessorInvocation = O.Required<
  MidasDepthImageProcessorInvocation,
  'type' | 'a_mult' | 'bg_th'
>;

/**
 * The MLSD processor node, with parameters flagged as required
 */
export type RequiredMlsdImageProcessorInvocation = O.Required<
  MlsdImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'thr_v' | 'thr_d'
>;

/**
 * The NormalBae processor node, with parameters flagged as required
 */
export type RequiredNormalbaeImageProcessorInvocation = O.Required<
  NormalbaeImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution'
>;

/**
 * The Openpose processor node, with parameters flagged as required
 */
export type RequiredOpenposeImageProcessorInvocation = O.Required<
  OpenposeImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'hand_and_face'
>;

/**
 * The Pidi processor node, with parameters flagged as required
 */
export type RequiredPidiImageProcessorInvocation = O.Required<
  PidiImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'safe' | 'scribble'
>;

/**
 * The ZoeDepth processor node, with parameters flagged as required
 */
export type RequiredZoeDepthImageProcessorInvocation = O.Required<
  ZoeDepthImageProcessorInvocation,
  'type'
>;

/**
 * Any ControlNet Processor node, with its parameters flagged as required
 */
export type RequiredControlNetProcessorNode = O.Required<
  | RequiredCannyImageProcessorInvocation
  | RequiredContentShuffleImageProcessorInvocation
  | RequiredHedImageProcessorInvocation
  | RequiredLineartAnimeImageProcessorInvocation
  | RequiredLineartImageProcessorInvocation
  | RequiredMediapipeFaceProcessorInvocation
  | RequiredMidasDepthImageProcessorInvocation
  | RequiredMlsdImageProcessorInvocation
  | RequiredNormalbaeImageProcessorInvocation
  | RequiredOpenposeImageProcessorInvocation
  | RequiredPidiImageProcessorInvocation
  | RequiredZoeDepthImageProcessorInvocation,
  'id'
>;

/**
 * Type guard for CannyImageProcessorInvocation
 */
export const isCannyImageProcessorInvocation = (
  obj: unknown
): obj is CannyImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'canny_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for ContentShuffleImageProcessorInvocation
 */
export const isContentShuffleImageProcessorInvocation = (
  obj: unknown
): obj is ContentShuffleImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'content_shuffle_image_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for HedImageprocessorInvocation
 */
export const isHedImageprocessorInvocation = (
  obj: unknown
): obj is HedImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'hed_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for LineartAnimeImageProcessorInvocation
 */
export const isLineartAnimeImageProcessorInvocation = (
  obj: unknown
): obj is LineartAnimeImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'lineart_anime_image_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for LineartImageProcessorInvocation
 */
export const isLineartImageProcessorInvocation = (
  obj: unknown
): obj is LineartImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'lineart_image_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for MediapipeFaceProcessorInvocation
 */
export const isMediapipeFaceProcessorInvocation = (
  obj: unknown
): obj is MediapipeFaceProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'mediapipe_face_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for MidasDepthImageProcessorInvocation
 */
export const isMidasDepthImageProcessorInvocation = (
  obj: unknown
): obj is MidasDepthImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'midas_depth_image_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for MlsdImageProcessorInvocation
 */
export const isMlsdImageProcessorInvocation = (
  obj: unknown
): obj is MlsdImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'mlsd_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for NormalbaeImageProcessorInvocation
 */
export const isNormalbaeImageProcessorInvocation = (
  obj: unknown
): obj is NormalbaeImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'normalbae_image_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for OpenposeImageProcessorInvocation
 */
export const isOpenposeImageProcessorInvocation = (
  obj: unknown
): obj is OpenposeImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'openpose_image_processor'
  ) {
    return true;
  }
  return false;
};

/**
 * Type guard for PidiImageProcessorInvocation
 */
export const isPidiImageProcessorInvocation = (
  obj: unknown
): obj is PidiImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'pidi_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for ZoeDepthImageProcessorInvocation
 */
export const isZoeDepthImageProcessorInvocation = (
  obj: unknown
): obj is ZoeDepthImageProcessorInvocation => {
  if (
    isObject(obj) &&
    'type' in obj &&
    obj.type === 'zoe_depth_image_processor'
  ) {
    return true;
  }
  return false;
};
