import type { EntityState } from '@reduxjs/toolkit';
import type {
  ParameterControlNetModel,
  ParameterIPAdapterModel,
  ParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';
import { isObject } from 'lodash-es';
import type { components } from 'services/api/schema';
import type {
  CannyImageProcessorInvocation,
  ColorMapImageProcessorInvocation,
  ContentShuffleImageProcessorInvocation,
  DepthAnythingImageProcessorInvocation,
  DWPoseImageProcessorInvocation,
  HedImageProcessorInvocation,
  LineartAnimeImageProcessorInvocation,
  LineartImageProcessorInvocation,
  MediapipeFaceProcessorInvocation,
  MidasDepthImageProcessorInvocation,
  MlsdImageProcessorInvocation,
  NormalbaeImageProcessorInvocation,
  PidiImageProcessorInvocation,
  ZoeDepthImageProcessorInvocation,
} from 'services/api/types';
import type { O } from 'ts-toolbelt';
import { z } from 'zod';

/**
 * Any ControlNet processor node
 */
export type ControlAdapterProcessorNode =
  | CannyImageProcessorInvocation
  | ColorMapImageProcessorInvocation
  | ContentShuffleImageProcessorInvocation
  | DepthAnythingImageProcessorInvocation
  | HedImageProcessorInvocation
  | LineartAnimeImageProcessorInvocation
  | LineartImageProcessorInvocation
  | MediapipeFaceProcessorInvocation
  | MidasDepthImageProcessorInvocation
  | MlsdImageProcessorInvocation
  | NormalbaeImageProcessorInvocation
  | DWPoseImageProcessorInvocation
  | PidiImageProcessorInvocation
  | ZoeDepthImageProcessorInvocation;

/**
 * Any ControlNet processor type
 */
export type ControlAdapterProcessorType = NonNullable<ControlAdapterProcessorNode['type'] | 'none'>;

/**
 * The Canny processor node, with parameters flagged as required
 */
export type RequiredCannyImageProcessorInvocation = O.Required<
  CannyImageProcessorInvocation,
  'type' | 'low_threshold' | 'high_threshold'
>;

/**
 * The Color Map processor node, with parameters flagged as required
 */
export type RequiredColorMapImageProcessorInvocation = O.Required<
  ColorMapImageProcessorInvocation,
  'type' | 'color_map_tile_size'
>;

/**
 * The ContentShuffle processor node, with parameters flagged as required
 */
export type RequiredContentShuffleImageProcessorInvocation = O.Required<
  ContentShuffleImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'w' | 'h' | 'f'
>;

/**
 * The DepthAnything processor node, with parameters flagged as required
 */
export type RequiredDepthAnythingImageProcessorInvocation = O.Required<
  DepthAnythingImageProcessorInvocation,
  'type' | 'model_size' | 'resolution' | 'offload'
>;

export const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;

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
 * The DWPose processor node, with parameters flagged as required
 */
export type RequiredDWPoseImageProcessorInvocation = O.Required<
  DWPoseImageProcessorInvocation,
  'type' | 'image_resolution' | 'draw_body' | 'draw_face' | 'draw_hands'
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
export type RequiredZoeDepthImageProcessorInvocation = O.Required<ZoeDepthImageProcessorInvocation, 'type'>;

/**
 * Any ControlNet Processor node, with its parameters flagged as required
 */
export type RequiredControlAdapterProcessorNode =
  | O.Required<
      | RequiredCannyImageProcessorInvocation
      | RequiredColorMapImageProcessorInvocation
      | RequiredContentShuffleImageProcessorInvocation
      | RequiredDepthAnythingImageProcessorInvocation
      | RequiredHedImageProcessorInvocation
      | RequiredLineartAnimeImageProcessorInvocation
      | RequiredLineartImageProcessorInvocation
      | RequiredMediapipeFaceProcessorInvocation
      | RequiredMidasDepthImageProcessorInvocation
      | RequiredMlsdImageProcessorInvocation
      | RequiredNormalbaeImageProcessorInvocation
      | RequiredDWPoseImageProcessorInvocation
      | RequiredPidiImageProcessorInvocation
      | RequiredZoeDepthImageProcessorInvocation,
      'id'
    >
  | { type: 'none' };

/**
 * Type guard for CannyImageProcessorInvocation
 */
export const isCannyImageProcessorInvocation = (obj: unknown): obj is CannyImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'canny_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for ColorMapImageProcessorInvocation
 */
export const isColorMapImageProcessorInvocation = (obj: unknown): obj is ColorMapImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'color_map_image_processor') {
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
  if (isObject(obj) && 'type' in obj && obj.type === 'content_shuffle_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for DepthAnythingImageProcessorInvocation
 */
export const isDepthAnythingImageProcessorInvocation = (obj: unknown): obj is DepthAnythingImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'depth_anything_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for HedImageprocessorInvocation
 */
export const isHedImageprocessorInvocation = (obj: unknown): obj is HedImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'hed_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for LineartAnimeImageProcessorInvocation
 */
export const isLineartAnimeImageProcessorInvocation = (obj: unknown): obj is LineartAnimeImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'lineart_anime_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for LineartImageProcessorInvocation
 */
export const isLineartImageProcessorInvocation = (obj: unknown): obj is LineartImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'lineart_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for MediapipeFaceProcessorInvocation
 */
export const isMediapipeFaceProcessorInvocation = (obj: unknown): obj is MediapipeFaceProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'mediapipe_face_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for MidasDepthImageProcessorInvocation
 */
export const isMidasDepthImageProcessorInvocation = (obj: unknown): obj is MidasDepthImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'midas_depth_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for MlsdImageProcessorInvocation
 */
export const isMlsdImageProcessorInvocation = (obj: unknown): obj is MlsdImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'mlsd_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for NormalbaeImageProcessorInvocation
 */
export const isNormalbaeImageProcessorInvocation = (obj: unknown): obj is NormalbaeImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'normalbae_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for DWPoseImageProcessorInvocation
 */
export const isDWPoseImageProcessorInvocation = (obj: unknown): obj is DWPoseImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'dwpose_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for PidiImageProcessorInvocation
 */
export const isPidiImageProcessorInvocation = (obj: unknown): obj is PidiImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'pidi_image_processor') {
    return true;
  }
  return false;
};

/**
 * Type guard for ZoeDepthImageProcessorInvocation
 */
export const isZoeDepthImageProcessorInvocation = (obj: unknown): obj is ZoeDepthImageProcessorInvocation => {
  if (isObject(obj) && 'type' in obj && obj.type === 'zoe_depth_image_processor') {
    return true;
  }
  return false;
};

export type ControlMode = NonNullable<components['schemas']['ControlNetInvocation']['control_mode']>;

export const zResizeMode = z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']);
export type ResizeMode = z.infer<typeof zResizeMode>;
export const isResizeMode = (v: unknown): v is ResizeMode => zResizeMode.safeParse(v).success;

export type ControlNetConfig = {
  type: 'controlnet';
  id: string;
  isEnabled: boolean;
  model: ParameterControlNetModel | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  controlMode: ControlMode;
  resizeMode: ResizeMode;
  controlImage: string | null;
  processedControlImage: string | null;
  processorType: ControlAdapterProcessorType;
  processorNode: RequiredControlAdapterProcessorNode;
  shouldAutoConfig: boolean;
};

export type T2IAdapterConfig = {
  type: 't2i_adapter';
  id: string;
  isEnabled: boolean;
  model: ParameterT2IAdapterModel | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  resizeMode: ResizeMode;
  controlImage: string | null;
  processedControlImage: string | null;
  processorType: ControlAdapterProcessorType;
  processorNode: RequiredControlAdapterProcessorNode;
  shouldAutoConfig: boolean;
};

export type IPAdapterConfig = {
  type: 'ip_adapter';
  id: string;
  isEnabled: boolean;
  controlImage: string | null;
  model: ParameterIPAdapterModel | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
};

export type ControlAdapterConfig = ControlNetConfig | IPAdapterConfig | T2IAdapterConfig;

export type ControlAdapterType = ControlAdapterConfig['type'];

export type ControlAdaptersState = EntityState<ControlAdapterConfig, string> & {
  pendingControlImages: string[];
};

export const isControlNet = (controlAdapter: ControlAdapterConfig): controlAdapter is ControlNetConfig => {
  return controlAdapter.type === 'controlnet';
};

export const isIPAdapter = (controlAdapter: ControlAdapterConfig): controlAdapter is IPAdapterConfig => {
  return controlAdapter.type === 'ip_adapter';
};

export const isT2IAdapter = (controlAdapter: ControlAdapterConfig): controlAdapter is T2IAdapterConfig => {
  return controlAdapter.type === 't2i_adapter';
};

export const isControlNetOrT2IAdapter = (
  controlAdapter: ControlAdapterConfig
): controlAdapter is ControlNetConfig | T2IAdapterConfig => {
  return isControlNet(controlAdapter) || isT2IAdapter(controlAdapter);
};
