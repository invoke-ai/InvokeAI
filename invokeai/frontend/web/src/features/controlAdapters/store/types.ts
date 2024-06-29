import type { EntityState } from '@reduxjs/toolkit';
import type {
  ParameterControlNetModel,
  ParameterIPAdapterModel,
  ParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';
import type { components } from 'services/api/schema';
import type { Invocation } from 'services/api/types';
import type { O } from 'ts-toolbelt';
import { z } from 'zod';

/**
 * Any ControlNet processor node
 */
export type ControlAdapterProcessorNode =
  | Invocation<'canny_image_processor'>
  | Invocation<'color_map_image_processor'>
  | Invocation<'content_shuffle_image_processor'>
  | Invocation<'depth_anything_image_processor'>
  | Invocation<'hed_image_processor'>
  | Invocation<'lineart_anime_image_processor'>
  | Invocation<'lineart_image_processor'>
  | Invocation<'mediapipe_face_processor'>
  | Invocation<'midas_depth_image_processor'>
  | Invocation<'mlsd_image_processor'>
  | Invocation<'normalbae_image_processor'>
  | Invocation<'dw_openpose_image_processor'>
  | Invocation<'pidi_image_processor'>
  | Invocation<'tile_image_processor'>
  | Invocation<'zoe_depth_image_processor'>;

/**
 * Any ControlNet processor type
 */
export type ControlAdapterProcessorType = NonNullable<ControlAdapterProcessorNode['type'] | 'none'>;
export const zControlAdapterProcessorType = z.enum([
  'canny_image_processor',
  'color_map_image_processor',
  'content_shuffle_image_processor',
  'depth_anything_image_processor',
  'hed_image_processor',
  'lineart_anime_image_processor',
  'lineart_image_processor',
  'mediapipe_face_processor',
  'midas_depth_image_processor',
  'mlsd_image_processor',
  'normalbae_image_processor',
  'dw_openpose_image_processor',
  'pidi_image_processor',
  'tile_image_processor',
  'zoe_depth_image_processor',
  'none',
]);
export const isControlAdapterProcessorType = (v: unknown): v is ControlAdapterProcessorType =>
  zControlAdapterProcessorType.safeParse(v).success;

/**
 * The Canny processor node, with parameters flagged as required
 */
export type RequiredCannyImageProcessorInvocation = O.Required<
  Invocation<'canny_image_processor'>,
  'type' | 'low_threshold' | 'high_threshold' | 'image_resolution' | 'detect_resolution'
>;

/**
 * The Color Map processor node, with parameters flagged as required
 */
export type RequiredColorMapImageProcessorInvocation = O.Required<
  Invocation<'color_map_image_processor'>,
  'type' | 'color_map_tile_size'
>;

/**
 * The ContentShuffle processor node, with parameters flagged as required
 */
export type RequiredContentShuffleImageProcessorInvocation = O.Required<
  Invocation<'content_shuffle_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution' | 'w' | 'h' | 'f'
>;

/**
 * The DepthAnything processor node, with parameters flagged as required
 */
export type RequiredDepthAnythingImageProcessorInvocation = O.Required<
  Invocation<'depth_anything_image_processor'>,
  'type' | 'model_size' | 'resolution' | 'offload'
>;

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;

/**
 * The HED processor node, with parameters flagged as required
 */
export type RequiredHedImageProcessorInvocation = O.Required<
  Invocation<'hed_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution' | 'scribble'
>;

/**
 * The Lineart Anime processor node, with parameters flagged as required
 */
export type RequiredLineartAnimeImageProcessorInvocation = O.Required<
  Invocation<'lineart_anime_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution'
>;

/**
 * The Lineart processor node, with parameters flagged as required
 */
export type RequiredLineartImageProcessorInvocation = O.Required<
  Invocation<'lineart_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution' | 'coarse'
>;

/**
 * The MediapipeFace processor node, with parameters flagged as required
 */
export type RequiredMediapipeFaceProcessorInvocation = O.Required<
  Invocation<'mediapipe_face_processor'>,
  'type' | 'max_faces' | 'min_confidence' | 'image_resolution' | 'detect_resolution'
>;

/**
 * The MidasDepth processor node, with parameters flagged as required
 */
export type RequiredMidasDepthImageProcessorInvocation = O.Required<
  Invocation<'midas_depth_image_processor'>,
  'type' | 'a_mult' | 'bg_th' | 'image_resolution' | 'detect_resolution'
>;

/**
 * The MLSD processor node, with parameters flagged as required
 */
export type RequiredMlsdImageProcessorInvocation = O.Required<
  Invocation<'mlsd_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution' | 'thr_v' | 'thr_d'
>;

/**
 * The NormalBae processor node, with parameters flagged as required
 */
export type RequiredNormalbaeImageProcessorInvocation = O.Required<
  Invocation<'normalbae_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution'
>;

/**
 * The DW Openpose processor node, with parameters flagged as required
 */
export type RequiredDWOpenposeImageProcessorInvocation = O.Required<
  Invocation<'dw_openpose_image_processor'>,
  'type' | 'image_resolution' | 'draw_body' | 'draw_face' | 'draw_hands'
>;

/**
 * The Pidi processor node, with parameters flagged as required
 */
export type RequiredPidiImageProcessorInvocation = O.Required<
  Invocation<'pidi_image_processor'>,
  'type' | 'detect_resolution' | 'image_resolution' | 'safe' | 'scribble'
>;

/**
 * The Tile processor node, with parameters flagged as required
 */
export type RequiredTileImageProcessorInvocation = O.Required<
  Invocation<'tile_image_processor'>,
  'type' | 'down_sampling_rate' | 'mode'
>;

/**
 * The ZoeDepth processor node, with parameters flagged as required
 */
export type RequiredZoeDepthImageProcessorInvocation = O.Required<Invocation<'zoe_depth_image_processor'>, 'type'>;

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
      | RequiredDWOpenposeImageProcessorInvocation
      | RequiredPidiImageProcessorInvocation
      | RequiredTileImageProcessorInvocation
      | RequiredZoeDepthImageProcessorInvocation,
      'id'
    >
  | { type: 'none' };

export type ControlMode = NonNullable<components['schemas']['ControlNetInvocation']['control_mode']>;

const zResizeMode = z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']);
export type ResizeMode = z.infer<typeof zResizeMode>;
export const isResizeMode = (v: unknown): v is ResizeMode => zResizeMode.safeParse(v).success;

const zIPMethod = z.enum(['full', 'style', 'composition']);
export type IPMethod = z.infer<typeof zIPMethod>;
export const isIPMethod = (v: unknown): v is IPMethod => zIPMethod.safeParse(v).success;

const zTileProcessorMode = z.enum(['regular', 'blur', 'var', 'super']);
export type TileProcessorMode = z.infer<typeof zTileProcessorMode>;
export const isTileProcessorMode = (v: unknown): v is TileProcessorMode => zTileProcessorMode.safeParse(v).success;

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

export type CLIPVisionModel = 'ViT-H' | 'ViT-G';

export type IPAdapterConfig = {
  type: 'ip_adapter';
  id: string;
  isEnabled: boolean;
  controlImage: string | null;
  model: ParameterIPAdapterModel | null;
  clipVisionModel: CLIPVisionModel;
  weight: number;
  method: IPMethod;
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
