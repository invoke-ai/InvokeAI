import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { AnyInvocation, ControlNetModelConfig, Invocation, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { z } from 'zod';

const zCannyEdgeDetectionFilterConfig = z.object({
  type: z.literal('canny_edge_detection'),
  low_threshold: z.number().int().gte(0).lte(255),
  high_threshold: z.number().int().gte(0).lte(255),
});
export type CannyEdgeDetectionFilterConfig = z.infer<typeof zCannyEdgeDetectionFilterConfig>;

const zColorMapFilterConfig = z.object({
  type: z.literal('color_map'),
  tile_size: z.number().int().gte(1),
});
export type ColorMapFilterConfig = z.infer<typeof zColorMapFilterConfig>;

const zContentShuffleFilterConfig = z.object({
  type: z.literal('content_shuffle'),
  scale_factor: z.number().int().gte(1),
});
export type ContentShuffleFilterConfig = z.infer<typeof zContentShuffleFilterConfig>;

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small', 'small_v2']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;

const zDepthAnythingFilterConfig = z.object({
  type: z.literal('depth_anything_depth_estimation'),
  model_size: zDepthAnythingModelSize,
});
export type DepthAnythingFilterConfig = z.infer<typeof zDepthAnythingFilterConfig>;

const zHEDEdgeDetectionFilterConfig = z.object({
  type: z.literal('hed_edge_detection'),
  scribble: z.boolean(),
});
export type HEDEdgeDetectionFilterConfig = z.infer<typeof zHEDEdgeDetectionFilterConfig>;

const zLineartAnimeEdgeDetectionFilterConfig = z.object({
  type: z.literal('lineart_anime_edge_detection'),
});
export type LineartAnimeEdgeDetectionFilterConfig = z.infer<typeof zLineartAnimeEdgeDetectionFilterConfig>;

const zLineartEdgeDetectionFilterConfig = z.object({
  type: z.literal('lineart_edge_detection'),
  coarse: z.boolean(),
});
export type LineartEdgeDetectionFilterConfig = z.infer<typeof zLineartEdgeDetectionFilterConfig>;

const zMediaPipeFaceDetectionFilterConfig = z.object({
  type: z.literal('mediapipe_face_detection'),
  max_faces: z.number().int().gte(1),
  min_confidence: z.number().gte(0).lte(1),
});
export type MediaPipeFaceDetectionFilterConfig = z.infer<typeof zMediaPipeFaceDetectionFilterConfig>;

const zMLSDDetectionFilterConfig = z.object({
  type: z.literal('mlsd_detection'),
  score_threshold: z.number().gte(0),
  distance_threshold: z.number().gte(0),
});
export type MLSDDetectionFilterConfig = z.infer<typeof zMLSDDetectionFilterConfig>;

const zNormalMapFilterConfig = z.object({
  type: z.literal('normal_map'),
});
export type NormalMapFilterConfig = z.infer<typeof zNormalMapFilterConfig>;

const zPiDiNetEdgeDetectionFilterConfig = z.object({
  type: z.literal('pidi_edge_detection'),
  quantize_edges: z.boolean(),
  scribble: z.boolean(),
});
export type PiDiNetEdgeDetectionFilterConfig = z.infer<typeof zPiDiNetEdgeDetectionFilterConfig>;

const zDWOpenposeDetectionFilterConfig = z.object({
  type: z.literal('dw_openpose_detection'),
  draw_body: z.boolean(),
  draw_face: z.boolean(),
  draw_hands: z.boolean(),
});
export type DWOpenposeDetectionFilterConfig = z.infer<typeof zDWOpenposeDetectionFilterConfig>;

const zSpandrelFilterConfig = z.object({
  type: z.literal('spandrel_filter'),
  model: zModelIdentifierField.nullable(),
  autoScale: z.boolean(),
  scale: z.number().gte(1).lte(16),
});
export type SpandrelFilterConfig = z.infer<typeof zSpandrelFilterConfig>;

const zFilterConfig = z.discriminatedUnion('type', [
  zCannyEdgeDetectionFilterConfig,
  zColorMapFilterConfig,
  zContentShuffleFilterConfig,
  zDepthAnythingFilterConfig,
  zHEDEdgeDetectionFilterConfig,
  zLineartAnimeEdgeDetectionFilterConfig,
  zLineartEdgeDetectionFilterConfig,
  zMediaPipeFaceDetectionFilterConfig,
  zMLSDDetectionFilterConfig,
  zNormalMapFilterConfig,
  zPiDiNetEdgeDetectionFilterConfig,
  zDWOpenposeDetectionFilterConfig,
  zSpandrelFilterConfig,
]);
export type FilterConfig = z.infer<typeof zFilterConfig>;

const zFilterType = z.enum([
  'canny_edge_detection',
  'color_map',
  'content_shuffle',
  'depth_anything_depth_estimation',
  'hed_edge_detection',
  'lineart_anime_edge_detection',
  'lineart_edge_detection',
  'mediapipe_face_detection',
  'mlsd_detection',
  'normal_map',
  'pidi_edge_detection',
  'dw_openpose_detection',
  'spandrel_filter',
]);
export type FilterType = z.infer<typeof zFilterType>;
export const isFilterType = (v: unknown): v is FilterType => zFilterType.safeParse(v).success;

type ImageFilterData<T extends FilterConfig['type']> = {
  type: T;
  buildDefaults(): Extract<FilterConfig, { type: T }>;
  buildNode(imageDTO: ImageWithDims, config: Extract<FilterConfig, { type: T }>): AnyInvocation;
  validateConfig?(config: Extract<FilterConfig, { type: T }>): boolean;
};

export const IMAGE_FILTERS: { [key in FilterConfig['type']]: ImageFilterData<key> } = {
  canny_edge_detection: {
    type: 'canny_edge_detection',
    buildDefaults: () => ({
      type: 'canny_edge_detection',
      low_threshold: 100,
      high_threshold: 200,
    }),
    buildNode: ({ image_name }, { low_threshold, high_threshold }): Invocation<'canny_edge_detection'> => ({
      id: getPrefixedId('canny_edge_detection'),
      type: 'canny_edge_detection',
      image: { image_name },
      low_threshold,
      high_threshold,
    }),
  },
  color_map: {
    type: 'color_map',
    buildDefaults: () => ({
      type: 'color_map',
      tile_size: 64,
    }),
    buildNode: ({ image_name }, { tile_size }): Invocation<'color_map'> => ({
      id: getPrefixedId('color_map'),
      type: 'color_map',
      image: { image_name },
      tile_size,
    }),
  },
  content_shuffle: {
    type: 'content_shuffle',
    buildDefaults: () => ({
      type: 'content_shuffle',
      scale_factor: 256,
    }),
    buildNode: ({ image_name }, { scale_factor }): Invocation<'content_shuffle'> => ({
      id: getPrefixedId('content_shuffle'),
      type: 'content_shuffle',
      image: { image_name },
      scale_factor,
    }),
  },
  depth_anything_depth_estimation: {
    type: 'depth_anything_depth_estimation',
    buildDefaults: () => ({
      type: 'depth_anything_depth_estimation',
      model_size: 'small_v2',
    }),
    buildNode: ({ image_name }, { model_size }): Invocation<'depth_anything_depth_estimation'> => ({
      id: getPrefixedId('depth_anything_depth_estimation'),
      type: 'depth_anything_depth_estimation',
      image: { image_name },
      model_size,
    }),
  },
  hed_edge_detection: {
    type: 'hed_edge_detection',
    buildDefaults: () => ({
      type: 'hed_edge_detection',
      scribble: false,
    }),
    buildNode: ({ image_name }, { scribble }): Invocation<'hed_edge_detection'> => ({
      id: getPrefixedId('hed_edge_detection'),
      type: 'hed_edge_detection',
      image: { image_name },
      scribble,
    }),
  },
  lineart_anime_edge_detection: {
    type: 'lineart_anime_edge_detection',
    buildDefaults: () => ({
      type: 'lineart_anime_edge_detection',
    }),
    buildNode: ({ image_name }): Invocation<'lineart_anime_edge_detection'> => ({
      id: getPrefixedId('lineart_anime_edge_detection'),
      type: 'lineart_anime_edge_detection',
      image: { image_name },
    }),
  },
  lineart_edge_detection: {
    type: 'lineart_edge_detection',
    buildDefaults: () => ({
      type: 'lineart_edge_detection',
      coarse: false,
    }),
    buildNode: ({ image_name }, { coarse }): Invocation<'lineart_edge_detection'> => ({
      id: getPrefixedId('lineart_edge_detection'),
      type: 'lineart_edge_detection',
      image: { image_name },
      coarse,
    }),
  },
  mediapipe_face_detection: {
    type: 'mediapipe_face_detection',
    buildDefaults: () => ({
      type: 'mediapipe_face_detection',
      max_faces: 1,
      min_confidence: 0.5,
    }),
    buildNode: ({ image_name }, { max_faces, min_confidence }): Invocation<'mediapipe_face_detection'> => ({
      id: getPrefixedId('mediapipe_face_detection'),
      type: 'mediapipe_face_detection',
      image: { image_name },
      max_faces,
      min_confidence,
    }),
  },
  mlsd_detection: {
    type: 'mlsd_detection',
    buildDefaults: () => ({
      type: 'mlsd_detection',
      score_threshold: 0.1,
      distance_threshold: 20.0,
    }),
    buildNode: ({ image_name }, { score_threshold, distance_threshold }): Invocation<'mlsd_detection'> => ({
      id: getPrefixedId('mlsd_detection'),
      type: 'mlsd_detection',
      image: { image_name },
      score_threshold,
      distance_threshold,
    }),
  },
  normal_map: {
    type: 'normal_map',
    buildDefaults: () => ({
      type: 'normal_map',
    }),
    buildNode: ({ image_name }): Invocation<'normal_map'> => ({
      id: getPrefixedId('normal_map'),
      type: 'normal_map',
      image: { image_name },
    }),
  },
  pidi_edge_detection: {
    type: 'pidi_edge_detection',
    buildDefaults: () => ({
      type: 'pidi_edge_detection',
      quantize_edges: false,
      scribble: false,
    }),
    buildNode: ({ image_name }, { quantize_edges, scribble }): Invocation<'pidi_edge_detection'> => ({
      id: getPrefixedId('pidi_edge_detection'),
      type: 'pidi_edge_detection',
      image: { image_name },
      quantize_edges,
      scribble,
    }),
  },
  dw_openpose_detection: {
    type: 'dw_openpose_detection',
    buildDefaults: () => ({
      type: 'dw_openpose_detection',
      draw_body: true,
      draw_face: true,
      draw_hands: true,
    }),
    buildNode: ({ image_name }, { draw_body, draw_face, draw_hands }): Invocation<'dw_openpose_detection'> => ({
      id: getPrefixedId('dw_openpose_detection'),
      type: 'dw_openpose_detection',
      image: { image_name },
      draw_body,
      draw_face,
      draw_hands,
    }),
  },
  spandrel_filter: {
    type: 'spandrel_filter',
    buildDefaults: () => ({
      type: 'spandrel_filter',
      model: null,
      autoScale: true,
      scale: 1,
    }),
    buildNode: (
      { image_name },
      { model, scale, autoScale }
    ): Invocation<'spandrel_image_to_image' | 'spandrel_image_to_image_autoscale'> => {
      assert(model !== null);
      if (autoScale) {
        const node: Invocation<'spandrel_image_to_image_autoscale'> = {
          id: getPrefixedId('spandrel_image_to_image_autoscale'),
          type: 'spandrel_image_to_image_autoscale',
          image_to_image_model: model,
          image: { image_name },
          scale,
        };
        return node;
      } else {
        const node: Invocation<'spandrel_image_to_image'> = {
          id: getPrefixedId('spandrel_image_to_image'),
          type: 'spandrel_image_to_image',
          image_to_image_model: model,
          image: { image_name },
        };
        return node;
      }
    },
    validateConfig: (config): boolean => {
      if (!config.model) {
        return false;
      }
      return true;
    },
  },
} as const;

/**
 * A map of the v1 processor names to the new filter types.
 */
const PROCESSOR_TO_FILTER_MAP: Record<string, FilterType> = {
  canny_image_processor: 'canny_edge_detection',
  mlsd_image_processor: 'mlsd_detection',
  depth_anything_image_processor: 'depth_anything_depth_estimation',
  normalbae_image_processor: 'normal_map',
  pidi_image_processor: 'pidi_edge_detection',
  lineart_image_processor: 'lineart_edge_detection',
  lineart_anime_image_processor: 'lineart_anime_edge_detection',
  hed_image_processor: 'hed_edge_detection',
  content_shuffle_image_processor: 'content_shuffle',
  dw_openpose_image_processor: 'dw_openpose_detection',
  mediapipe_face_processor: 'mediapipe_face_detection',
  zoe_depth_image_processor: 'depth_anything_depth_estimation',
  color_map_image_processor: 'color_map',
};

/**
 * Gets the default filter for a control model. If the model has a default, it will be used, otherwise the default
 * filter for the model type will be used.
 */
export const getFilterForModel = (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null) => {
  if (!modelConfig) {
    // No model, use the default filter
    return IMAGE_FILTERS.canny_edge_detection;
  }

  const preprocessor = modelConfig?.default_settings?.preprocessor;
  if (!preprocessor) {
    // No preprocessor, use the default filter
    return IMAGE_FILTERS.canny_edge_detection;
  }

  if (isFilterType(preprocessor)) {
    // Valid filter type, use it
    return IMAGE_FILTERS[preprocessor];
  }

  const filterName = PROCESSOR_TO_FILTER_MAP[preprocessor];
  if (!filterName) {
    // No filter found, use the default filter
    return IMAGE_FILTERS.canny_edge_detection;
  }

  // Found a filter, use it
  return IMAGE_FILTERS[filterName];
};
