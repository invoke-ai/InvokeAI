import type { ModelConfig } from '@features/models';

import { isSupportedFilterType } from './filterGraphs';

export const PROCESSOR_TO_FILTER_MAP: Readonly<Record<string, string>> = {
  canny_image_processor: 'canny_edge_detection',
  color_map_image_processor: 'color_map',
  content_shuffle_image_processor: 'content_shuffle',
  depth_anything_image_processor: 'depth_anything_depth_estimation',
  dw_openpose_image_processor: 'dw_openpose_detection',
  hed_image_processor: 'hed_edge_detection',
  lineart_anime_image_processor: 'lineart_anime_edge_detection',
  lineart_image_processor: 'lineart_edge_detection',
  mediapipe_face_processor: 'mediapipe_face_detection',
  mlsd_image_processor: 'mlsd_detection',
  normalbae_image_processor: 'normal_map',
  pidi_image_processor: 'pidi_edge_detection',
  zoe_depth_image_processor: 'depth_anything_depth_estimation',
};

export const resolveDefaultFilterForModel = (
  model: Pick<ModelConfig, 'default_settings'> | null | undefined
): string | null => {
  const preprocessor = model?.default_settings?.preprocessor;
  if (!preprocessor) {
    return null;
  }
  if (isSupportedFilterType(preprocessor)) {
    return preprocessor;
  }
  return PROCESSOR_TO_FILTER_MAP[preprocessor] ?? null;
};
