import i18n from 'i18next';
import type { BaseModelType } from 'services/api/types';

import type { ControlAdapterProcessorType, RequiredControlAdapterProcessorNode } from './types';

type ControlNetProcessorsDict = Record<
  ControlAdapterProcessorType,
  {
    type: ControlAdapterProcessorType | 'none';
    label: string;
    description: string;
    buildDefaults(baseModel?: BaseModelType): RequiredControlAdapterProcessorNode | { type: 'none' };
  }
>;
/**
 * A dict of ControlNet processors, including:
 * - type
 * - label
 * - description
 * - default values
 *
 * TODO: Generate from the OpenAPI schema
 */
export const CONTROLNET_PROCESSORS: ControlNetProcessorsDict = {
  none: {
    type: 'none',
    get label() {
      return i18n.t('controlnet.none');
    },
    get description() {
      return i18n.t('controlnet.noneDescription');
    },
    buildDefaults: () => ({
      type: 'none',
    }),
  },
  canny_image_processor: {
    type: 'canny_image_processor',
    get label() {
      return i18n.t('controlnet.canny');
    },
    get description() {
      return i18n.t('controlnet.cannyDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'canny_image_processor',
      type: 'canny_image_processor',
      low_threshold: 100,
      high_threshold: 200,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
    }),
  },
  color_map_image_processor: {
    type: 'color_map_image_processor',
    get label() {
      return i18n.t('controlnet.colorMap');
    },
    get description() {
      return i18n.t('controlnet.colorMapDescription');
    },
    buildDefaults: () => ({
      id: 'color_map_image_processor',
      type: 'color_map_image_processor',
      color_map_tile_size: 64,
    }),
  },
  content_shuffle_image_processor: {
    type: 'content_shuffle_image_processor',
    get label() {
      return i18n.t('controlnet.contentShuffle');
    },
    get description() {
      return i18n.t('controlnet.contentShuffleDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'content_shuffle_image_processor',
      type: 'content_shuffle_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      h: baseModel === 'sdxl' ? 1024 : 512,
      w: baseModel === 'sdxl' ? 1024 : 512,
      f: baseModel === 'sdxl' ? 512 : 256,
    }),
  },
  depth_anything_image_processor: {
    type: 'depth_anything_image_processor',
    get label() {
      return i18n.t('controlnet.depthAnything');
    },
    get description() {
      return i18n.t('controlnet.depthAnythingDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'depth_anything_image_processor',
      type: 'depth_anything_image_processor',
      model_size: 'small',
      resolution: baseModel === 'sdxl' ? 1024 : 512,
    }),
  },
  hed_image_processor: {
    type: 'hed_image_processor',
    get label() {
      return i18n.t('controlnet.hed');
    },
    get description() {
      return i18n.t('controlnet.hedDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'hed_image_processor',
      type: 'hed_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      scribble: false,
    }),
  },
  lineart_anime_image_processor: {
    type: 'lineart_anime_image_processor',
    get label() {
      return i18n.t('controlnet.lineartAnime');
    },
    get description() {
      return i18n.t('controlnet.lineartAnimeDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'lineart_anime_image_processor',
      type: 'lineart_anime_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
    }),
  },
  lineart_image_processor: {
    type: 'lineart_image_processor',
    get label() {
      return i18n.t('controlnet.lineart');
    },
    get description() {
      return i18n.t('controlnet.lineartDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'lineart_image_processor',
      type: 'lineart_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      coarse: false,
    }),
  },
  mediapipe_face_processor: {
    type: 'mediapipe_face_processor',
    get label() {
      return i18n.t('controlnet.mediapipeFace');
    },
    get description() {
      return i18n.t('controlnet.mediapipeFaceDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'mediapipe_face_processor',
      type: 'mediapipe_face_processor',
      max_faces: 1,
      min_confidence: 0.5,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
    }),
  },
  midas_depth_image_processor: {
    type: 'midas_depth_image_processor',
    get label() {
      return i18n.t('controlnet.depthMidas');
    },
    get description() {
      return i18n.t('controlnet.depthMidasDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'midas_depth_image_processor',
      type: 'midas_depth_image_processor',
      a_mult: 2,
      bg_th: 0.1,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
    }),
  },
  mlsd_image_processor: {
    type: 'mlsd_image_processor',
    get label() {
      return i18n.t('controlnet.mlsd');
    },
    get description() {
      return i18n.t('controlnet.mlsdDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'mlsd_image_processor',
      type: 'mlsd_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      thr_d: 0.1,
      thr_v: 0.1,
    }),
  },
  normalbae_image_processor: {
    type: 'normalbae_image_processor',
    get label() {
      return i18n.t('controlnet.normalBae');
    },
    get description() {
      return i18n.t('controlnet.normalBaeDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'normalbae_image_processor',
      type: 'normalbae_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
    }),
  },
  dw_openpose_image_processor: {
    type: 'dw_openpose_image_processor',
    get label() {
      return i18n.t('controlnet.dwOpenpose');
    },
    get description() {
      return i18n.t('controlnet.dwOpenposeDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'dw_openpose_image_processor',
      type: 'dw_openpose_image_processor',
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      draw_body: true,
      draw_face: false,
      draw_hands: false,
    }),
  },
  pidi_image_processor: {
    type: 'pidi_image_processor',
    get label() {
      return i18n.t('controlnet.pidi');
    },
    get description() {
      return i18n.t('controlnet.pidiDescription');
    },
    buildDefaults: (baseModel?: BaseModelType) => ({
      id: 'pidi_image_processor',
      type: 'pidi_image_processor',
      detect_resolution: baseModel === 'sdxl' ? 1024 : 512,
      image_resolution: baseModel === 'sdxl' ? 1024 : 512,
      scribble: false,
      safe: false,
    }),
  },
  zoe_depth_image_processor: {
    type: 'zoe_depth_image_processor',
    get label() {
      return i18n.t('controlnet.depthZoe');
    },
    get description() {
      return i18n.t('controlnet.depthZoeDescription');
    },
    buildDefaults: () => ({
      id: 'zoe_depth_image_processor',
      type: 'zoe_depth_image_processor',
    }),
  },
};
