import i18n from 'i18next';

import type { ControlAdapterProcessorType, RequiredControlAdapterProcessorNode } from './types';

type ControlNetProcessorsDict = Record<
  ControlAdapterProcessorType,
  {
    type: ControlAdapterProcessorType | 'none';
    label: string;
    description: string;
    default: RequiredControlAdapterProcessorNode | { type: 'none' };
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
    default: {
      type: 'none',
    },
  },
  canny_image_processor: {
    type: 'canny_image_processor',
    get label() {
      return i18n.t('controlnet.canny');
    },
    get description() {
      return i18n.t('controlnet.cannyDescription');
    },
    default: {
      id: 'canny_image_processor',
      type: 'canny_image_processor',
      low_threshold: 100,
      high_threshold: 200,
    },
  },
  color_map_image_processor: {
    type: 'color_map_image_processor',
    get label() {
      return i18n.t('controlnet.colorMap');
    },
    get description() {
      return i18n.t('controlnet.colorMapDescription');
    },
    default: {
      id: 'color_map_image_processor',
      type: 'color_map_image_processor',
      color_map_tile_size: 64,
    },
  },
  content_shuffle_image_processor: {
    type: 'content_shuffle_image_processor',
    get label() {
      return i18n.t('controlnet.contentShuffle');
    },
    get description() {
      return i18n.t('controlnet.contentShuffleDescription');
    },
    default: {
      id: 'content_shuffle_image_processor',
      type: 'content_shuffle_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
      h: 512,
      w: 512,
      f: 256,
    },
  },
  depth_anything_image_processor: {
    type: 'depth_anything_image_processor',
    get label() {
      return i18n.t('controlnet.depthAnything');
    },
    get description() {
      return i18n.t('controlnet.depthAnythingDescription');
    },
    default: {
      id: 'depth_anything_image_processor',
      type: 'depth_anything_image_processor',
      model_size: 'small',
      resolution: 512,
      offload: false,
    },
  },
  hed_image_processor: {
    type: 'hed_image_processor',
    get label() {
      return i18n.t('controlnet.hed');
    },
    get description() {
      return i18n.t('controlnet.hedDescription');
    },
    default: {
      id: 'hed_image_processor',
      type: 'hed_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
      scribble: false,
    },
  },
  lineart_anime_image_processor: {
    type: 'lineart_anime_image_processor',
    get label() {
      return i18n.t('controlnet.lineartAnime');
    },
    get description() {
      return i18n.t('controlnet.lineartAnimeDescription');
    },
    default: {
      id: 'lineart_anime_image_processor',
      type: 'lineart_anime_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
    },
  },
  lineart_image_processor: {
    type: 'lineart_image_processor',
    get label() {
      return i18n.t('controlnet.lineart');
    },
    get description() {
      return i18n.t('controlnet.lineartDescription');
    },
    default: {
      id: 'lineart_image_processor',
      type: 'lineart_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
      coarse: false,
    },
  },
  mediapipe_face_processor: {
    type: 'mediapipe_face_processor',
    get label() {
      return i18n.t('controlnet.mediapipeFace');
    },
    get description() {
      return i18n.t('controlnet.mediapipeFaceDescription');
    },
    default: {
      id: 'mediapipe_face_processor',
      type: 'mediapipe_face_processor',
      max_faces: 1,
      min_confidence: 0.5,
    },
  },
  midas_depth_image_processor: {
    type: 'midas_depth_image_processor',
    get label() {
      return i18n.t('controlnet.depthMidas');
    },
    get description() {
      return i18n.t('controlnet.depthMidasDescription');
    },
    default: {
      id: 'midas_depth_image_processor',
      type: 'midas_depth_image_processor',
      a_mult: 2,
      bg_th: 0.1,
    },
  },
  mlsd_image_processor: {
    type: 'mlsd_image_processor',
    get label() {
      return i18n.t('controlnet.mlsd');
    },
    get description() {
      return i18n.t('controlnet.mlsdDescription');
    },
    default: {
      id: 'mlsd_image_processor',
      type: 'mlsd_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
      thr_d: 0.1,
      thr_v: 0.1,
    },
  },
  normalbae_image_processor: {
    type: 'normalbae_image_processor',
    get label() {
      return i18n.t('controlnet.normalBae');
    },
    get description() {
      return i18n.t('controlnet.normalBaeDescription');
    },
    default: {
      id: 'normalbae_image_processor',
      type: 'normalbae_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
    },
  },
  openpose_image_processor: {
    type: 'openpose_image_processor',
    get label() {
      return i18n.t('controlnet.openPose');
    },
    get description() {
      return i18n.t('controlnet.openPoseDescription');
    },
    default: {
      id: 'openpose_image_processor',
      type: 'openpose_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
      hand_and_face: false,
    },
  },
  dwpose_image_processor: {
    type: 'dwpose_image_processor',
    get label() {
      return i18n.t('controlnet.dwPose');
    },
    get description() {
      return i18n.t('controlnet.dwPoseDescription');
    },
    default: {
      id: 'dwpose_image_processor',
      type: 'dwpose_image_processor',
      draw_body: true,
      draw_face: false,
      draw_hands: false,
    },
  },
  pidi_image_processor: {
    type: 'pidi_image_processor',
    get label() {
      return i18n.t('controlnet.pidi');
    },
    get description() {
      return i18n.t('controlnet.pidiDescription');
    },
    default: {
      id: 'pidi_image_processor',
      type: 'pidi_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
      scribble: false,
      safe: false,
    },
  },
  zoe_depth_image_processor: {
    type: 'zoe_depth_image_processor',
    get label() {
      return i18n.t('controlnet.depthZoe');
    },
    get description() {
      return i18n.t('controlnet.depthZoeDescription');
    },
    default: {
      id: 'zoe_depth_image_processor',
      type: 'zoe_depth_image_processor',
    },
  },
};

export const CONTROLNET_MODEL_DEFAULT_PROCESSORS: {
  [key: string]: ControlAdapterProcessorType;
} = {
  canny: 'canny_image_processor',
  mlsd: 'mlsd_image_processor',
  depth: 'depth_anything_image_processor',
  bae: 'normalbae_image_processor',
  sketch: 'pidi_image_processor',
  scribble: 'lineart_image_processor',
  lineart: 'lineart_image_processor',
  lineart_anime: 'lineart_anime_image_processor',
  softedge: 'hed_image_processor',
  shuffle: 'content_shuffle_image_processor',
  openpose: 'dwpose_image_processor',
  mediapipe: 'mediapipe_face_processor',
  pidi: 'pidi_image_processor',
  zoe: 'zoe_depth_image_processor',
  color: 'color_map_image_processor',
};
