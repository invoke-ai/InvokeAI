import {
  ControlNetProcessorType,
  RequiredControlNetProcessorNode,
} from './types';
import i18n from 'i18next';

type ControlNetProcessorsDict = Record<
  ControlNetProcessorType,
  {
    type: ControlNetProcessorType | 'none';
    label: string;
    description: string;
    default: RequiredControlNetProcessorNode | { type: 'none' };
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
  [key: string]: ControlNetProcessorType;
} = {
  canny: 'canny_image_processor',
  mlsd: 'mlsd_image_processor',
  depth: 'midas_depth_image_processor',
  bae: 'normalbae_image_processor',
  lineart: 'lineart_image_processor',
  lineart_anime: 'lineart_anime_image_processor',
  softedge: 'hed_image_processor',
  shuffle: 'content_shuffle_image_processor',
  openpose: 'openpose_image_processor',
  mediapipe: 'mediapipe_face_processor',
  pidi: 'pidi_image_processor',
  zoe: 'zoe_depth_image_processor',
};
