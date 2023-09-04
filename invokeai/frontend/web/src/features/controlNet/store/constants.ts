import {
  ControlNetProcessorType,
  RequiredControlNetProcessorNode,
} from './types';

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
    label: 'none',
    description: '',
    default: {
      type: 'none',
    },
  },
  canny_image_processor: {
    type: 'canny_image_processor',
    label: 'Canny',
    description: '',
    default: {
      id: 'canny_image_processor',
      type: 'canny_image_processor',
      low_threshold: 100,
      high_threshold: 200,
    },
  },
  content_shuffle_image_processor: {
    type: 'content_shuffle_image_processor',
    label: 'Content Shuffle',
    description: '',
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
    label: 'HED',
    description: '',
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
    label: 'Lineart Anime',
    description: '',
    default: {
      id: 'lineart_anime_image_processor',
      type: 'lineart_anime_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
    },
  },
  lineart_image_processor: {
    type: 'lineart_image_processor',
    label: 'Lineart',
    description: '',
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
    label: 'Mediapipe Face',
    description: '',
    default: {
      id: 'mediapipe_face_processor',
      type: 'mediapipe_face_processor',
      max_faces: 1,
      min_confidence: 0.5,
    },
  },
  midas_depth_image_processor: {
    type: 'midas_depth_image_processor',
    label: 'Depth (Midas)',
    description: '',
    default: {
      id: 'midas_depth_image_processor',
      type: 'midas_depth_image_processor',
      a_mult: 2,
      bg_th: 0.1,
    },
  },
  mlsd_image_processor: {
    type: 'mlsd_image_processor',
    label: 'M-LSD',
    description: '',
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
    label: 'Normal BAE',
    description: '',
    default: {
      id: 'normalbae_image_processor',
      type: 'normalbae_image_processor',
      detect_resolution: 512,
      image_resolution: 512,
    },
  },
  openpose_image_processor: {
    type: 'openpose_image_processor',
    label: 'Openpose',
    description: '',
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
    label: 'PIDI',
    description: '',
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
    label: 'Depth (Zoe)',
    description: '',
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
};
