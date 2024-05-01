import { deepClone } from 'common/util/deepClone';
import type {
  ParameterControlNetModel,
  ParameterIPAdapterModel,
  ParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';
import { merge } from 'lodash-es';
import type {
  BaseModelType,
  CannyImageProcessorInvocation,
  ColorMapImageProcessorInvocation,
  ContentShuffleImageProcessorInvocation,
  ControlNetModelConfig,
  DepthAnythingImageProcessorInvocation,
  DWOpenposeImageProcessorInvocation,
  HedImageProcessorInvocation,
  ImageDTO,
  LineartAnimeImageProcessorInvocation,
  LineartImageProcessorInvocation,
  MediapipeFaceProcessorInvocation,
  MidasDepthImageProcessorInvocation,
  MlsdImageProcessorInvocation,
  NormalbaeImageProcessorInvocation,
  PidiImageProcessorInvocation,
  T2IAdapterModelConfig,
  ZoeDepthImageProcessorInvocation,
} from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;

export type CannyProcessorConfig = Required<
  Pick<CannyImageProcessorInvocation, 'type' | 'low_threshold' | 'high_threshold'>
>;
export type ColorMapProcessorConfig = Required<Pick<ColorMapImageProcessorInvocation, 'type' | 'color_map_tile_size'>>;
export type ContentShuffleProcessorConfig = Required<
  Pick<ContentShuffleImageProcessorInvocation, 'type' | 'w' | 'h' | 'f'>
>;
export type DepthAnythingProcessorConfig = Required<Pick<DepthAnythingImageProcessorInvocation, 'type' | 'model_size'>>;
export type HedProcessorConfig = Required<Pick<HedImageProcessorInvocation, 'type' | 'scribble'>>;
export type LineartAnimeProcessorConfig = Required<Pick<LineartAnimeImageProcessorInvocation, 'type'>>;
export type LineartProcessorConfig = Required<Pick<LineartImageProcessorInvocation, 'type' | 'coarse'>>;
export type MediapipeFaceProcessorConfig = Required<
  Pick<MediapipeFaceProcessorInvocation, 'type' | 'max_faces' | 'min_confidence'>
>;
export type MidasDepthProcessorConfig = Required<Pick<MidasDepthImageProcessorInvocation, 'type' | 'a_mult' | 'bg_th'>>;
export type MlsdProcessorConfig = Required<Pick<MlsdImageProcessorInvocation, 'type' | 'thr_v' | 'thr_d'>>;
export type NormalbaeProcessorConfig = Required<Pick<NormalbaeImageProcessorInvocation, 'type'>>;
export type DWOpenposeProcessorConfig = Required<
  Pick<DWOpenposeImageProcessorInvocation, 'type' | 'draw_body' | 'draw_face' | 'draw_hands'>
>;
export type PidiProcessorConfig = Required<Pick<PidiImageProcessorInvocation, 'type' | 'safe' | 'scribble'>>;
export type ZoeDepthProcessorConfig = Required<Pick<ZoeDepthImageProcessorInvocation, 'type'>>;

export type ProcessorConfig =
  | CannyProcessorConfig
  | ColorMapProcessorConfig
  | ContentShuffleProcessorConfig
  | DepthAnythingProcessorConfig
  | HedProcessorConfig
  | LineartAnimeProcessorConfig
  | LineartProcessorConfig
  | MediapipeFaceProcessorConfig
  | MidasDepthProcessorConfig
  | MlsdProcessorConfig
  | NormalbaeProcessorConfig
  | DWOpenposeProcessorConfig
  | PidiProcessorConfig
  | ZoeDepthProcessorConfig;

export type ImageWithDims = {
  imageName: string;
  width: number;
  height: number;
};

type ControlAdapterBase = {
  id: string;
  isEnabled: boolean;
  weight: number;
  image: ImageWithDims | null;
  processedImage: ImageWithDims | null;
  processorConfig: ProcessorConfig | null;
  beginEndStepPct: [number, number];
};

const zControlMode = z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']);
export type ControlMode = z.infer<typeof zControlMode>;
export const isControlMode = (v: unknown): v is ControlMode => zControlMode.safeParse(v).success;

export type ControlNetConfig = ControlAdapterBase & {
  type: 'controlnet';
  model: ParameterControlNetModel | null;
  controlMode: ControlMode;
};

export type T2IAdapterConfig = ControlAdapterBase & {
  type: 't2i_adapter';
  model: ParameterT2IAdapterModel | null;
};

const zCLIPVisionModel = z.enum(['ViT-H', 'ViT-G']);
export type CLIPVisionModel = z.infer<typeof zCLIPVisionModel>;
export const isCLIPVisionModel = (v: unknown): v is CLIPVisionModel => zCLIPVisionModel.safeParse(v).success;

const zIPMethod = z.enum(['full', 'style', 'composition']);
export type IPMethod = z.infer<typeof zIPMethod>;
export const isIPMethod = (v: unknown): v is IPMethod => zIPMethod.safeParse(v).success;

export type IPAdapterConfig = {
  id: string;
  type: 'ip_adapter';
  isEnabled: boolean;
  weight: number;
  method: IPMethod;
  image: ImageWithDims | null;
  model: ParameterIPAdapterModel | null;
  clipVisionModel: CLIPVisionModel;
  beginEndStepPct: [number, number];
};

type ProcessorData<T extends ProcessorConfig['type']> = {
  labelTKey: string;
  descriptionTKey: string;
  buildDefaults(baseModel?: BaseModelType): Extract<ProcessorConfig, { type: T }>;
};

type ControlNetProcessorsDict = {
  [key in ProcessorConfig['type']]: ProcessorData<key>;
};
/**
 * A dict of ControlNet processors, including:
 * - label translation key
 * - description translation key
 * - a builder to create default values for the config
 *
 * TODO: Generate from the OpenAPI schema
 */
export const CONTROLNET_PROCESSORS: ControlNetProcessorsDict = {
  canny_image_processor: {
    labelTKey: 'controlnet.canny',
    descriptionTKey: 'controlnet.cannyDescription',
    buildDefaults: () => ({
      id: `canny_image_processor_${uuidv4()}`,
      type: 'canny_image_processor',
      low_threshold: 100,
      high_threshold: 200,
    }),
  },
  color_map_image_processor: {
    labelTKey: 'controlnet.colorMap',
    descriptionTKey: 'controlnet.colorMapDescription',
    buildDefaults: () => ({
      id: `color_map_image_processor_${uuidv4()}`,
      type: 'color_map_image_processor',
      color_map_tile_size: 64,
    }),
  },
  content_shuffle_image_processor: {
    labelTKey: 'controlnet.contentShuffle',
    descriptionTKey: 'controlnet.contentShuffleDescription',
    buildDefaults: (baseModel) => ({
      id: `content_shuffle_image_processor_${uuidv4()}`,
      type: 'content_shuffle_image_processor',
      h: baseModel === 'sdxl' ? 1024 : 512,
      w: baseModel === 'sdxl' ? 1024 : 512,
      f: baseModel === 'sdxl' ? 512 : 256,
    }),
  },
  depth_anything_image_processor: {
    labelTKey: 'controlnet.depthAnything',
    descriptionTKey: 'controlnet.depthAnythingDescription',
    buildDefaults: () => ({
      id: `depth_anything_image_processor_${uuidv4()}`,
      type: 'depth_anything_image_processor',
      model_size: 'small',
    }),
  },
  hed_image_processor: {
    labelTKey: 'controlnet.hed',
    descriptionTKey: 'controlnet.hedDescription',
    buildDefaults: () => ({
      id: `hed_image_processor_${uuidv4()}`,
      type: 'hed_image_processor',
      scribble: false,
    }),
  },
  lineart_anime_image_processor: {
    labelTKey: 'controlnet.lineartAnime',
    descriptionTKey: 'controlnet.lineartAnimeDescription',
    buildDefaults: () => ({
      id: `lineart_anime_image_processor_${uuidv4()}`,
      type: 'lineart_anime_image_processor',
    }),
  },
  lineart_image_processor: {
    labelTKey: 'controlnet.lineart',
    descriptionTKey: 'controlnet.lineartDescription',
    buildDefaults: () => ({
      id: `lineart_image_processor_${uuidv4()}`,
      type: 'lineart_image_processor',
      coarse: false,
    }),
  },
  mediapipe_face_processor: {
    labelTKey: 'controlnet.mediapipeFace',
    descriptionTKey: 'controlnet.mediapipeFaceDescription',
    buildDefaults: () => ({
      id: `mediapipe_face_processor_${uuidv4()}`,
      type: 'mediapipe_face_processor',
      max_faces: 1,
      min_confidence: 0.5,
    }),
  },
  midas_depth_image_processor: {
    labelTKey: 'controlnet.depthMidas',
    descriptionTKey: 'controlnet.depthMidasDescription',
    buildDefaults: () => ({
      id: `midas_depth_image_processor_${uuidv4()}`,
      type: 'midas_depth_image_processor',
      a_mult: 2,
      bg_th: 0.1,
    }),
  },
  mlsd_image_processor: {
    labelTKey: 'controlnet.mlsd',
    descriptionTKey: 'controlnet.mlsdDescription',
    buildDefaults: () => ({
      id: `mlsd_image_processor_${uuidv4()}`,
      type: 'mlsd_image_processor',
      thr_d: 0.1,
      thr_v: 0.1,
    }),
  },
  normalbae_image_processor: {
    labelTKey: 'controlnet.normalBae',
    descriptionTKey: 'controlnet.normalBaeDescription',
    buildDefaults: () => ({
      id: `normalbae_image_processor_${uuidv4()}`,
      type: 'normalbae_image_processor',
    }),
  },
  dw_openpose_image_processor: {
    labelTKey: 'controlnet.dwOpenpose',
    descriptionTKey: 'controlnet.dwOpenposeDescription',
    buildDefaults: () => ({
      id: `dw_openpose_image_processor_${uuidv4()}`,
      type: 'dw_openpose_image_processor',
      draw_body: true,
      draw_face: false,
      draw_hands: false,
    }),
  },
  pidi_image_processor: {
    labelTKey: 'controlnet.pidi',
    descriptionTKey: 'controlnet.pidiDescription',
    buildDefaults: () => ({
      id: `pidi_image_processor_${uuidv4()}`,
      type: 'pidi_image_processor',
      scribble: false,
      safe: false,
    }),
  },
  zoe_depth_image_processor: {
    labelTKey: 'controlnet.depthZoe',
    descriptionTKey: 'controlnet.depthZoeDescription',
    buildDefaults: () => ({
      id: `zoe_depth_image_processor_${uuidv4()}`,
      type: 'zoe_depth_image_processor',
    }),
  },
};
export const zProcessorType = z.enum([
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
  'zoe_depth_image_processor',
]);
export type ProcessorType = z.infer<typeof zProcessorType>;
export const isProcessorType = (v: unknown): v is ProcessorType => zProcessorType.safeParse(v).success;

export const initialControlNet: Omit<ControlNetConfig, 'id'> = {
  type: 'controlnet',
  isEnabled: true,
  model: null,
  weight: 1,
  beginEndStepPct: [0, 0],
  controlMode: 'balanced',
  image: null,
  processedImage: null,
  processorConfig: CONTROLNET_PROCESSORS.canny_image_processor.buildDefaults(),
};

export const initialT2IAdapter: Omit<T2IAdapterConfig, 'id'> = {
  type: 't2i_adapter',
  isEnabled: true,
  model: null,
  weight: 1,
  beginEndStepPct: [0, 0],
  image: null,
  processedImage: null,
  processorConfig: CONTROLNET_PROCESSORS.canny_image_processor.buildDefaults(),
};

export const initialIPAdapter: Omit<IPAdapterConfig, 'id'> = {
  type: 'ip_adapter',
  isEnabled: true,
  image: null,
  model: null,
  beginEndStepPct: [0, 0],
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
};

export const buildControlNet = (id: string, overrides?: Partial<ControlNetConfig>): ControlNetConfig => {
  return merge(deepClone(initialControlNet), { id, ...overrides });
};

export const buildT2IAdapter = (id: string, overrides?: Partial<T2IAdapterConfig>): T2IAdapterConfig => {
  return merge(deepClone(initialT2IAdapter), { id, ...overrides });
};

export const buildIPAdapter = (id: string, overrides?: Partial<IPAdapterConfig>): IPAdapterConfig => {
  return merge(deepClone(initialIPAdapter), { id, ...overrides });
};

export const buildControlAdapterProcessor = (
  modelConfig: ControlNetModelConfig | T2IAdapterModelConfig
): ProcessorConfig | null => {
  const defaultPreprocessor = modelConfig.default_settings?.preprocessor;
  if (!isProcessorType(defaultPreprocessor)) {
    return null;
  }
  const processorConfig = CONTROLNET_PROCESSORS[defaultPreprocessor].buildDefaults(modelConfig.base);
  return processorConfig;
};

export const imageDTOToImageWithDims = ({ image_name, width, height }: ImageDTO): ImageWithDims => ({
  imageName: image_name,
  width,
  height,
});
