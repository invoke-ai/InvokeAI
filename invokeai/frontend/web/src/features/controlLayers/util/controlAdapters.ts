import { deepClone } from 'common/util/deepClone';
import type {
  ParameterControlNetModel,
  ParameterIPAdapterModel,
  ParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';
import { merge, omit } from 'lodash-es';
import type {
  BaseModelType,
  CannyImageProcessorInvocation,
  ColorMapImageProcessorInvocation,
  ContentShuffleImageProcessorInvocation,
  ControlNetModelConfig,
  DepthAnythingImageProcessorInvocation,
  DWOpenposeImageProcessorInvocation,
  Graph,
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
import { z } from 'zod';

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;

export type CannyProcessorConfig = Required<
  Pick<CannyImageProcessorInvocation, 'id' | 'type' | 'low_threshold' | 'high_threshold'>
>;
export type ColorMapProcessorConfig = Required<
  Pick<ColorMapImageProcessorInvocation, 'id' | 'type' | 'color_map_tile_size'>
>;
export type ContentShuffleProcessorConfig = Required<
  Pick<ContentShuffleImageProcessorInvocation, 'id' | 'type' | 'w' | 'h' | 'f'>
>;
export type DepthAnythingProcessorConfig = Required<
  Pick<DepthAnythingImageProcessorInvocation, 'id' | 'type' | 'model_size'>
>;
export type HedProcessorConfig = Required<Pick<HedImageProcessorInvocation, 'id' | 'type' | 'scribble'>>;
type LineartAnimeProcessorConfig = Required<Pick<LineartAnimeImageProcessorInvocation, 'id' | 'type'>>;
export type LineartProcessorConfig = Required<Pick<LineartImageProcessorInvocation, 'id' | 'type' | 'coarse'>>;
export type MediapipeFaceProcessorConfig = Required<
  Pick<MediapipeFaceProcessorInvocation, 'id' | 'type' | 'max_faces' | 'min_confidence'>
>;
export type MidasDepthProcessorConfig = Required<
  Pick<MidasDepthImageProcessorInvocation, 'id' | 'type' | 'a_mult' | 'bg_th'>
>;
export type MlsdProcessorConfig = Required<Pick<MlsdImageProcessorInvocation, 'id' | 'type' | 'thr_v' | 'thr_d'>>;
type NormalbaeProcessorConfig = Required<Pick<NormalbaeImageProcessorInvocation, 'id' | 'type'>>;
export type DWOpenposeProcessorConfig = Required<
  Pick<DWOpenposeImageProcessorInvocation, 'id' | 'type' | 'draw_body' | 'draw_face' | 'draw_hands'>
>;
export type PidiProcessorConfig = Required<Pick<PidiImageProcessorInvocation, 'id' | 'type' | 'safe' | 'scribble'>>;
type ZoeDepthProcessorConfig = Required<Pick<ZoeDepthImageProcessorInvocation, 'id' | 'type'>>;

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
  name: string;
  width: number;
  height: number;
};

type ControlAdapterBase = {
  id: string;
  weight: number;
  image: ImageWithDims | null;
  processedImage: ImageWithDims | null;
  isProcessingImage: boolean;
  processorConfig: ProcessorConfig | null;
  beginEndStepPct: [number, number];
};

const zControlModeV2 = z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']);
export type ControlModeV2 = z.infer<typeof zControlModeV2>;
export const isControlModeV2 = (v: unknown): v is ControlModeV2 => zControlModeV2.safeParse(v).success;

export type ControlNetConfigV2 = ControlAdapterBase & {
  type: 'controlnet';
  model: ParameterControlNetModel | null;
  controlMode: ControlModeV2;
};
export const isControlNetConfigV2 = (ca: ControlNetConfigV2 | T2IAdapterConfigV2): ca is ControlNetConfigV2 =>
  ca.type === 'controlnet';

export type T2IAdapterConfigV2 = ControlAdapterBase & {
  type: 't2i_adapter';
  model: ParameterT2IAdapterModel | null;
};
export const isT2IAdapterConfigV2 = (ca: ControlNetConfigV2 | T2IAdapterConfigV2): ca is T2IAdapterConfigV2 =>
  ca.type === 't2i_adapter';

const zCLIPVisionModelV2 = z.enum(['ViT-H', 'ViT-G']);
export type CLIPVisionModelV2 = z.infer<typeof zCLIPVisionModelV2>;
export const isCLIPVisionModelV2 = (v: unknown): v is CLIPVisionModelV2 => zCLIPVisionModelV2.safeParse(v).success;

const zIPMethodV2 = z.enum(['full', 'style', 'composition']);
export type IPMethodV2 = z.infer<typeof zIPMethodV2>;
export const isIPMethodV2 = (v: unknown): v is IPMethodV2 => zIPMethodV2.safeParse(v).success;

export type IPAdapterConfigV2 = {
  id: string;
  type: 'ip_adapter';
  weight: number;
  method: IPMethodV2;
  image: ImageWithDims | null;
  model: ParameterIPAdapterModel | null;
  clipVisionModel: CLIPVisionModelV2;
  beginEndStepPct: [number, number];
};

const zProcessorTypeV2 = z.enum([
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
export type ProcessorTypeV2 = z.infer<typeof zProcessorTypeV2>;
export const isProcessorTypeV2 = (v: unknown): v is ProcessorTypeV2 => zProcessorTypeV2.safeParse(v).success;

type ProcessorData<T extends ProcessorTypeV2> = {
  type: T;
  labelTKey: string;
  descriptionTKey: string;
  buildDefaults(baseModel?: BaseModelType): Extract<ProcessorConfig, { type: T }>;
  buildNode(
    image: ImageWithDims,
    config: Extract<ProcessorConfig, { type: T }>
  ): Extract<Graph['nodes'][string], { type: T }>;
};

const minDim = (image: ImageWithDims): number => Math.min(image.width, image.height);

type CAProcessorsData = {
  [key in ProcessorTypeV2]: ProcessorData<key>;
};
/**
 * A dict of ControlNet processors, including:
 * - label translation key
 * - description translation key
 * - a builder to create default values for the config
 * - a builder to create the node for the config
 *
 * TODO: Generate from the OpenAPI schema
 */
export const CA_PROCESSOR_DATA: CAProcessorsData = {
  canny_image_processor: {
    type: 'canny_image_processor',
    labelTKey: 'controlnet.canny',
    descriptionTKey: 'controlnet.cannyDescription',
    buildDefaults: () => ({
      id: 'canny_image_processor',
      type: 'canny_image_processor',
      low_threshold: 100,
      high_threshold: 200,
    }),
    buildNode: (image, config) => ({
      ...config,
      type: 'canny_image_processor',
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  color_map_image_processor: {
    type: 'color_map_image_processor',
    labelTKey: 'controlnet.colorMap',
    descriptionTKey: 'controlnet.colorMapDescription',
    buildDefaults: () => ({
      id: 'color_map_image_processor',
      type: 'color_map_image_processor',
      color_map_tile_size: 64,
    }),
    buildNode: (image, config) => ({
      ...config,
      type: 'color_map_image_processor',
      image: { image_name: image.name },
    }),
  },
  content_shuffle_image_processor: {
    type: 'content_shuffle_image_processor',
    labelTKey: 'controlnet.contentShuffle',
    descriptionTKey: 'controlnet.contentShuffleDescription',
    buildDefaults: (baseModel) => ({
      id: 'content_shuffle_image_processor',
      type: 'content_shuffle_image_processor',
      h: baseModel === 'sdxl' ? 1024 : 512,
      w: baseModel === 'sdxl' ? 1024 : 512,
      f: baseModel === 'sdxl' ? 512 : 256,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  depth_anything_image_processor: {
    type: 'depth_anything_image_processor',
    labelTKey: 'controlnet.depthAnything',
    descriptionTKey: 'controlnet.depthAnythingDescription',
    buildDefaults: () => ({
      id: 'depth_anything_image_processor',
      type: 'depth_anything_image_processor',
      model_size: 'small',
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      resolution: minDim(image),
    }),
  },
  hed_image_processor: {
    type: 'hed_image_processor',
    labelTKey: 'controlnet.hed',
    descriptionTKey: 'controlnet.hedDescription',
    buildDefaults: () => ({
      id: 'hed_image_processor',
      type: 'hed_image_processor',
      scribble: false,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  lineart_anime_image_processor: {
    type: 'lineart_anime_image_processor',
    labelTKey: 'controlnet.lineartAnime',
    descriptionTKey: 'controlnet.lineartAnimeDescription',
    buildDefaults: () => ({
      id: 'lineart_anime_image_processor',
      type: 'lineart_anime_image_processor',
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  lineart_image_processor: {
    type: 'lineart_image_processor',
    labelTKey: 'controlnet.lineart',
    descriptionTKey: 'controlnet.lineartDescription',
    buildDefaults: () => ({
      id: 'lineart_image_processor',
      type: 'lineart_image_processor',
      coarse: false,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  mediapipe_face_processor: {
    type: 'mediapipe_face_processor',
    labelTKey: 'controlnet.mediapipeFace',
    descriptionTKey: 'controlnet.mediapipeFaceDescription',
    buildDefaults: () => ({
      id: 'mediapipe_face_processor',
      type: 'mediapipe_face_processor',
      max_faces: 1,
      min_confidence: 0.5,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  midas_depth_image_processor: {
    type: 'midas_depth_image_processor',
    labelTKey: 'controlnet.depthMidas',
    descriptionTKey: 'controlnet.depthMidasDescription',
    buildDefaults: () => ({
      id: 'midas_depth_image_processor',
      type: 'midas_depth_image_processor',
      a_mult: 2,
      bg_th: 0.1,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  mlsd_image_processor: {
    type: 'mlsd_image_processor',
    labelTKey: 'controlnet.mlsd',
    descriptionTKey: 'controlnet.mlsdDescription',
    buildDefaults: () => ({
      id: 'mlsd_image_processor',
      type: 'mlsd_image_processor',
      thr_d: 0.1,
      thr_v: 0.1,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  normalbae_image_processor: {
    type: 'normalbae_image_processor',
    labelTKey: 'controlnet.normalBae',
    descriptionTKey: 'controlnet.normalBaeDescription',
    buildDefaults: () => ({
      id: 'normalbae_image_processor',
      type: 'normalbae_image_processor',
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  dw_openpose_image_processor: {
    type: 'dw_openpose_image_processor',
    labelTKey: 'controlnet.dwOpenpose',
    descriptionTKey: 'controlnet.dwOpenposeDescription',
    buildDefaults: () => ({
      id: 'dw_openpose_image_processor',
      type: 'dw_openpose_image_processor',
      draw_body: true,
      draw_face: false,
      draw_hands: false,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      image_resolution: minDim(image),
    }),
  },
  pidi_image_processor: {
    type: 'pidi_image_processor',
    labelTKey: 'controlnet.pidi',
    descriptionTKey: 'controlnet.pidiDescription',
    buildDefaults: () => ({
      id: 'pidi_image_processor',
      type: 'pidi_image_processor',
      scribble: false,
      safe: false,
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
      detect_resolution: minDim(image),
      image_resolution: minDim(image),
    }),
  },
  zoe_depth_image_processor: {
    type: 'zoe_depth_image_processor',
    labelTKey: 'controlnet.depthZoe',
    descriptionTKey: 'controlnet.depthZoeDescription',
    buildDefaults: () => ({
      id: 'zoe_depth_image_processor',
      type: 'zoe_depth_image_processor',
    }),
    buildNode: (image, config) => ({
      ...config,
      image: { image_name: image.name },
    }),
  },
};

export const initialControlNetV2: Omit<ControlNetConfigV2, 'id'> = {
  type: 'controlnet',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  controlMode: 'balanced',
  image: null,
  processedImage: null,
  isProcessingImage: false,
  processorConfig: CA_PROCESSOR_DATA.canny_image_processor.buildDefaults(),
};

export const initialT2IAdapterV2: Omit<T2IAdapterConfigV2, 'id'> = {
  type: 't2i_adapter',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  image: null,
  processedImage: null,
  isProcessingImage: false,
  processorConfig: CA_PROCESSOR_DATA.canny_image_processor.buildDefaults(),
};

export const initialIPAdapterV2: Omit<IPAdapterConfigV2, 'id'> = {
  type: 'ip_adapter',
  image: null,
  model: null,
  beginEndStepPct: [0, 1],
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
};

export const buildControlNet = (id: string, overrides?: Partial<ControlNetConfigV2>): ControlNetConfigV2 => {
  return merge(deepClone(initialControlNetV2), { id, ...overrides });
};

export const buildT2IAdapter = (id: string, overrides?: Partial<T2IAdapterConfigV2>): T2IAdapterConfigV2 => {
  return merge(deepClone(initialT2IAdapterV2), { id, ...overrides });
};

export const buildIPAdapter = (id: string, overrides?: Partial<IPAdapterConfigV2>): IPAdapterConfigV2 => {
  return merge(deepClone(initialIPAdapterV2), { id, ...overrides });
};

export const buildControlAdapterProcessorV2 = (
  modelConfig: ControlNetModelConfig | T2IAdapterModelConfig
): ProcessorConfig | null => {
  const defaultPreprocessor = modelConfig.default_settings?.preprocessor;
  if (!isProcessorTypeV2(defaultPreprocessor)) {
    return null;
  }
  const processorConfig = CA_PROCESSOR_DATA[defaultPreprocessor].buildDefaults(modelConfig.base);
  return processorConfig;
};

export const imageDTOToImageWithDims = ({ image_name, width, height }: ImageDTO): ImageWithDims => ({
  name: image_name,
  width,
  height,
});

export const t2iAdapterToControlNet = (t2iAdapter: T2IAdapterConfigV2): ControlNetConfigV2 => {
  return {
    ...deepClone(t2iAdapter),
    type: 'controlnet',
    controlMode: initialControlNetV2.controlMode,
  };
};

export const controlNetToT2IAdapter = (controlNet: ControlNetConfigV2): T2IAdapterConfigV2 => {
  return {
    ...omit(deepClone(controlNet), 'controlMode'),
    type: 't2i_adapter',
  };
};
