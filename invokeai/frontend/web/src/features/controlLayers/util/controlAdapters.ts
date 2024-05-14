import { deepClone } from 'common/util/deepClone';
import { zModelIdentifierField } from 'features/nodes/types/common';
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

const zId = z.string().min(1);

const zCannyProcessorConfig = z.object({
  id: zId,
  type: z.literal('canny_image_processor'),
  low_threshold: z.number().int().gte(0).lte(255),
  high_threshold: z.number().int().gte(0).lte(255),
});
export type _CannyProcessorConfig = Required<
  Pick<CannyImageProcessorInvocation, 'id' | 'type' | 'low_threshold' | 'high_threshold'>
>;
export type CannyProcessorConfig = z.infer<typeof zCannyProcessorConfig>;

const zColorMapProcessorConfig = z.object({
  id: zId,
  type: z.literal('color_map_image_processor'),
  color_map_tile_size: z.number().int().gte(1),
});
export type _ColorMapProcessorConfig = Required<
  Pick<ColorMapImageProcessorInvocation, 'id' | 'type' | 'color_map_tile_size'>
>;
export type ColorMapProcessorConfig = z.infer<typeof zColorMapProcessorConfig>;

const zContentShuffleProcessorConfig = z.object({
  id: zId,
  type: z.literal('content_shuffle_image_processor'),
  w: z.number().int().gte(0),
  h: z.number().int().gte(0),
  f: z.number().int().gte(0),
});
export type _ContentShuffleProcessorConfig = Required<
  Pick<ContentShuffleImageProcessorInvocation, 'id' | 'type' | 'w' | 'h' | 'f'>
>;
export type ContentShuffleProcessorConfig = z.infer<typeof zContentShuffleProcessorConfig>;

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;
const zDepthAnythingProcessorConfig = z.object({
  id: zId,
  type: z.literal('depth_anything_image_processor'),
  model_size: zDepthAnythingModelSize,
});
export type _DepthAnythingProcessorConfig = Required<
  Pick<DepthAnythingImageProcessorInvocation, 'id' | 'type' | 'model_size'>
>;
export type DepthAnythingProcessorConfig = z.infer<typeof zDepthAnythingProcessorConfig>;

const zHedProcessorConfig = z.object({
  id: zId,
  type: z.literal('hed_image_processor'),
  scribble: z.boolean(),
});
export type _HedProcessorConfig = Required<Pick<HedImageProcessorInvocation, 'id' | 'type' | 'scribble'>>;
export type HedProcessorConfig = z.infer<typeof zHedProcessorConfig>;

const zLineartAnimeProcessorConfig = z.object({
  id: zId,
  type: z.literal('lineart_anime_image_processor'),
});
export type _LineartAnimeProcessorConfig = Required<Pick<LineartAnimeImageProcessorInvocation, 'id' | 'type'>>;
export type LineartAnimeProcessorConfig = z.infer<typeof zLineartAnimeProcessorConfig>;

const zLineartProcessorConfig = z.object({
  id: zId,
  type: z.literal('lineart_image_processor'),
  coarse: z.boolean(),
});
export type _LineartProcessorConfig = Required<Pick<LineartImageProcessorInvocation, 'id' | 'type' | 'coarse'>>;
export type LineartProcessorConfig = z.infer<typeof zLineartProcessorConfig>;

const zMediapipeFaceProcessorConfig = z.object({
  id: zId,
  type: z.literal('mediapipe_face_processor'),
  max_faces: z.number().int().gte(1),
  min_confidence: z.number().gte(0).lte(1),
});
export type _MediapipeFaceProcessorConfig = Required<
  Pick<MediapipeFaceProcessorInvocation, 'id' | 'type' | 'max_faces' | 'min_confidence'>
>;
export type MediapipeFaceProcessorConfig = z.infer<typeof zMediapipeFaceProcessorConfig>;

const zMidasDepthProcessorConfig = z.object({
  id: zId,
  type: z.literal('midas_depth_image_processor'),
  a_mult: z.number().gte(0),
  bg_th: z.number().gte(0),
});
export type _MidasDepthProcessorConfig = Required<
  Pick<MidasDepthImageProcessorInvocation, 'id' | 'type' | 'a_mult' | 'bg_th'>
>;
export type MidasDepthProcessorConfig = z.infer<typeof zMidasDepthProcessorConfig>;

const zMlsdProcessorConfig = z.object({
  id: zId,
  type: z.literal('mlsd_image_processor'),
  thr_v: z.number().gte(0),
  thr_d: z.number().gte(0),
});
export type _MlsdProcessorConfig = Required<Pick<MlsdImageProcessorInvocation, 'id' | 'type' | 'thr_v' | 'thr_d'>>;
export type MlsdProcessorConfig = z.infer<typeof zMlsdProcessorConfig>;

const zNormalbaeProcessorConfig = z.object({
  id: zId,
  type: z.literal('normalbae_image_processor'),
});
export type _NormalbaeProcessorConfig = Required<Pick<NormalbaeImageProcessorInvocation, 'id' | 'type'>>;
export type NormalbaeProcessorConfig = z.infer<typeof zNormalbaeProcessorConfig>;

const zDWOpenposeProcessorConfig = z.object({
  id: zId,
  type: z.literal('dw_openpose_image_processor'),
  draw_body: z.boolean(),
  draw_face: z.boolean(),
  draw_hands: z.boolean(),
});
export type _DWOpenposeProcessorConfig = Required<
  Pick<DWOpenposeImageProcessorInvocation, 'id' | 'type' | 'draw_body' | 'draw_face' | 'draw_hands'>
>;
export type DWOpenposeProcessorConfig = z.infer<typeof zDWOpenposeProcessorConfig>;

const zPidiProcessorConfig = z.object({
  id: zId,
  type: z.literal('pidi_image_processor'),
  safe: z.boolean(),
  scribble: z.boolean(),
});
export type _PidiProcessorConfig = Required<Pick<PidiImageProcessorInvocation, 'id' | 'type' | 'safe' | 'scribble'>>;
export type PidiProcessorConfig = z.infer<typeof zPidiProcessorConfig>;

const zZoeDepthProcessorConfig = z.object({
  id: zId,
  type: z.literal('zoe_depth_image_processor'),
});
export type _ZoeDepthProcessorConfig = Required<Pick<ZoeDepthImageProcessorInvocation, 'id' | 'type'>>;
export type ZoeDepthProcessorConfig = z.infer<typeof zZoeDepthProcessorConfig>;

const zProcessorConfig = z.discriminatedUnion('type', [
  zCannyProcessorConfig,
  zColorMapProcessorConfig,
  zContentShuffleProcessorConfig,
  zDepthAnythingProcessorConfig,
  zHedProcessorConfig,
  zLineartAnimeProcessorConfig,
  zLineartProcessorConfig,
  zMediapipeFaceProcessorConfig,
  zMidasDepthProcessorConfig,
  zMlsdProcessorConfig,
  zNormalbaeProcessorConfig,
  zDWOpenposeProcessorConfig,
  zPidiProcessorConfig,
  zZoeDepthProcessorConfig,
]);
export type ProcessorConfig = z.infer<typeof zProcessorConfig>;

export const zImageWithDims = z.object({
  name: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
export type ImageWithDims = z.infer<typeof zImageWithDims>;

const zBeginEndStepPct = z
  .tuple([z.number().gte(0).lte(1), z.number().gte(0).lte(1)])
  .refine(([begin, end]) => begin < end, {
    message: 'Begin must be less than end',
  });

const zControlAdapterBase = z.object({
  id: zId,
  weight: z.number().gte(0).lte(1),
  image: zImageWithDims.nullable(),
  processedImage: zImageWithDims.nullable(),
  processorConfig: zProcessorConfig.nullable(),
  processorPendingBatchId: z.string().nullable().default(null),
  beginEndStepPct: zBeginEndStepPct,
});

const zControlModeV2 = z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']);
export type ControlModeV2 = z.infer<typeof zControlModeV2>;
export const isControlModeV2 = (v: unknown): v is ControlModeV2 => zControlModeV2.safeParse(v).success;

export const zControlNetConfigV2 = zControlAdapterBase.extend({
  type: z.literal('controlnet'),
  model: zModelIdentifierField.nullable(),
  controlMode: zControlModeV2,
});
export type ControlNetConfigV2 = z.infer<typeof zControlNetConfigV2>;

export const zT2IAdapterConfigV2 = zControlAdapterBase.extend({
  type: z.literal('t2i_adapter'),
  model: zModelIdentifierField.nullable(),
});
export type T2IAdapterConfigV2 = z.infer<typeof zT2IAdapterConfigV2>;

const zCLIPVisionModelV2 = z.enum(['ViT-H', 'ViT-G']);
export type CLIPVisionModelV2 = z.infer<typeof zCLIPVisionModelV2>;
export const isCLIPVisionModelV2 = (v: unknown): v is CLIPVisionModelV2 => zCLIPVisionModelV2.safeParse(v).success;

const zIPMethodV2 = z.enum(['full', 'style', 'composition']);
export type IPMethodV2 = z.infer<typeof zIPMethodV2>;
export const isIPMethodV2 = (v: unknown): v is IPMethodV2 => zIPMethodV2.safeParse(v).success;

export const zIPAdapterConfigV2 = z.object({
  id: zId,
  type: z.literal('ip_adapter'),
  weight: z.number().gte(0).lte(1),
  method: zIPMethodV2,
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  clipVisionModel: zCLIPVisionModelV2,
  beginEndStepPct: zBeginEndStepPct,
});
export type IPAdapterConfigV2 = z.infer<typeof zIPAdapterConfigV2>;

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
  processorConfig: CA_PROCESSOR_DATA.canny_image_processor.buildDefaults(),
  processorPendingBatchId: null,
};

export const initialT2IAdapterV2: Omit<T2IAdapterConfigV2, 'id'> = {
  type: 't2i_adapter',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  image: null,
  processedImage: null,
  processorConfig: CA_PROCESSOR_DATA.canny_image_processor.buildDefaults(),
  processorPendingBatchId: null,
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
