import type { SerializableObject } from 'common/types';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { AspectRatioState } from 'features/parameters/components/DocumentSize/types';
import type { ParameterHeight, ParameterLoRAModel, ParameterWidth } from 'features/parameters/types/parameterSchemas';
import { zParameterNegativePrompt, zParameterPositivePrompt } from 'features/parameters/types/parameterSchemas';
import type { AnyInvocation, BaseModelType, ImageDTO, S } from 'services/api/types';
import { z } from 'zod';

const zId = z.string().min(1);
const zName = z.string().min(1).nullable();

const zImageWithDims = z.object({
  image_name: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
export type ImageWithDims = z.infer<typeof zImageWithDims>;

const zBeginEndStepPct = z
  .tuple([z.number().gte(0).lte(1), z.number().gte(0).lte(1)])
  .refine(([begin, end]) => begin < end, {
    message: 'Begin must be less than end',
  });

const zControlModeV2 = z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']);
export type ControlModeV2 = z.infer<typeof zControlModeV2>;
export const isControlModeV2 = (v: unknown): v is ControlModeV2 => zControlModeV2.safeParse(v).success;

const zCLIPVisionModelV2 = z.enum(['ViT-H', 'ViT-G']);
export type CLIPVisionModelV2 = z.infer<typeof zCLIPVisionModelV2>;
export const isCLIPVisionModelV2 = (v: unknown): v is CLIPVisionModelV2 => zCLIPVisionModelV2.safeParse(v).success;

const zIPMethodV2 = z.enum(['full', 'style', 'composition']);
export type IPMethodV2 = z.infer<typeof zIPMethodV2>;
export const isIPMethodV2 = (v: unknown): v is IPMethodV2 => zIPMethodV2.safeParse(v).success;

const zCannyProcessorConfig = z.object({
  id: zId,
  type: z.literal('canny_image_processor'),
  low_threshold: z.number().int().gte(0).lte(255),
  high_threshold: z.number().int().gte(0).lte(255),
});
export type CannyProcessorConfig = z.infer<typeof zCannyProcessorConfig>;

const zColorMapProcessorConfig = z.object({
  id: zId,
  type: z.literal('color_map_image_processor'),
  color_map_tile_size: z.number().int().gte(1),
});
export type ColorMapProcessorConfig = z.infer<typeof zColorMapProcessorConfig>;

const zContentShuffleProcessorConfig = z.object({
  id: zId,
  type: z.literal('content_shuffle_image_processor'),
  w: z.number().int().gte(0),
  h: z.number().int().gte(0),
  f: z.number().int().gte(0),
});
export type ContentShuffleProcessorConfig = z.infer<typeof zContentShuffleProcessorConfig>;

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small', 'small_v2']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;
const zDepthAnythingProcessorConfig = z.object({
  id: zId,
  type: z.literal('depth_anything_image_processor'),
  model_size: zDepthAnythingModelSize,
});
export type DepthAnythingProcessorConfig = z.infer<typeof zDepthAnythingProcessorConfig>;

const zHedProcessorConfig = z.object({
  id: zId,
  type: z.literal('hed_image_processor'),
  scribble: z.boolean(),
});
export type HedProcessorConfig = z.infer<typeof zHedProcessorConfig>;

const zLineartAnimeProcessorConfig = z.object({
  id: zId,
  type: z.literal('lineart_anime_image_processor'),
});
export type LineartAnimeProcessorConfig = z.infer<typeof zLineartAnimeProcessorConfig>;

const zLineartProcessorConfig = z.object({
  id: zId,
  type: z.literal('lineart_image_processor'),
  coarse: z.boolean(),
});
export type LineartProcessorConfig = z.infer<typeof zLineartProcessorConfig>;

const zMediapipeFaceProcessorConfig = z.object({
  id: zId,
  type: z.literal('mediapipe_face_processor'),
  max_faces: z.number().int().gte(1),
  min_confidence: z.number().gte(0).lte(1),
});
export type MediapipeFaceProcessorConfig = z.infer<typeof zMediapipeFaceProcessorConfig>;

const zMidasDepthProcessorConfig = z.object({
  id: zId,
  type: z.literal('midas_depth_image_processor'),
  a_mult: z.number().gte(0),
  bg_th: z.number().gte(0),
});
export type MidasDepthProcessorConfig = z.infer<typeof zMidasDepthProcessorConfig>;

const zMlsdProcessorConfig = z.object({
  id: zId,
  type: z.literal('mlsd_image_processor'),
  thr_v: z.number().gte(0),
  thr_d: z.number().gte(0),
});
export type MlsdProcessorConfig = z.infer<typeof zMlsdProcessorConfig>;

const zNormalbaeProcessorConfig = z.object({
  id: zId,
  type: z.literal('normalbae_image_processor'),
});
export type NormalbaeProcessorConfig = z.infer<typeof zNormalbaeProcessorConfig>;

const zDWOpenposeProcessorConfig = z.object({
  id: zId,
  type: z.literal('dw_openpose_image_processor'),
  draw_body: z.boolean(),
  draw_face: z.boolean(),
  draw_hands: z.boolean(),
});
export type DWOpenposeProcessorConfig = z.infer<typeof zDWOpenposeProcessorConfig>;

const zPidiProcessorConfig = z.object({
  id: zId,
  type: z.literal('pidi_image_processor'),
  safe: z.boolean(),
  scribble: z.boolean(),
});
export type PidiProcessorConfig = z.infer<typeof zPidiProcessorConfig>;

const zZoeDepthProcessorConfig = z.object({
  id: zId,
  type: z.literal('zoe_depth_image_processor'),
});
export type ZoeDepthProcessorConfig = z.infer<typeof zZoeDepthProcessorConfig>;

const zFilterConfig = z.discriminatedUnion('type', [
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
export type FilterConfig = z.infer<typeof zFilterConfig>;

const zFilterType = z.enum([
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
export type FilterType = z.infer<typeof zFilterType>;
export const isFilterType = (v: unknown): v is FilterType => zFilterType.safeParse(v).success;

const minDim = (image: ImageWithDims): number => Math.min(image.width, image.height);

type ImageFilterData<T extends FilterConfig['type']> = {
  type: T;
  labelTKey: string;
  descriptionTKey: string;
  buildDefaults(baseModel?: BaseModelType): Extract<FilterConfig, { type: T }>;
  buildNode(imageDTO: ImageWithDims, config: Extract<FilterConfig, { type: T }>): Extract<AnyInvocation, { type: T }>;
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
export const IMAGE_FILTERS: { [key in FilterConfig['type']]: ImageFilterData<key> } = {
  canny_image_processor: {
    type: 'canny_image_processor',
    labelTKey: 'controlnet.canny',
    descriptionTKey: 'controlnet.cannyDescription',
    buildDefaults: (): CannyProcessorConfig => ({
      id: 'canny_image_processor',
      type: 'canny_image_processor',
      low_threshold: 100,
      high_threshold: 200,
    }),
    buildNode: (imageDTO: ImageDTO, config: CannyProcessorConfig): S['CannyImageProcessorInvocation'] => ({
      ...config,
      type: 'canny_image_processor',
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  color_map_image_processor: {
    type: 'color_map_image_processor',
    labelTKey: 'controlnet.colorMap',
    descriptionTKey: 'controlnet.colorMapDescription',
    buildDefaults: (): ColorMapProcessorConfig => ({
      id: 'color_map_image_processor',
      type: 'color_map_image_processor',
      color_map_tile_size: 64,
    }),
    buildNode: (imageDTO: ImageDTO, config: ColorMapProcessorConfig): S['ColorMapImageProcessorInvocation'] => ({
      ...config,
      type: 'color_map_image_processor',
      image: { image_name: imageDTO.image_name },
    }),
  },
  content_shuffle_image_processor: {
    type: 'content_shuffle_image_processor',
    labelTKey: 'controlnet.contentShuffle',
    descriptionTKey: 'controlnet.contentShuffleDescription',
    buildDefaults: (baseModel: BaseModelType): ContentShuffleProcessorConfig => ({
      id: 'content_shuffle_image_processor',
      type: 'content_shuffle_image_processor',
      h: baseModel === 'sdxl' ? 1024 : 512,
      w: baseModel === 'sdxl' ? 1024 : 512,
      f: baseModel === 'sdxl' ? 512 : 256,
    }),
    buildNode: (
      imageDTO: ImageDTO,
      config: ContentShuffleProcessorConfig
    ): S['ContentShuffleImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  depth_anything_image_processor: {
    type: 'depth_anything_image_processor',
    labelTKey: 'controlnet.depthAnything',
    descriptionTKey: 'controlnet.depthAnythingDescription',
    buildDefaults: (): DepthAnythingProcessorConfig => ({
      id: 'depth_anything_image_processor',
      type: 'depth_anything_image_processor',
      model_size: 'small_v2',
    }),
    buildNode: (
      imageDTO: ImageDTO,
      config: DepthAnythingProcessorConfig
    ): S['DepthAnythingImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      resolution: minDim(imageDTO),
    }),
  },
  hed_image_processor: {
    type: 'hed_image_processor',
    labelTKey: 'controlnet.hed',
    descriptionTKey: 'controlnet.hedDescription',
    buildDefaults: (): HedProcessorConfig => ({
      id: 'hed_image_processor',
      type: 'hed_image_processor',
      scribble: false,
    }),
    buildNode: (imageDTO: ImageDTO, config: HedProcessorConfig): S['HedImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  lineart_anime_image_processor: {
    type: 'lineart_anime_image_processor',
    labelTKey: 'controlnet.lineartAnime',
    descriptionTKey: 'controlnet.lineartAnimeDescription',
    buildDefaults: (): LineartAnimeProcessorConfig => ({
      id: 'lineart_anime_image_processor',
      type: 'lineart_anime_image_processor',
    }),
    buildNode: (
      imageDTO: ImageDTO,
      config: LineartAnimeProcessorConfig
    ): S['LineartAnimeImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  lineart_image_processor: {
    type: 'lineart_image_processor',
    labelTKey: 'controlnet.lineart',
    descriptionTKey: 'controlnet.lineartDescription',
    buildDefaults: (): LineartProcessorConfig => ({
      id: 'lineart_image_processor',
      type: 'lineart_image_processor',
      coarse: false,
    }),
    buildNode: (imageDTO: ImageDTO, config: LineartProcessorConfig): S['LineartImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  mediapipe_face_processor: {
    type: 'mediapipe_face_processor',
    labelTKey: 'controlnet.mediapipeFace',
    descriptionTKey: 'controlnet.mediapipeFaceDescription',
    buildDefaults: (): MediapipeFaceProcessorConfig => ({
      id: 'mediapipe_face_processor',
      type: 'mediapipe_face_processor',
      max_faces: 1,
      min_confidence: 0.5,
    }),
    buildNode: (imageDTO: ImageDTO, config: MediapipeFaceProcessorConfig): S['MediapipeFaceProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  midas_depth_image_processor: {
    type: 'midas_depth_image_processor',
    labelTKey: 'controlnet.depthMidas',
    descriptionTKey: 'controlnet.depthMidasDescription',
    buildDefaults: (): MidasDepthProcessorConfig => ({
      id: 'midas_depth_image_processor',
      type: 'midas_depth_image_processor',
      a_mult: 2,
      bg_th: 0.1,
    }),
    buildNode: (imageDTO: ImageDTO, config: MidasDepthProcessorConfig): S['MidasDepthImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  mlsd_image_processor: {
    type: 'mlsd_image_processor',
    labelTKey: 'controlnet.mlsd',
    descriptionTKey: 'controlnet.mlsdDescription',
    buildDefaults: (): MlsdProcessorConfig => ({
      id: 'mlsd_image_processor',
      type: 'mlsd_image_processor',
      thr_d: 0.1,
      thr_v: 0.1,
    }),
    buildNode: (imageDTO: ImageDTO, config: MlsdProcessorConfig): S['MlsdImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  normalbae_image_processor: {
    type: 'normalbae_image_processor',
    labelTKey: 'controlnet.normalBae',
    descriptionTKey: 'controlnet.normalBaeDescription',
    buildDefaults: (): NormalbaeProcessorConfig => ({
      id: 'normalbae_image_processor',
      type: 'normalbae_image_processor',
    }),
    buildNode: (imageDTO: ImageDTO, config: NormalbaeProcessorConfig): S['NormalbaeImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  dw_openpose_image_processor: {
    type: 'dw_openpose_image_processor',
    labelTKey: 'controlnet.dwOpenpose',
    descriptionTKey: 'controlnet.dwOpenposeDescription',
    buildDefaults: (): DWOpenposeProcessorConfig => ({
      id: 'dw_openpose_image_processor',
      type: 'dw_openpose_image_processor',
      draw_body: true,
      draw_face: false,
      draw_hands: false,
    }),
    buildNode: (imageDTO: ImageDTO, config: DWOpenposeProcessorConfig): S['DWOpenposeImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      image_resolution: minDim(imageDTO),
    }),
  },
  pidi_image_processor: {
    type: 'pidi_image_processor',
    labelTKey: 'controlnet.pidi',
    descriptionTKey: 'controlnet.pidiDescription',
    buildDefaults: (): PidiProcessorConfig => ({
      id: 'pidi_image_processor',
      type: 'pidi_image_processor',
      scribble: false,
      safe: false,
    }),
    buildNode: (imageDTO: ImageDTO, config: PidiProcessorConfig): S['PidiImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
      detect_resolution: minDim(imageDTO),
      image_resolution: minDim(imageDTO),
    }),
  },
  zoe_depth_image_processor: {
    type: 'zoe_depth_image_processor',
    labelTKey: 'controlnet.depthZoe',
    descriptionTKey: 'controlnet.depthZoeDescription',
    buildDefaults: (): ZoeDepthProcessorConfig => ({
      id: 'zoe_depth_image_processor',
      type: 'zoe_depth_image_processor',
    }),
    buildNode: (imageDTO: ImageDTO, config: ZoeDepthProcessorConfig): S['ZoeDepthImageProcessorInvocation'] => ({
      ...config,
      image: { image_name: imageDTO.image_name },
    }),
  },
} as const;

const zTool = z.enum(['brush', 'eraser', 'move', 'rect', 'view', 'bbox', 'colorPicker']);
export type Tool = z.infer<typeof zTool>;

const zPoints = z.array(z.number()).refine((points) => points.length % 2 === 0, {
  message: 'Must have an even number of points',
});

const zRgbColor = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
});
export type RgbColor = z.infer<typeof zRgbColor>;
const zRgbaColor = zRgbColor.extend({
  a: z.number().min(0).max(1),
});
export type RgbaColor = z.infer<typeof zRgbaColor>;
export const RGBA_BLACK: RgbaColor = { r: 0, g: 0, b: 0, a: 1 };

const zOpacity = z.number().gte(0).lte(1);

const zDimensions = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
export type Dimensions = z.infer<typeof zDimensions>;

const zCoordinate = z.object({
  x: z.number(),
  y: z.number(),
});
export type Coordinate = z.infer<typeof zCoordinate>;

const zRect = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
export type Rect = z.infer<typeof zRect>;

const zCanvasBrushLineState = z.object({
  id: zId,
  type: z.literal('brush_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
  color: zRgbaColor,
  clip: zRect.nullable(),
});
export type CanvasBrushLineState = z.infer<typeof zCanvasBrushLineState>;

const zCanvasEraserLineState = z.object({
  id: zId,
  type: z.literal('eraser_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
  clip: zRect.nullable(),
});
export type CanvasEraserLineState = z.infer<typeof zCanvasEraserLineState>;

const zCanvasRectState = z.object({
  id: zId,
  type: z.literal('rect'),
  rect: zRect,
  color: zRgbaColor,
});
export type CanvasRectState = z.infer<typeof zCanvasRectState>;

const zCanvasImageState = z.object({
  id: zId,
  type: z.literal('image'),
  image: zImageWithDims,
});
export type CanvasImageState = z.infer<typeof zCanvasImageState>;

const zCanvasObjectState = z.discriminatedUnion('type', [
  zCanvasImageState,
  zCanvasBrushLineState,
  zCanvasEraserLineState,
  zCanvasRectState,
]);

const zIPAdapterConfig = z.object({
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  method: zIPMethodV2,
  clipVisionModel: zCLIPVisionModelV2,
});
export type IPAdapterConfig = z.infer<typeof zIPAdapterConfig>;

const zCanvasEntityBase = z.object({
  id: zId,
  name: zName,
  isEnabled: z.boolean(),
  isLocked: z.boolean(),
});

const zCanvasIPAdapterState = zCanvasEntityBase.extend({
  type: z.literal('ip_adapter'),
  ipAdapter: zIPAdapterConfig,
});
export type CanvasIPAdapterState = z.infer<typeof zCanvasIPAdapterState>;

const zFillStyle = z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']);
export type FillStyle = z.infer<typeof zFillStyle>;
export const isFillStyle = (v: unknown): v is FillStyle => zFillStyle.safeParse(v).success;
const zFill = z.object({ style: zFillStyle, color: zRgbColor });
export type Fill = z.infer<typeof zFill>;

const zRegionalGuidanceIPAdapterConfig = z.object({
  id: zId,
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  method: zIPMethodV2,
  clipVisionModel: zCLIPVisionModelV2,
});
export type RegionalGuidanceIPAdapterConfig = z.infer<typeof zRegionalGuidanceIPAdapterConfig>;

const zCanvasRegionalGuidanceState = zCanvasEntityBase.extend({
  type: z.literal('regional_guidance'),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  fill: zFill,
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zRegionalGuidanceIPAdapterConfig),
  autoNegative: z.boolean(),
});
export type CanvasRegionalGuidanceState = z.infer<typeof zCanvasRegionalGuidanceState>;

const zCanvasInpaintMaskState = zCanvasEntityBase.extend({
  type: z.literal('inpaint_mask'),
  position: zCoordinate,
  fill: zFill,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
});
export type CanvasInpaintMaskState = z.infer<typeof zCanvasInpaintMaskState>;

const zControlNetConfig = z.object({
  type: z.literal('controlnet'),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  controlMode: zControlModeV2,
});
export type ControlNetConfig = z.infer<typeof zControlNetConfig>;

const zT2IAdapterConfig = z.object({
  type: z.literal('t2i_adapter'),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
});
export type T2IAdapterConfig = z.infer<typeof zT2IAdapterConfig>;

export const zCanvasRasterLayerState = zCanvasEntityBase.extend({
  type: z.literal('raster_layer'),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
});
export type CanvasRasterLayerState = z.infer<typeof zCanvasRasterLayerState>;

const zCanvasControlLayerState = zCanvasRasterLayerState.extend({
  type: z.literal('control_layer'),
  withTransparencyEffect: z.boolean(),
  controlAdapter: z.discriminatedUnion('type', [zControlNetConfig, zT2IAdapterConfig]),
});
export type CanvasControlLayerState = z.infer<typeof zCanvasControlLayerState>;

export const initialControlNet: ControlNetConfig = {
  type: 'controlnet',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  controlMode: 'balanced',
};

export const initialT2IAdapter: T2IAdapterConfig = {
  type: 't2i_adapter',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
};

export const initialIPAdapter: IPAdapterConfig = {
  image: null,
  model: null,
  beginEndStepPct: [0, 1],
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
};

export const imageDTOToImageWithDims = ({ image_name, width, height }: ImageDTO): ImageWithDims => ({
  image_name,
  width,
  height,
});

export const imageDTOToImageObject = (imageDTO: ImageDTO, overrides?: Partial<CanvasImageState>): CanvasImageState => {
  const { width, height, image_name } = imageDTO;
  return {
    id: getPrefixedId('image'),
    type: 'image',
    image: {
      image_name,
      width,
      height,
    },
    ...overrides,
  };
};

const zBoundingBoxScaleMethod = z.enum(['none', 'auto', 'manual']);
export type BoundingBoxScaleMethod = z.infer<typeof zBoundingBoxScaleMethod>;
export const isBoundingBoxScaleMethod = (v: unknown): v is BoundingBoxScaleMethod =>
  zBoundingBoxScaleMethod.safeParse(v).success;

export type CanvasEntityState =
  | CanvasRasterLayerState
  | CanvasControlLayerState
  | CanvasRegionalGuidanceState
  | CanvasInpaintMaskState
  | CanvasIPAdapterState;

export type CanvasEntityType = CanvasEntityState['type'];
export type CanvasEntityIdentifier<T extends CanvasEntityType = CanvasEntityType> = { id: string; type: T };

export type LoRA = {
  id: string;
  isEnabled: boolean;
  model: ParameterLoRAModel;
  weight: number;
};

export type StagingAreaImage = {
  imageDTO: ImageDTO;
  offsetX: number;
  offsetY: number;
};

export type CanvasState = {
  _version: 3;
  selectedEntityIdentifier: CanvasEntityIdentifier | null;
  quickSwitchEntityIdentifier: CanvasEntityIdentifier | null;
  inpaintMasks: {
    isHidden: boolean;
    entities: CanvasInpaintMaskState[];
  };
  rasterLayers: {
    isHidden: boolean;
    entities: CanvasRasterLayerState[];
  };
  controlLayers: {
    isHidden: boolean;
    entities: CanvasControlLayerState[];
  };
  regions: {
    isHidden: boolean;
    entities: CanvasRegionalGuidanceState[];
  };
  ipAdapters: {
    entities: CanvasIPAdapterState[];
  };
  bbox: {
    rect: {
      x: number;
      y: number;
      width: ParameterWidth;
      height: ParameterHeight;
    };
    aspectRatio: AspectRatioState;
    scaledSize: {
      width: ParameterWidth;
      height: ParameterHeight;
    };
    scaleMethod: BoundingBoxScaleMethod;
    optimalDimension: number;
  };
};

export type StageAttrs = {
  x: Coordinate['x'];
  y: Coordinate['y'];
  width: Dimensions['width'];
  height: Dimensions['height'];
  scale: number;
};

export type EntityIdentifierPayload<
  T extends SerializableObject | void = void,
  U extends CanvasEntityType = CanvasEntityType,
> = T extends void
  ? {
      entityIdentifier: CanvasEntityIdentifier<U>;
    }
  : {
      entityIdentifier: CanvasEntityIdentifier<U>;
    } & T;

export type EntityMovedPayload = EntityIdentifierPayload<{ position: Coordinate }>;
export type EntityBrushLineAddedPayload = EntityIdentifierPayload<{ brushLine: CanvasBrushLineState }>;
export type EntityEraserLineAddedPayload = EntityIdentifierPayload<{ eraserLine: CanvasEraserLineState }>;
export type EntityRectAddedPayload = EntityIdentifierPayload<{ rect: CanvasRectState }>;
export type EntityRasterizedPayload = EntityIdentifierPayload<{
  imageObject: CanvasImageState;
  rect: Rect;
  replaceObjects: boolean;
}>;

/**
 * A helper type to remove `[index: string]: any;` from a type.
 * This is useful for some Konva types that include `[index: string]: any;` in addition to statically named
 * properties, effectively widening the type signature to `Record<string, any>`. For example, `LineConfig`,
 * `RectConfig`, `ImageConfig`, etc all include `[index: string]: any;` in their type signature.
 * TODO(psyche): Fix this upstream.
 */
// export type RemoveIndexString<T> = {
//   [K in keyof T as string extends K ? never : K]: T[K];
// };

export type GenerationMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';

export function isDrawableEntityType(entityType: CanvasEntityState['type']) {
  return (
    entityType === 'raster_layer' ||
    entityType === 'control_layer' ||
    entityType === 'regional_guidance' ||
    entityType === 'inpaint_mask'
  );
}

export function isDrawableEntity(
  entity: CanvasEntityState
): entity is CanvasRasterLayerState | CanvasControlLayerState | CanvasRegionalGuidanceState | CanvasInpaintMaskState {
  return isDrawableEntityType(entity.type);
}

export const getEntityIdentifier = <T extends CanvasEntityType>(
  entity: Extract<CanvasEntityState, { type: T }>
): CanvasEntityIdentifier<T> => {
  return { id: entity.id, type: entity.type };
};

export const isMaskEntityIdentifier = (
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'inpaint_mask' | 'regional_guidance'> => {
  return entityIdentifier.type === 'inpaint_mask' || entityIdentifier.type === 'regional_guidance';
};
