import type { CanvasControlAdapter } from 'features/controlLayers/konva/CanvasControlAdapter';
import { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { AspectRatioState } from 'features/parameters/components/DocumentSize/types';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterHeight,
  ParameterLoRAModel,
  ParameterMaskBlurMethod,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterPrecision,
  ParameterScheduler,
  ParameterSDXLRefinerModel,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterVAEModel,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import {
  zAutoNegative,
  zParameterNegativePrompt,
  zParameterPositivePrompt,
} from 'features/parameters/types/parameterSchemas';
import type {
  AnyInvocation,
  BaseModelType,
  ControlNetModelConfig,
  ImageDTO,
  S,
  T2IAdapterModelConfig,
} from 'services/api/types';
import { z } from 'zod';

export const zId = z.string().min(1);

export const zImageWithDims = z.object({
  image_name: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
export type ImageWithDims = z.infer<typeof zImageWithDims>;

export const zBeginEndStepPct = z
  .tuple([z.number().gte(0).lte(1), z.number().gte(0).lte(1)])
  .refine(([begin, end]) => begin < end, {
    message: 'Begin must be less than end',
  });

export const zControlModeV2 = z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']);
export type ControlModeV2 = z.infer<typeof zControlModeV2>;
export const isControlModeV2 = (v: unknown): v is ControlModeV2 => zControlModeV2.safeParse(v).success;

export const zCLIPVisionModelV2 = z.enum(['ViT-H', 'ViT-G']);
export type CLIPVisionModelV2 = z.infer<typeof zCLIPVisionModelV2>;
export const isCLIPVisionModelV2 = (v: unknown): v is CLIPVisionModelV2 => zCLIPVisionModelV2.safeParse(v).success;

export const zIPMethodV2 = z.enum(['full', 'style', 'composition']);
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

export const zFilterConfig = z.discriminatedUnion('type', [
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

const zTool = z.enum(['brush', 'eraser', 'move', 'rect', 'view', 'bbox', 'eyeDropper']);
export type Tool = z.infer<typeof zTool>;
export function isDrawingTool(tool: Tool): tool is 'brush' | 'eraser' | 'rect' {
  return tool === 'brush' || tool === 'eraser' || tool === 'rect';
}

const zDrawingTool = zTool.extract(['brush', 'eraser']);

const zPoints = z.array(z.number()).refine((points) => points.length % 2 === 0, {
  message: 'Must have an even number of points',
});
const zOLD_VectorMaskLine = z.object({
  id: zId,
  type: z.literal('vector_mask_line'),
  tool: zDrawingTool,
  strokeWidth: z.number().min(1),
  points: zPoints,
});

const zOLD_VectorMaskRect = z.object({
  id: zId,
  type: z.literal('vector_mask_rect'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
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
export const RGBA_RED: RgbaColor = { r: 255, g: 0, b: 0, a: 1 };
export const RGBA_WHITE: RgbaColor = { r: 255, g: 255, b: 255, a: 1 };

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

const zLayerEffect = z.enum(['LightnessToAlphaFilter']);
export type LayerEffect = z.infer<typeof zLayerEffect>;

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
export type CanvasObjectState = z.infer<typeof zCanvasObjectState>;
export function isCanvasBrushLineState(obj: CanvasObjectState): obj is CanvasBrushLineState {
  return obj.type === 'brush_line';
}

const zIPAdapterConfig = z.object({
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  method: zIPMethodV2,
  clipVisionModel: zCLIPVisionModelV2,
});
export type IPAdapterConfig = z.infer<typeof zIPAdapterConfig>;

export const zCanvasIPAdapterState = z.object({
  id: zId,
  name: z.string().nullable(),
  type: z.literal('ip_adapter'),
  isEnabled: z.boolean(),
  ipAdapter: zIPAdapterConfig,
});
export type CanvasIPAdapterState = z.infer<typeof zCanvasIPAdapterState>;

const zMaskObject = z
  .discriminatedUnion('type', [
    zOLD_VectorMaskLine,
    zOLD_VectorMaskRect,
    zCanvasBrushLineState,
    zCanvasEraserLineState,
    zCanvasRectState,
  ])
  .transform((val) => {
    // Migrate old vector mask objects to new format
    if (val.type === 'vector_mask_line') {
      const { tool, ...rest } = val;
      if (tool === 'brush') {
        const asBrushline: CanvasBrushLineState = {
          ...rest,
          type: 'brush_line',
          color: { r: 255, g: 255, b: 255, a: 1 },
          clip: null,
        };
        return asBrushline;
      } else if (tool === 'eraser') {
        const asEraserLine: CanvasEraserLineState = {
          ...rest,
          type: 'eraser_line',
          clip: null,
        };
        return asEraserLine;
      }
    } else if (val.type === 'vector_mask_rect') {
      const asRectShape: CanvasRectState = {
        ...val,
        type: 'rect',
        color: { r: 255, g: 255, b: 255, a: 1 },
      };
      return asRectShape;
    } else {
      return val;
    }
  })
  .pipe(z.discriminatedUnion('type', [zCanvasBrushLineState, zCanvasEraserLineState, zCanvasRectState]));

const zFillStyle = z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']);
export type FillStyle = z.infer<typeof zFillStyle>;
export const isFillStyle = (v: unknown): v is FillStyle => zFillStyle.safeParse(v).success;
const zFill = z.object({ style: zFillStyle, color: zRgbColor });
export type Fill = z.infer<typeof zFill>;

const zImageCache = z.object({
  imageName: z.string(),
  rect: zRect,
});
export type ImageCache = z.infer<typeof zImageCache>;

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

export const zCanvasRegionalGuidanceState = z.object({
  id: zId,
  name: z.string().nullable(),
  type: z.literal('regional_guidance'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  fill: zFill,
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zRegionalGuidanceIPAdapterConfig),
  autoNegative: zAutoNegative,
  rasterizationCache: z.array(zImageCache),
});
export type CanvasRegionalGuidanceState = z.infer<typeof zCanvasRegionalGuidanceState>;

const zCanvasInpaintMaskState = z.object({
  id: z.literal('inpaint_mask'),
  type: z.literal('inpaint_mask'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  fill: zFill,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  rasterizationCache: z.array(zImageCache),
});
export type CanvasInpaintMaskState = z.infer<typeof zCanvasInpaintMaskState>;

const zCanvasControlAdapterStateBase = z.object({
  id: zId,
  type: z.literal('control_adapter'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  opacity: zOpacity,
  filters: z.array(zLayerEffect),
  weight: z.number().gte(-1).lte(2),
  imageObject: zCanvasImageState.nullable(),
  processedImageObject: zCanvasImageState.nullable(),
  processorConfig: zFilterConfig.nullable(),
  processorPendingBatchId: z.string().nullable().default(null),
  beginEndStepPct: zBeginEndStepPct,
  model: zModelIdentifierField.nullable(),
});
const zCanvasControlNetState = zCanvasControlAdapterStateBase.extend({
  adapterType: z.literal('controlnet'),
  controlMode: zControlModeV2,
});
export type CanvasControlNetState = z.infer<typeof zCanvasControlNetState>;
const zCanvasT2IAdapteState = zCanvasControlAdapterStateBase.extend({
  adapterType: z.literal('t2i_adapter'),
});
export type CanvasT2IAdapterState = z.infer<typeof zCanvasT2IAdapteState>;

export const zCanvasControlAdapterState = z.discriminatedUnion('adapterType', [
  zCanvasControlNetState,
  zCanvasT2IAdapteState,
]);
export type CanvasControlAdapterState = z.infer<typeof zCanvasControlAdapterState>;

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

export const zCanvasRasterLayerState = z.object({
  id: zId,
  name: z.string().nullable(),
  type: z.literal('raster_layer'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  rasterizationCache: z.array(zImageCache),
});
export type CanvasRasterLayerState = z.infer<typeof zCanvasRasterLayerState>;

export const zCanvasControlLayerState = zCanvasRasterLayerState.extend({
  type: z.literal('control_layer'),
  withTransparencyEffect: z.boolean(),
  controlAdapter: z.discriminatedUnion('type', [zControlNetConfig, zT2IAdapterConfig]),
});
export type CanvasControlLayerState = z.infer<typeof zCanvasControlLayerState>;

export const initialControlNetV2: ControlNetConfig = {
  type: 'controlnet',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  controlMode: 'balanced',
};

export const initialT2IAdapterV2: T2IAdapterConfig = {
  type: 't2i_adapter',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
};

export const initialIPAdapterV2: IPAdapterConfig = {
  image: null,
  model: null,
  beginEndStepPct: [0, 1],
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
};

export const buildControlAdapterProcessorV2 = (
  modelConfig: ControlNetModelConfig | T2IAdapterModelConfig
): FilterConfig | null => {
  const defaultPreprocessor = modelConfig.default_settings?.preprocessor;
  if (!isFilterType(defaultPreprocessor)) {
    return null;
  }
  const processorConfig = IMAGE_FILTERS[defaultPreprocessor].buildDefaults(modelConfig.base);
  return processorConfig;
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
export type CanvasEntityIdentifier = Pick<CanvasEntityState, 'id' | 'type'>;

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

const zCanvasBackgroundStyle = z.enum(['checkerboard', 'dynamicGrid', 'solid']);
export type CanvasBackgroundStyle = z.infer<typeof zCanvasBackgroundStyle>;
export const isCanvasBackgroundStyle = (v: unknown): v is CanvasBackgroundStyle =>
  zCanvasBackgroundStyle.safeParse(v).success;

export type CanvasV2State = {
  _version: 3;
  selectedEntityIdentifier: CanvasEntityIdentifier | null;
  inpaintMask: CanvasInpaintMaskState;
  rasterLayers: { entities: CanvasRasterLayerState[]; compositeRasterizationCache: ImageCache[] };
  controlLayers: { entities: CanvasControlLayerState[] };
  ipAdapters: { entities: CanvasIPAdapterState[] };
  regions: { entities: CanvasRegionalGuidanceState[] };
  loras: LoRA[];
  tool: {
    selected: Tool;
    selectedBuffer: Tool | null;
    invertScroll: boolean;
    brush: { width: number };
    eraser: { width: number };
    fill: RgbaColor;
  };
  settings: {
    imageSmoothing: boolean;
    showHUD: boolean;
    autoSave: boolean;
    preserveMaskedArea: boolean;
    cropToBboxOnSave: boolean;
    clipToBbox: boolean;
    canvasBackgroundStyle: CanvasBackgroundStyle;
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
  };
  compositing: {
    maskBlur: number;
    maskBlurMethod: ParameterMaskBlurMethod;
    canvasCoherenceMode: ParameterCanvasCoherenceMode;
    canvasCoherenceMinDenoise: ParameterStrength;
    canvasCoherenceEdgeSize: number;
    infillMethod: string;
    infillTileSize: number;
    infillPatchmatchDownscaleSize: number;
    infillColorValue: RgbaColor;
  };
  params: {
    cfgScale: ParameterCFGScale;
    cfgRescaleMultiplier: ParameterCFGRescaleMultiplier;
    img2imgStrength: ParameterStrength;
    iterations: number;
    scheduler: ParameterScheduler;
    seed: ParameterSeed;
    shouldRandomizeSeed: boolean;
    steps: ParameterSteps;
    model: ParameterModel | null;
    vae: ParameterVAEModel | null;
    vaePrecision: ParameterPrecision;
    seamlessXAxis: boolean;
    seamlessYAxis: boolean;
    clipSkip: number;
    shouldUseCpuNoise: boolean;
    positivePrompt: ParameterPositivePrompt;
    negativePrompt: ParameterNegativePrompt;
    positivePrompt2: ParameterPositiveStylePromptSDXL;
    negativePrompt2: ParameterNegativeStylePromptSDXL;
    shouldConcatPrompts: boolean;
    refinerModel: ParameterSDXLRefinerModel | null;
    refinerSteps: number;
    refinerCFGScale: number;
    refinerScheduler: ParameterScheduler;
    refinerPositiveAestheticScore: number;
    refinerNegativeAestheticScore: number;
    refinerStart: number;
  };
  session: {
    isStaging: boolean;
    stagedImages: StagingAreaImage[];
    selectedStagedImageIndex: number;
  };
  filter: {
    autoProcess: boolean;
    config: FilterConfig;
  };
};

export type StageAttrs = {
  x: Coordinate['x'];
  y: Coordinate['y'];
  width: Dimensions['width'];
  height: Dimensions['height'];
  scale: number;
};
export type PositionChangedArg = { id: string; position: Coordinate };
export type ScaleChangedArg = { id: string; scale: Coordinate; position: Coordinate };
export type BboxChangedArg = { id: string; bbox: Rect | null };

export type EntityIdentifierPayload<T = object> = { entityIdentifier: CanvasEntityIdentifier } & T;
export type EntityMovedPayload = EntityIdentifierPayload<{ position: Coordinate }>;
export type EntityBrushLineAddedPayload = EntityIdentifierPayload<{ brushLine: CanvasBrushLineState }>;
export type EntityEraserLineAddedPayload = EntityIdentifierPayload<{ eraserLine: CanvasEraserLineState }>;
export type EntityRectAddedPayload = EntityIdentifierPayload<{ rect: CanvasRectState }>;
export type EntityRasterizedPayload = EntityIdentifierPayload<{
  imageObject: CanvasImageState;
  rect: Rect;
  replaceObjects: boolean;
}>;
export type ImageObjectAddedArg = { id: string; imageDTO: ImageDTO; position?: Coordinate };

//#region Type guards
export const isLine = (obj: CanvasObjectState): obj is CanvasBrushLineState | CanvasEraserLineState => {
  return obj.type === 'brush_line' || obj.type === 'eraser_line';
};

/**
 * A helper type to remove `[index: string]: any;` from a type.
 * This is useful for some Konva types that include `[index: string]: any;` in addition to statically named
 * properties, effectively widening the type signature to `Record<string, any>`. For example, `LineConfig`,
 * `RectConfig`, `ImageConfig`, etc all include `[index: string]: any;` in their type signature.
 * TODO(psyche): Fix this upstream.
 */
export type RemoveIndexString<T> = {
  [K in keyof T as string extends K ? never : K]: T[K];
};

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

export function isDrawableEntityAdapter(
  adapter: CanvasLayerAdapter | CanvasControlAdapter | CanvasMaskAdapter
): adapter is CanvasLayerAdapter | CanvasMaskAdapter {
  return adapter instanceof CanvasLayerAdapter || adapter instanceof CanvasMaskAdapter;
}

export const getEntityIdentifier = (entity: CanvasEntityState): CanvasEntityIdentifier => {
  return { id: entity.id, type: entity.type };
};
