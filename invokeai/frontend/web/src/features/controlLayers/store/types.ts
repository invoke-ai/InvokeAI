import type { CanvasControlAdapter } from 'features/controlLayers/konva/CanvasControlAdapter';
import { CanvasInpaintMask } from 'features/controlLayers/konva/CanvasInpaintMask';
import { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import { CanvasRegion } from 'features/controlLayers/konva/CanvasRegion';
import { getObjectId } from 'features/controlLayers/konva/util';
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
import type { IRect } from 'konva/lib/types';
import type {
  AnyInvocation,
  BaseModelType,
  ControlNetModelConfig,
  ImageDTO,
  T2IAdapterModelConfig,
} from 'services/api/types';
import { z } from 'zod';

export const zId = z.string().min(1);

export const zImageWithDims = z.object({
  name: z.string(),
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

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
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

export const zProcessorConfig = z.discriminatedUnion('type', [
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
  buildNode(image: ImageWithDims, config: Extract<ProcessorConfig, { type: T }>): Extract<AnyInvocation, { type: T }>;
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

const zTool = z.enum(['brush', 'eraser', 'move', 'rect', 'view', 'bbox']);
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

const zBrushLine = z.object({
  id: zId,
  type: z.literal('brush_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
  color: zRgbaColor,
  clip: zRect.nullable(),
});
export type BrushLine = z.infer<typeof zBrushLine>;

const zEraserline = z.object({
  id: zId,
  type: z.literal('eraser_line'),
  strokeWidth: z.number().min(1),
  points: zPoints,
  clip: zRect.nullable(),
});
export type EraserLine = z.infer<typeof zEraserline>;

const zRectShape = z.object({
  id: zId,
  type: z.literal('rect_shape'),
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
  color: zRgbaColor,
});
export type RectShape = z.infer<typeof zRectShape>;

const zFilter = z.enum(['LightnessToAlphaFilter']);
export type Filter = z.infer<typeof zFilter>;

const zImageObject = z.object({
  id: zId,
  type: z.literal('image'),
  image: zImageWithDims,
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
  filters: z.array(zFilter),
});
export type ImageObject = z.infer<typeof zImageObject>;

const zRenderableObject = z.discriminatedUnion('type', [zImageObject, zBrushLine, zEraserline, zRectShape]);
export type RenderableObject = z.infer<typeof zRenderableObject>;

export const zLayerEntity = z.object({
  id: zId,
  type: z.literal('layer'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zRenderableObject),
});
export type LayerEntity = z.infer<typeof zLayerEntity>;

export const zIPAdapterEntity = z.object({
  id: zId,
  type: z.literal('ip_adapter'),
  isEnabled: z.boolean(),
  weight: z.number().gte(-1).lte(2),
  method: zIPMethodV2,
  imageObject: zImageObject.nullable(),
  model: zModelIdentifierField.nullable(),
  clipVisionModel: zCLIPVisionModelV2,
  beginEndStepPct: zBeginEndStepPct,
});
export type IPAdapterEntity = z.infer<typeof zIPAdapterEntity>;
export type IPAdapterConfig = Pick<
  IPAdapterEntity,
  'weight' | 'imageObject' | 'beginEndStepPct' | 'model' | 'clipVisionModel' | 'method'
>;

const zMaskObject = z
  .discriminatedUnion('type', [zOLD_VectorMaskLine, zOLD_VectorMaskRect, zBrushLine, zEraserline, zRectShape])
  .transform((val) => {
    // Migrate old vector mask objects to new format
    if (val.type === 'vector_mask_line') {
      const { tool, ...rest } = val;
      if (tool === 'brush') {
        const asBrushline: BrushLine = {
          ...rest,
          type: 'brush_line',
          color: { r: 255, g: 255, b: 255, a: 1 },
          clip: null,
        };
        return asBrushline;
      } else if (tool === 'eraser') {
        const asEraserLine: EraserLine = {
          ...rest,
          type: 'eraser_line',
          clip: null,
        };
        return asEraserLine;
      }
    } else if (val.type === 'vector_mask_rect') {
      const asRectShape: RectShape = {
        ...val,
        type: 'rect_shape',
        color: { r: 255, g: 255, b: 255, a: 1 },
      };
      return asRectShape;
    } else {
      return val;
    }
  })
  .pipe(z.discriminatedUnion('type', [zBrushLine, zEraserline, zRectShape]));

export const zRegionEntity = z.object({
  id: zId,
  type: z.literal('regional_guidance'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  objects: z.array(zMaskObject),
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  ipAdapters: z.array(zIPAdapterEntity),
  fill: zRgbColor,
  autoNegative: zAutoNegative,
  imageCache: zImageWithDims.nullable(),
});
export type RegionEntity = z.infer<typeof zRegionEntity>;

const zColorFill = z.object({
  type: z.literal('color_fill'),
  color: zRgbaColor,
});
const zImageFill = z.object({
  type: z.literal('image_fill'),
  src: z.string(),
});
const zFill = z.discriminatedUnion('type', [zColorFill, zImageFill]);
const zInpaintMaskEntity = z.object({
  id: z.literal('inpaint_mask'),
  type: z.literal('inpaint_mask'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  objects: z.array(zMaskObject),
  fill: zRgbColor,
  imageCache: zImageWithDims.nullable(),
});
export type InpaintMaskEntity = z.infer<typeof zInpaintMaskEntity>;

const zInitialImageEntity = z.object({
  id: z.literal('initial_image'),
  type: z.literal('initial_image'),
  isEnabled: z.boolean(),
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  imageObject: zImageObject.nullable(),
});
export type InitialImageEntity = z.infer<typeof zInitialImageEntity>;

const zControlAdapterEntityBase = z.object({
  id: zId,
  type: z.literal('control_adapter'),
  isEnabled: z.boolean(),
  position: zCoordinate,
  bbox: zRect.nullable(),
  bboxNeedsUpdate: z.boolean(),
  opacity: zOpacity,
  filters: z.array(zFilter),
  weight: z.number().gte(-1).lte(2),
  imageObject: zImageObject.nullable(),
  processedImageObject: zImageObject.nullable(),
  processorConfig: zProcessorConfig.nullable(),
  processorPendingBatchId: z.string().nullable().default(null),
  beginEndStepPct: zBeginEndStepPct,
  model: zModelIdentifierField.nullable(),
});
const zControlNetEntity = zControlAdapterEntityBase.extend({
  adapterType: z.literal('controlnet'),
  controlMode: zControlModeV2,
});
export type ControlNetData = z.infer<typeof zControlNetEntity>;
const zT2IAdapterEntity = zControlAdapterEntityBase.extend({
  adapterType: z.literal('t2i_adapter'),
});
export type T2IAdapterData = z.infer<typeof zT2IAdapterEntity>;

export const zControlAdapterEntity = z.discriminatedUnion('adapterType', [zControlNetEntity, zT2IAdapterEntity]);
export type ControlAdapterEntity = z.infer<typeof zControlAdapterEntity>;
export type ControlNetConfig = Pick<
  ControlNetData,
  | 'adapterType'
  | 'weight'
  | 'imageObject'
  | 'processedImageObject'
  | 'processorConfig'
  | 'beginEndStepPct'
  | 'model'
  | 'controlMode'
>;
export type T2IAdapterConfig = Pick<
  T2IAdapterData,
  'adapterType' | 'weight' | 'imageObject' | 'processedImageObject' | 'processorConfig' | 'beginEndStepPct' | 'model'
>;

export const initialControlNetV2: ControlNetConfig = {
  adapterType: 'controlnet',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  controlMode: 'balanced',
  imageObject: null,
  processedImageObject: null,
  processorConfig: CA_PROCESSOR_DATA.canny_image_processor.buildDefaults(),
};

export const initialT2IAdapterV2: T2IAdapterConfig = {
  adapterType: 't2i_adapter',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
  imageObject: null,
  processedImageObject: null,
  processorConfig: CA_PROCESSOR_DATA.canny_image_processor.buildDefaults(),
};

export const initialIPAdapterV2: IPAdapterConfig = {
  imageObject: null,
  model: null,
  beginEndStepPct: [0, 1],
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
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

export const imageDTOToImageObject = (imageDTO: ImageDTO, overrides?: Partial<ImageObject>): ImageObject => {
  const { width, height, image_name } = imageDTO;
  return {
    id: getObjectId('image'),
    type: 'image',
    x: 0,
    y: 0,
    width,
    height,
    filters: [],
    image: {
      name: image_name,
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

export type CanvasEntity =
  | LayerEntity
  | ControlAdapterEntity
  | RegionEntity
  | InpaintMaskEntity
  | IPAdapterEntity
  | InitialImageEntity;
export type CanvasEntityIdentifier = Pick<CanvasEntity, 'id' | 'type'>;

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

export type CanvasV2State = {
  _version: 3;
  selectedEntityIdentifier: CanvasEntityIdentifier | null;
  inpaintMask: InpaintMaskEntity;
  layers: {
    imageCache: ImageWithDims | null;
    entities: LayerEntity[];
  };
  controlAdapters: { entities: ControlAdapterEntity[] };
  ipAdapters: { entities: IPAdapterEntity[] };
  regions: { entities: RegionEntity[] };
  loras: LoRA[];
  initialImage: InitialImageEntity;
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
    maskOpacity: number;
    showHUD: boolean;
    autoSave: boolean;
    preserveMaskedArea: boolean;
    cropToBboxOnSave: boolean;
    clipToBbox: boolean;
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
    isActive: boolean;
    isStaging: boolean;
    stagedImages: StagingAreaImage[];
    selectedStagedImageIndex: number;
  };
};

export type StageAttrs = { position: Coordinate; dimensions: Dimensions; scale: number };
export type PositionChangedArg = { id: string; position: Coordinate };
export type ScaleChangedArg = { id: string; scale: Coordinate; position: Coordinate };
export type BboxChangedArg = { id: string; bbox: Rect | null };
export type EraserLineAddedArg = {
  id: string;
  points: [number, number, number, number];
  width: number;
  clip: Rect | null;
};
export type BrushLineAddedArg = EraserLineAddedArg & { color: RgbaColor };
export type PointAddedToLineArg = { id: string; point: [number, number] };
export type RectShapeAddedArg = { id: string; rect: IRect; color: RgbaColor };
export type ImageObjectAddedArg = { id: string; imageDTO: ImageDTO; position?: Coordinate };

//#region Type guards
export const isLine = (obj: RenderableObject): obj is BrushLine | EraserLine => {
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

export function isDrawableEntity(entity: CanvasEntity): entity is LayerEntity | RegionEntity | InpaintMaskEntity {
  return entity.type === 'layer' || entity.type === 'regional_guidance' || entity.type === 'inpaint_mask';
}

export function isDrawableEntityAdapter(
  adapter: CanvasLayer | CanvasRegion | CanvasControlAdapter | CanvasInpaintMask
): adapter is CanvasLayer | CanvasRegion | CanvasInpaintMask {
  return adapter instanceof CanvasLayer || adapter instanceof CanvasRegion || adapter instanceof CanvasInpaintMask;
}

export function isDrawableEntityType(
  entityType: CanvasEntity['type']
): entityType is 'layer' | 'regional_guidance' | 'inpaint_mask' {
  return entityType === 'layer' || entityType === 'regional_guidance' || entityType === 'inpaint_mask';
}
