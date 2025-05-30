import { deepClone } from 'common/util/deepClone';
import { fetchModelConfigByIdentifier } from 'features/metadata/util/modelFetchingHelpers';
import type { ProgressImage } from 'features/nodes/types/common';
import { zMainModelBase, zModelIdentifierField } from 'features/nodes/types/common';
import type { ParameterLoRAModel } from 'features/parameters/types/parameterSchemas';
import {
  zParameterCanvasCoherenceMode,
  zParameterCFGRescaleMultiplier,
  zParameterCFGScale,
  zParameterCLIPEmbedModel,
  zParameterCLIPGEmbedModel,
  zParameterCLIPLEmbedModel,
  zParameterControlLoRAModel,
  zParameterGuidance,
  zParameterImageDimension,
  zParameterMaskBlurMethod,
  zParameterModel,
  zParameterNegativePrompt,
  zParameterNegativeStylePromptSDXL,
  zParameterPositivePrompt,
  zParameterPositiveStylePromptSDXL,
  zParameterPrecision,
  zParameterScheduler,
  zParameterSDXLRefinerModel,
  zParameterSeed,
  zParameterSteps,
  zParameterStrength,
  zParameterT5EncoderModel,
  zParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import type { JsonObject } from 'type-fest';
import { z } from 'zod';

const zId = z.string().min(1);
const zName = z.string().min(1).nullable();

const zServerValidatedModelIdentifierField = zModelIdentifierField.refine(async (modelIdentifier) => {
  try {
    await fetchModelConfigByIdentifier(modelIdentifier);
    return true;
  } catch {
    return false;
  }
});

const zImageWithDims = z
  .object({
    image_name: z.string(),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
  })
  .refine(async (v) => {
    const { image_name } = v;
    const imageDTO = await getImageDTOSafe(image_name, { forceRefetch: true });
    return imageDTO !== null;
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

const zCLIPVisionModelV2 = z.enum(['ViT-H', 'ViT-G', 'ViT-L']);
export type CLIPVisionModelV2 = z.infer<typeof zCLIPVisionModelV2>;
export const isCLIPVisionModelV2 = (v: unknown): v is CLIPVisionModelV2 => zCLIPVisionModelV2.safeParse(v).success;

const zIPMethodV2 = z.enum(['full', 'style', 'composition', 'style_strong', 'style_precise']);
export type IPMethodV2 = z.infer<typeof zIPMethodV2>;
export const isIPMethodV2 = (v: unknown): v is IPMethodV2 => zIPMethodV2.safeParse(v).success;

const zTool = z.enum(['brush', 'eraser', 'move', 'rect', 'view', 'bbox', 'colorPicker']);
export type Tool = z.infer<typeof zTool>;

const zPoints = z.array(z.number()).refine((points) => points.length % 2 === 0, {
  message: 'Must have an even number of coordinate components',
});
const zPointsWithPressure = z.array(z.number()).refine((points) => points.length % 3 === 0, {
  message: 'Must have a number of components divisible by 3',
});

const zRgbColor = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
});
export type RgbColor = z.infer<typeof zRgbColor>;
export const zRgbaColor = zRgbColor.extend({
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
const zCoordinateWithPressure = z.object({
  x: z.number(),
  y: z.number(),
  pressure: z.number(),
});
export type CoordinateWithPressure = z.infer<typeof zCoordinateWithPressure>;

const SAM_POINT_LABELS = {
  background: -1,
  neutral: 0,
  foreground: 1,
} as const;

const zSAMPointLabel = z.nativeEnum(SAM_POINT_LABELS);
export type SAMPointLabel = z.infer<typeof zSAMPointLabel>;

export const zSAMPointLabelString = z.enum(['background', 'neutral', 'foreground']);
export type SAMPointLabelString = z.infer<typeof zSAMPointLabelString>;

/**
 * A mapping of SAM point labels (as numbers) to their string representations.
 */
export const SAM_POINT_LABEL_NUMBER_TO_STRING: Record<SAMPointLabel, SAMPointLabelString> = {
  '-1': 'background',
  0: 'neutral',
  1: 'foreground',
};

/**
 * A mapping of SAM point labels (as strings) to their numeric representations.
 */
export const SAM_POINT_LABEL_STRING_TO_NUMBER: Record<SAMPointLabelString, SAMPointLabel> = {
  background: -1,
  neutral: 0,
  foreground: 1,
};

const zSAMPoint = z.object({
  x: z.number().int().gte(0),
  y: z.number().int().gte(0),
  label: zSAMPointLabel,
});
type SAMPoint = z.infer<typeof zSAMPoint>;
export type SAMPointWithId = SAMPoint & { id: string };

const zRect = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
export type Rect = z.infer<typeof zRect>;

const zRectWithRotation = zRect.extend({
  rotation: z.number(),
});
export type RectWithRotation = z.infer<typeof zRectWithRotation>;

const zCanvasBrushLineState = z.object({
  id: zId,
  type: z.literal('brush_line'),
  strokeWidth: z.number().min(1),
  /**
   * Points without pressure are in the format [x1, y1, x2, y2, ...]
   */
  points: zPoints,
  color: zRgbaColor,
  clip: zRect.nullable(),
});
export type CanvasBrushLineState = z.infer<typeof zCanvasBrushLineState>;

const zCanvasBrushLineWithPressureState = z.object({
  id: zId,
  type: z.literal('brush_line_with_pressure'),
  strokeWidth: z.number().min(1),
  /**
   * Points with pressure are in the format [x1, y1, pressure1, x2, y2, pressure2, ...]
   */
  points: zPointsWithPressure,
  color: zRgbaColor,
  clip: zRect.nullable(),
});
export type CanvasBrushLineWithPressureState = z.infer<typeof zCanvasBrushLineWithPressureState>;

const zCanvasEraserLineState = z.object({
  id: zId,
  type: z.literal('eraser_line'),
  strokeWidth: z.number().min(1),
  /**
   * Points without pressure are in the format [x1, y1, x2, y2, ...]
   */
  points: zPoints,
  clip: zRect.nullable(),
});
export type CanvasEraserLineState = z.infer<typeof zCanvasEraserLineState>;

const zCanvasEraserLineWithPressureState = z.object({
  id: zId,
  type: z.literal('eraser_line_with_pressure'),
  strokeWidth: z.number().min(1),
  /**
   * Points with pressure are in the format [x1, y1, pressure1, x2, y2, pressure2, ...]
   */
  points: zPointsWithPressure,
  clip: zRect.nullable(),
});
export type CanvasEraserLineWithPressureState = z.infer<typeof zCanvasEraserLineWithPressureState>;

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

const zCanvasObjectState = z.union([
  zCanvasImageState,
  zCanvasBrushLineState,
  zCanvasEraserLineState,
  zCanvasRectState,
  zCanvasBrushLineWithPressureState,
  zCanvasEraserLineWithPressureState,
]);
export type CanvasObjectState = z.infer<typeof zCanvasObjectState>;

const zIPAdapterConfig = z.object({
  type: z.literal('ip_adapter'),
  image: zImageWithDims.nullable(),
  model: zServerValidatedModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  method: zIPMethodV2,
  clipVisionModel: zCLIPVisionModelV2,
});
export type IPAdapterConfig = z.infer<typeof zIPAdapterConfig>;

const zFLUXReduxImageInfluence = z.enum(['lowest', 'low', 'medium', 'high', 'highest']);
export const isFLUXReduxImageInfluence = (v: unknown): v is FLUXReduxImageInfluence =>
  zFLUXReduxImageInfluence.safeParse(v).success;
export type FLUXReduxImageInfluence = z.infer<typeof zFLUXReduxImageInfluence>;
const zFLUXReduxConfig = z.object({
  type: z.literal('flux_redux'),
  image: zImageWithDims.nullable(),
  model: zServerValidatedModelIdentifierField.nullable(),
  imageInfluence: zFLUXReduxImageInfluence.default('highest'),
});
export type FLUXReduxConfig = z.infer<typeof zFLUXReduxConfig>;

const zChatGPT4oReferenceImageConfig = z.object({
  type: z.literal('chatgpt_4o_reference_image'),
  image: zImageWithDims.nullable(),
  /**
   * TODO(psyche): Technically there is no model for ChatGPT 4o reference images - it's just a field in the API call.
   * But we use a model drop down to switch between different ref image types, so there needs to be a model here else
   * there will be no way to switch between ref image types.
   */
  model: zServerValidatedModelIdentifierField.nullable(),
});
export type ChatGPT4oReferenceImageConfig = z.infer<typeof zChatGPT4oReferenceImageConfig>;

const zCanvasEntityBase = z.object({
  id: zId,
  name: zName,
  isEnabled: z.boolean(),
  isLocked: z.boolean(),
});

const zCanvasReferenceImageState = zCanvasEntityBase.extend({
  type: z.literal('reference_image'),
  // This should be named `referenceImage` but we need to keep it as `ipAdapter` for backwards compatibility
  ipAdapter: z.discriminatedUnion('type', [zIPAdapterConfig, zFLUXReduxConfig, zChatGPT4oReferenceImageConfig]),
});
export type CanvasReferenceImageState = z.infer<typeof zCanvasReferenceImageState>;

export const isIPAdapterConfig = (config: CanvasReferenceImageState['ipAdapter']): config is IPAdapterConfig =>
  config.type === 'ip_adapter';

export const isFLUXReduxConfig = (config: CanvasReferenceImageState['ipAdapter']): config is FLUXReduxConfig =>
  config.type === 'flux_redux';
export const isChatGPT4oReferenceImageConfig = (
  config: CanvasReferenceImageState['ipAdapter']
): config is ChatGPT4oReferenceImageConfig => config.type === 'chatgpt_4o_reference_image';

const zFillStyle = z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']);
export type FillStyle = z.infer<typeof zFillStyle>;
export const isFillStyle = (v: unknown): v is FillStyle => zFillStyle.safeParse(v).success;
const zFill = z.object({ style: zFillStyle, color: zRgbColor });

const zRegionalGuidanceReferenceImageState = z.object({
  id: zId,
  ipAdapter: z.discriminatedUnion('type', [zIPAdapterConfig, zFLUXReduxConfig]),
});
export type RegionalGuidanceReferenceImageState = z.infer<typeof zRegionalGuidanceReferenceImageState>;

const zCanvasRegionalGuidanceState = zCanvasEntityBase.extend({
  type: z.literal('regional_guidance'),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  fill: zFill,
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  referenceImages: z.array(zRegionalGuidanceReferenceImageState),
  autoNegative: z.boolean(),
});
export type CanvasRegionalGuidanceState = z.infer<typeof zCanvasRegionalGuidanceState>;

const zCanvasInpaintMaskState = zCanvasEntityBase.extend({
  type: z.literal('inpaint_mask'),
  position: zCoordinate,
  fill: zFill,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  noiseLevel: z.number().gte(0).lte(1).optional(),
  denoiseLimit: z.number().gte(0).lte(1).optional(),
});
export type CanvasInpaintMaskState = z.infer<typeof zCanvasInpaintMaskState>;

const zControlNetConfig = z.object({
  type: z.literal('controlnet'),
  model: zServerValidatedModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  controlMode: zControlModeV2,
});
export type ControlNetConfig = z.infer<typeof zControlNetConfig>;

const zT2IAdapterConfig = z.object({
  type: z.literal('t2i_adapter'),
  model: zServerValidatedModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
});
export type T2IAdapterConfig = z.infer<typeof zT2IAdapterConfig>;

const zControlLoRAConfig = z.object({
  type: z.literal('control_lora'),
  weight: z.number().gte(-1).lte(2),
  model: zServerValidatedModelIdentifierField.nullable(),
});
export type ControlLoRAConfig = z.infer<typeof zControlLoRAConfig>;

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
  controlAdapter: z.discriminatedUnion('type', [zControlNetConfig, zT2IAdapterConfig, zControlLoRAConfig]),
});
export type CanvasControlLayerState = z.infer<typeof zCanvasControlLayerState>;

const zBoundingBoxScaleMethod = z.enum(['none', 'auto', 'manual']);
export type BoundingBoxScaleMethod = z.infer<typeof zBoundingBoxScaleMethod>;
export const isBoundingBoxScaleMethod = (v: unknown): v is BoundingBoxScaleMethod =>
  zBoundingBoxScaleMethod.safeParse(v).success;

const zCanvasEntityState = z.discriminatedUnion('type', [
  zCanvasRasterLayerState,
  zCanvasControlLayerState,
  zCanvasRegionalGuidanceState,
  zCanvasInpaintMaskState,
  zCanvasReferenceImageState,
]);
export type CanvasEntityState = z.infer<typeof zCanvasEntityState>;

const zCanvasRenderableEntityState = z.discriminatedUnion('type', [
  zCanvasRasterLayerState,
  zCanvasControlLayerState,
  zCanvasRegionalGuidanceState,
  zCanvasInpaintMaskState,
]);
export type CanvasRenderableEntityState = z.infer<typeof zCanvasRenderableEntityState>;
export type CanvasRenderableEntityType = CanvasRenderableEntityState['type'];

const zCanvasEntityType = z.union([
  zCanvasRasterLayerState.shape.type,
  zCanvasControlLayerState.shape.type,
  zCanvasRegionalGuidanceState.shape.type,
  zCanvasInpaintMaskState.shape.type,
  zCanvasReferenceImageState.shape.type,
]);
export type CanvasEntityType = z.infer<typeof zCanvasEntityType>;

export const zCanvasEntityIdentifer = z.object({
  id: zId,
  type: zCanvasEntityType,
});
export type CanvasEntityIdentifier<T extends CanvasEntityType = CanvasEntityType> = { id: string; type: T };
export type CanvasRenderableEntityIdentifier = CanvasEntityIdentifier<CanvasRenderableEntityType>;
export type LoRA = {
  id: string;
  isEnabled: boolean;
  model: ParameterLoRAModel;
  weight: number;
};

export type StagingAreaImage = {
  type: 'staged';
  sessionId: string;
  imageDTO: ImageDTO;
  offsetX: number;
  offsetY: number;
};
export type StagingAreaProgressImage = {
  type: 'progress';
  sessionId: string;
};
export type EphemeralProgressImage = { sessionId: string; image: ProgressImage };

export const zAspectRatioID = z.enum(['Free', '16:9', '3:2', '4:3', '1:1', '3:4', '2:3', '9:16']);

export const zImagen3AspectRatioID = z.enum(['16:9', '4:3', '1:1', '3:4', '9:16']);
export const isImagenAspectRatioID = (v: unknown): v is z.infer<typeof zImagen3AspectRatioID> =>
  zImagen3AspectRatioID.safeParse(v).success;

export const zChatGPT4oAspectRatioID = z.enum(['3:2', '1:1', '2:3']);
export const isChatGPT4oAspectRatioID = (v: unknown): v is z.infer<typeof zChatGPT4oAspectRatioID> =>
  zChatGPT4oAspectRatioID.safeParse(v).success;

export type AspectRatioID = z.infer<typeof zAspectRatioID>;
export const isAspectRatioID = (v: unknown): v is AspectRatioID => zAspectRatioID.safeParse(v).success;

const zAspectRatioConfig = z.object({
  id: zAspectRatioID,
  value: z.number().gt(0),
  isLocked: z.boolean(),
});
type AspectRatioConfig = z.infer<typeof zAspectRatioConfig>;

export const DEFAULT_ASPECT_RATIO_CONFIG: AspectRatioConfig = {
  id: '1:1',
  value: 1,
  isLocked: false,
};

const zBboxState = z.object({
  rect: z.object({
    x: z.number().int(),
    y: z.number().int(),
    width: zParameterImageDimension,
    height: zParameterImageDimension,
  }),
  aspectRatio: zAspectRatioConfig,
  scaledSize: z.object({
    width: zParameterImageDimension,
    height: zParameterImageDimension,
  }),
  scaleMethod: zBoundingBoxScaleMethod,
  modelBase: zMainModelBase,
});

const zParamsState = z.object({
  maskBlur: z.number().default(16),
  maskBlurMethod: zParameterMaskBlurMethod.default('box'),
  canvasCoherenceMode: zParameterCanvasCoherenceMode.default('Gaussian Blur'),
  canvasCoherenceMinDenoise: zParameterStrength.default(0),
  canvasCoherenceEdgeSize: z.number().default(16),
  infillMethod: z.string().default('lama'),
  infillTileSize: z.number().default(32),
  infillPatchmatchDownscaleSize: z.number().default(1),
  infillColorValue: zRgbaColor.default({ r: 0, g: 0, b: 0, a: 1 }),
  cfgScale: zParameterCFGScale.default(7.5),
  cfgRescaleMultiplier: zParameterCFGRescaleMultiplier.default(0),
  guidance: zParameterGuidance.default(4),
  img2imgStrength: zParameterStrength.default(0.75),
  optimizedDenoisingEnabled: z.boolean().default(true),
  iterations: z.number().default(1),
  scheduler: zParameterScheduler.default('dpmpp_3m_k'),
  upscaleScheduler: zParameterScheduler.default('kdpm_2'),
  upscaleCfgScale: zParameterCFGScale.default(2),
  seed: zParameterSeed.default(0),
  shouldRandomizeSeed: z.boolean().default(true),
  steps: zParameterSteps.default(30),
  model: zParameterModel.nullable().default(null),
  vae: zParameterVAEModel.nullable().default(null),
  vaePrecision: zParameterPrecision.default('fp32'),
  fluxVAE: zParameterVAEModel.nullable().default(null),
  seamlessXAxis: z.boolean().default(false),
  seamlessYAxis: z.boolean().default(false),
  clipSkip: z.number().default(0),
  shouldUseCpuNoise: z.boolean().default(true),
  positivePrompt: zParameterPositivePrompt.default(''),
  negativePrompt: zParameterNegativePrompt.default(''),
  positivePrompt2: zParameterPositiveStylePromptSDXL.default(''),
  negativePrompt2: zParameterNegativeStylePromptSDXL.default(''),
  shouldConcatPrompts: z.boolean().default(true),
  refinerModel: zParameterSDXLRefinerModel.nullable().default(null),
  refinerSteps: z.number().default(20),
  refinerCFGScale: z.number().default(7.5),
  refinerScheduler: zParameterScheduler.default('euler'),
  refinerPositiveAestheticScore: z.number().default(6),
  refinerNegativeAestheticScore: z.number().default(2.5),
  refinerStart: z.number().default(0.8),
  t5EncoderModel: zParameterT5EncoderModel.nullable().default(null),
  clipEmbedModel: zParameterCLIPEmbedModel.nullable().default(null),
  clipLEmbedModel: zParameterCLIPLEmbedModel.nullable().default(null),
  clipGEmbedModel: zParameterCLIPGEmbedModel.nullable().default(null),
  controlLora: zParameterControlLoRAModel.nullable().default(null),
});
export type ParamsState = z.infer<typeof zParamsState>;
const INITIAL_PARAMS_STATE = zParamsState.parse({});
export const getInitialParamsState = () => deepClone(INITIAL_PARAMS_STATE);

const zInpaintMasks = z.object({
  isHidden: z.boolean(),
  entities: z.array(zCanvasInpaintMaskState),
});
const zRasterLayers = z.object({
  isHidden: z.boolean(),
  entities: z.array(zCanvasRasterLayerState),
});
const zControlLayers = z.object({
  isHidden: z.boolean(),
  entities: z.array(zCanvasControlLayerState),
});
const zRegionalGuidance = z.object({
  isHidden: z.boolean(),
  entities: z.array(zCanvasRegionalGuidanceState),
});
const zReferenceImages = z.object({
  entities: z.array(zCanvasReferenceImageState),
});
const zCanvasState = z.object({
  _version: z.literal(3).default(3),
  selectedEntityIdentifier: zCanvasEntityIdentifer.nullable().default(null),
  bookmarkedEntityIdentifier: zCanvasEntityIdentifer.nullable().default(null),
  inpaintMasks: zInpaintMasks.default({ isHidden: false, entities: [] }),
  rasterLayers: zRasterLayers.default({ isHidden: false, entities: [] }),
  controlLayers: zControlLayers.default({ isHidden: false, entities: [] }),
  regionalGuidance: zRegionalGuidance.default({ isHidden: false, entities: [] }),
  referenceImages: zReferenceImages.default({ entities: [] }),
  bbox: zBboxState.default({
    rect: { x: 0, y: 0, width: 512, height: 512 },
    aspectRatio: DEFAULT_ASPECT_RATIO_CONFIG,
    scaleMethod: 'auto',
    scaledSize: { width: 512, height: 512 },
    modelBase: 'sd-1',
  }),
});
export type CanvasState = z.infer<typeof zCanvasState>;

/**
 * Gets a fresh canvas initial state with no references in memory to existing objects.
 */
const CANVAS_INITIAL_STATE = zCanvasState.parse({});
export const getInitialCanvasState = () => deepClone(CANVAS_INITIAL_STATE);

export const zCanvasMetadata = z.object({
  inpaintMasks: z.array(zCanvasInpaintMaskState),
  rasterLayers: z.array(zCanvasRasterLayerState),
  controlLayers: z.array(zCanvasControlLayerState),
  regionalGuidance: z.array(zCanvasRegionalGuidanceState),
  referenceImages: z.array(zCanvasReferenceImageState),
});
export type CanvasMetadata = z.infer<typeof zCanvasMetadata>;

export type StageAttrs = {
  x: Coordinate['x'];
  y: Coordinate['y'];
  width: Dimensions['width'];
  height: Dimensions['height'];
  scale: number;
};

export type EntityIdentifierPayload<
  T extends JsonObject | void = void,
  U extends CanvasEntityType = CanvasEntityType,
> = T extends void
  ? {
      entityIdentifier: CanvasEntityIdentifier<U>;
    }
  : {
      entityIdentifier: CanvasEntityIdentifier<U>;
    } & T;

export type EntityMovedToPayload = EntityIdentifierPayload<{ position: Coordinate }>;
export type EntityMovedByPayload = EntityIdentifierPayload<{ offset: Coordinate }>;
export type EntityBrushLineAddedPayload = EntityIdentifierPayload<{
  brushLine: CanvasBrushLineState | CanvasBrushLineWithPressureState;
}>;
export type EntityEraserLineAddedPayload = EntityIdentifierPayload<{
  eraserLine: CanvasEraserLineState | CanvasEraserLineWithPressureState;
}>;
export type EntityRectAddedPayload = EntityIdentifierPayload<{ rect: CanvasRectState }>;
export type EntityRasterizedPayload = EntityIdentifierPayload<{
  imageObject: CanvasImageState;
  position: Coordinate;
  replaceObjects: boolean;
  isSelected?: boolean;
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

export type CanvasEntityStateFromType<T extends CanvasEntityType> = Extract<CanvasEntityState, { type: T }>;

export function isRenderableEntityType(
  entityType: CanvasEntityState['type']
): entityType is CanvasRenderableEntityState['type'] {
  return (
    entityType === 'raster_layer' ||
    entityType === 'control_layer' ||
    entityType === 'regional_guidance' ||
    entityType === 'inpaint_mask'
  );
}

export function isRasterLayerEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'raster_layer'> {
  return entityIdentifier.type === 'raster_layer';
}

export function isControlLayerEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'control_layer'> {
  return entityIdentifier.type === 'control_layer';
}

export function isInpaintMaskEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'inpaint_mask'> {
  return entityIdentifier.type === 'inpaint_mask';
}

export function isRegionalGuidanceEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'regional_guidance'> {
  return entityIdentifier.type === 'regional_guidance';
}

export function isFilterableEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'raster_layer'> | CanvasEntityIdentifier<'control_layer'> {
  return isRasterLayerEntityIdentifier(entityIdentifier) || isControlLayerEntityIdentifier(entityIdentifier);
}

export function isSegmentableEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'raster_layer'> | CanvasEntityIdentifier<'control_layer'> {
  return isRasterLayerEntityIdentifier(entityIdentifier) || isControlLayerEntityIdentifier(entityIdentifier);
}

export function isTransformableEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is
  | CanvasEntityIdentifier<'raster_layer'>
  | CanvasEntityIdentifier<'control_layer'>
  | CanvasEntityIdentifier<'inpaint_mask'>
  | CanvasEntityIdentifier<'regional_guidance'> {
  return (
    isRasterLayerEntityIdentifier(entityIdentifier) ||
    isControlLayerEntityIdentifier(entityIdentifier) ||
    isInpaintMaskEntityIdentifier(entityIdentifier) ||
    isRegionalGuidanceEntityIdentifier(entityIdentifier)
  );
}

export function isSaveableEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'raster_layer'> | CanvasEntityIdentifier<'control_layer'> {
  return isRasterLayerEntityIdentifier(entityIdentifier) || isControlLayerEntityIdentifier(entityIdentifier);
}

export function isRenderableEntity(entity: CanvasEntityState): entity is CanvasRenderableEntityState {
  return isRenderableEntityType(entity.type);
}

export function isRenderableEntityIdentifier(
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasRenderableEntityIdentifier {
  return isRenderableEntityType(entityIdentifier.type);
}

export const getEntityIdentifier = <T extends CanvasEntityType>(
  entity: Extract<CanvasEntityState, { type: T }>
): CanvasEntityIdentifier<T> => {
  return { id: entity.id, type: entity.type };
};

export const isMaskEntityIdentifier = (
  entityIdentifier: CanvasEntityIdentifier
): entityIdentifier is CanvasEntityIdentifier<'inpaint_mask' | 'regional_guidance'> => {
  return isInpaintMaskEntityIdentifier(entityIdentifier) || isRegionalGuidanceEntityIdentifier(entityIdentifier);
};
