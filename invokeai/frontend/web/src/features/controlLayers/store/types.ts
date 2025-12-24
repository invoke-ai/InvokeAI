import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import { zMainModelBase, zModelIdentifierField } from 'features/nodes/types/common';
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
  zParameterPositivePrompt,
  zParameterPrecision,
  zParameterScheduler,
  zParameterSDXLRefinerModel,
  zParameterSeed,
  zParameterSteps,
  zParameterStrength,
  zParameterT5EncoderModel,
  zParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import type { JsonObject } from 'type-fest';
import { z } from 'zod';

const zId = z.string().min(1);
const zName = z.string().min(1).nullable();

export const zImageWithDims = z.object({
  image_name: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
export type ImageWithDims = z.infer<typeof zImageWithDims>;

const zCropBox = z.object({
  x: z.number().min(0),
  y: z.number().min(0),
  width: z.number().positive(),
  height: z.number().positive(),
});
// This new schema is an extension of zImageWithDims, with an optional crop field.
//
// When we added cropping support to certain entities (e.g. Ref Images, video Starting Frame Image), we changed
// their schemas from using zImageWithDims to this new schema. To support loading pre-existing entities that
// were created before cropping was supported, we can use zod's preprocess to transform old data into the new format.
// Its essentially a data migration step.
//
// This parsing happens currently in two places:
// - Recalling metadata.
// - Loading/rehydrating persisted client state from storage.
const zCroppableImageWithDims = z.preprocess(
  (val) => {
    try {
      const imageWithDims = zImageWithDims.parse(val);
      const migrated = { original: { image: deepClone(imageWithDims) } };
      return migrated;
    } catch {
      return val;
    }
  },
  z.object({
    original: z.object({ image: zImageWithDims }),
    crop: z
      .object({
        box: zCropBox,
        ratio: z.number().gt(0).nullable(),
        image: zImageWithDims,
      })
      .optional(),
  })
);
export type CroppableImageWithDims = z.infer<typeof zCroppableImageWithDims>;

const zImageWithDimsDataURL = z.object({
  dataURL: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});

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

const _zTool = z.enum(['brush', 'eraser', 'move', 'rect', 'view', 'bbox', 'colorPicker', 'text']);
export type Tool = z.infer<typeof _zTool>;

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
export const RGBA_WHITE: RgbaColor = { r: 255, g: 255, b: 255, a: 1 };

const zOpacity = z.number().gte(0).lte(1);

const _zDimensions = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
export type Dimensions = z.infer<typeof _zDimensions>;

const zCoordinate = z.object({
  x: z.number(),
  y: z.number(),
});
export type Coordinate = z.infer<typeof zCoordinate>;
const _zCoordinateWithPressure = z.object({
  x: z.number(),
  y: z.number(),
  pressure: z.number(),
});
export type CoordinateWithPressure = z.infer<typeof _zCoordinateWithPressure>;

const SAM_POINT_LABELS = {
  background: -1,
  neutral: 0,
  foreground: 1,
} as const;

const zSAMPointLabel = z.nativeEnum(SAM_POINT_LABELS);
export type SAMPointLabel = z.infer<typeof zSAMPointLabel>;

export const zSAMPointLabelString = z.enum(['background', 'neutral', 'foreground']);
export type SAMPointLabelString = z.infer<typeof zSAMPointLabelString>;

export const zSAMModel = z.enum(['SAM1', 'SAM2']);
export type SAMModel = z.infer<typeof zSAMModel>;

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

const _zSAMPoint = z.object({
  x: z.number().int().gte(0),
  y: z.number().int().gte(0),
  label: zSAMPointLabel,
});
type SAMPoint = z.infer<typeof _zSAMPoint>;
export type SAMPointWithId = SAMPoint & { id: string };

const zRect = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number().min(1),
  height: z.number().min(1),
});
export type Rect = z.infer<typeof zRect>;

const _zRectWithRotation = zRect.extend({
  rotation: z.number(),
});
export type RectWithRotation = z.infer<typeof _zRectWithRotation>;

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
  image: z.union([zImageWithDims, zImageWithDimsDataURL]),
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
  image: zCroppableImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  method: zIPMethodV2,
  clipVisionModel: zCLIPVisionModelV2,
});
export type IPAdapterConfig = z.infer<typeof zIPAdapterConfig>;

const zRegionalGuidanceIPAdapterConfig = z.object({
  type: z.literal('ip_adapter'),
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(-1).lte(2),
  beginEndStepPct: zBeginEndStepPct,
  method: zIPMethodV2,
  clipVisionModel: zCLIPVisionModelV2,
});
export type RegionalGuidanceIPAdapterConfig = z.infer<typeof zRegionalGuidanceIPAdapterConfig>;

const zFLUXReduxImageInfluence = z.enum(['lowest', 'low', 'medium', 'high', 'highest']);
export const isFLUXReduxImageInfluence = (v: unknown): v is FLUXReduxImageInfluence =>
  zFLUXReduxImageInfluence.safeParse(v).success;
export type FLUXReduxImageInfluence = z.infer<typeof zFLUXReduxImageInfluence>;
const zFLUXReduxConfig = z.object({
  type: z.literal('flux_redux'),
  image: zCroppableImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  imageInfluence: zFLUXReduxImageInfluence.default('highest'),
});
export type FLUXReduxConfig = z.infer<typeof zFLUXReduxConfig>;
const zRegionalGuidanceFLUXReduxConfig = z.object({
  type: z.literal('flux_redux'),
  image: zImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
  imageInfluence: zFLUXReduxImageInfluence.default('highest'),
});
type RegionalGuidanceFLUXReduxConfig = z.infer<typeof zRegionalGuidanceFLUXReduxConfig>;

const zFluxKontextReferenceImageConfig = z.object({
  type: z.literal('flux_kontext_reference_image'),
  image: zCroppableImageWithDims.nullable(),
  model: zModelIdentifierField.nullable(),
});
export type FluxKontextReferenceImageConfig = z.infer<typeof zFluxKontextReferenceImageConfig>;

const zCanvasEntityBase = z.object({
  id: zId,
  name: zName,
  isEnabled: z.boolean(),
  isLocked: z.boolean(),
});

export const zRefImageState = z.object({
  id: zId,
  isEnabled: z.boolean().default(true),
  config: z.discriminatedUnion('type', [zIPAdapterConfig, zFLUXReduxConfig, zFluxKontextReferenceImageConfig]),
});
export type RefImageState = z.infer<typeof zRefImageState>;

export const isIPAdapterConfig = (config: RefImageState['config']): config is IPAdapterConfig =>
  config.type === 'ip_adapter';

export const isFLUXReduxConfig = (config: RefImageState['config']): config is FLUXReduxConfig =>
  config.type === 'flux_redux';

export const isFluxKontextReferenceImageConfig = (
  config: RefImageState['config']
): config is FluxKontextReferenceImageConfig => config.type === 'flux_kontext_reference_image';

const zFillStyle = z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']);
export type FillStyle = z.infer<typeof zFillStyle>;
export const isFillStyle = (v: unknown): v is FillStyle => zFillStyle.safeParse(v).success;
const zFill = z.object({ style: zFillStyle, color: zRgbColor });

const zRegionalGuidanceRefImageState = z.object({
  id: zId,
  config: z.discriminatedUnion('type', [zRegionalGuidanceIPAdapterConfig, zRegionalGuidanceFLUXReduxConfig]),
});
export type RegionalGuidanceRefImageState = z.infer<typeof zRegionalGuidanceRefImageState>;

export const isRegionalGuidanceIPAdapterConfig = (
  config: RegionalGuidanceRefImageState['config']
): config is RegionalGuidanceIPAdapterConfig => config.type === 'ip_adapter';

export const isRegionalGuidanceFLUXReduxConfig = (
  config: RegionalGuidanceRefImageState['config']
): config is RegionalGuidanceFLUXReduxConfig => config.type === 'flux_redux';

const zCanvasRegionalGuidanceState = zCanvasEntityBase.extend({
  type: z.literal('regional_guidance'),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  fill: zFill,
  positivePrompt: zParameterPositivePrompt.nullable(),
  negativePrompt: zParameterNegativePrompt.nullable(),
  referenceImages: z.array(zRegionalGuidanceRefImageState),
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

const zControlLoRAConfig = z.object({
  type: z.literal('control_lora'),
  weight: z.number().gte(-1).lte(2),
  model: zModelIdentifierField.nullable(),
});
export type ControlLoRAConfig = z.infer<typeof zControlLoRAConfig>;

const zZImageControlConfig = z.object({
  type: z.literal('z_image_control'),
  model: zModelIdentifierField.nullable(),
  weight: z.number().gte(0).lte(2), // control_context_scale, recommended 0.65-0.80
  beginEndStepPct: zBeginEndStepPct,
});
export type ZImageControlConfig = z.infer<typeof zZImageControlConfig>;

/**
 * All simple params normalized to `[-1, 1]` except sharpness `[0, 1]`.
 *
 * - Brightness: -1 (darken) to 1 (brighten)
 * - Contrast: -1 (decrease contrast) to 1 (increase contrast)
 * - Saturation: -1 (desaturate) to 1 (saturate)
 * - Temperature: -1 (cooler/blue) to 1 (warmer/yellow)
 * - Tint: -1 (greener) to 1 (more magenta)
 * - Sharpness: 0 (no sharpening) to 1 (maximum sharpening)
 */
export const zSimpleAdjustmentsConfig = z.object({
  brightness: z.number().gte(-1).lte(1),
  contrast: z.number().gte(-1).lte(1),
  saturation: z.number().gte(-1).lte(1),
  temperature: z.number().gte(-1).lte(1),
  tint: z.number().gte(-1).lte(1),
  sharpness: z.number().gte(0).lte(1),
});
export type SimpleAdjustmentsConfig = z.infer<typeof zSimpleAdjustmentsConfig>;

const zUint8 = z.number().int().min(0).max(255);
const zChannelPoints = z.array(z.tuple([zUint8, zUint8])).min(2);
const zChannelName = z.enum(['master', 'r', 'g', 'b']);
const zCurvesAdjustmentsConfig = z.record(zChannelName, zChannelPoints);
export type ChannelName = z.infer<typeof zChannelName>;
export type ChannelPoints = z.infer<typeof zChannelPoints>;
export type CurvesAdjustmentsConfig = z.infer<typeof zCurvesAdjustmentsConfig>;

/**
 * The curves adjustments are stored as LUTs in the Konva node attributes. Konva will use these values when applying
 * the filter.
 */
export const zCurvesAdjustmentsLUTs = z.record(zChannelName, z.array(zUint8));

const zRasterLayerAdjustments = z.object({
  version: z.literal(1),
  enabled: z.boolean(),
  collapsed: z.boolean(),
  mode: z.enum(['simple', 'curves']),
  simple: zSimpleAdjustmentsConfig,
  curves: zCurvesAdjustmentsConfig,
});
export type RasterLayerAdjustments = z.infer<typeof zRasterLayerAdjustments>;

const zCanvasRasterLayerState = zCanvasEntityBase.extend({
  type: z.literal('raster_layer'),
  position: zCoordinate,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  // Optional per-layer color adjustments (simple + curves). When undefined, no adjustments are applied.
  adjustments: zRasterLayerAdjustments.optional(),
});
export type CanvasRasterLayerState = z.infer<typeof zCanvasRasterLayerState>;

const zCanvasControlLayerState = zCanvasRasterLayerState.extend({
  type: z.literal('control_layer'),
  withTransparencyEffect: z.boolean(),
  controlAdapter: z.discriminatedUnion('type', [
    zControlNetConfig,
    zT2IAdapterConfig,
    zControlLoRAConfig,
    zZImageControlConfig,
  ]),
});
export type CanvasControlLayerState = z.infer<typeof zCanvasControlLayerState>;

const zBoundingBoxScaleMethod = z.enum(['none', 'auto', 'manual']);
export type BoundingBoxScaleMethod = z.infer<typeof zBoundingBoxScaleMethod>;
export const isBoundingBoxScaleMethod = (v: unknown): v is BoundingBoxScaleMethod =>
  zBoundingBoxScaleMethod.safeParse(v).success;

const _zCanvasEntityState = z.discriminatedUnion('type', [
  zCanvasRasterLayerState,
  zCanvasControlLayerState,
  zCanvasRegionalGuidanceState,
  zCanvasInpaintMaskState,
]);
export type CanvasEntityState = z.infer<typeof _zCanvasEntityState>;

const zCanvasEntityType = z.union([
  zCanvasRasterLayerState.shape.type,
  zCanvasControlLayerState.shape.type,
  zCanvasRegionalGuidanceState.shape.type,
  zCanvasInpaintMaskState.shape.type,
]);
export type CanvasEntityType = z.infer<typeof zCanvasEntityType>;

export const zCanvasEntityIdentifer = z.object({
  id: zId,
  type: zCanvasEntityType,
});
export type CanvasEntityIdentifier<T extends CanvasEntityType = CanvasEntityType> = { id: string; type: T };

export const zLoRA = z.object({
  id: z.string(),
  isEnabled: z.boolean(),
  model: zModelIdentifierField,
  weight: z.number().gte(-10).lte(10),
});
export type LoRA = z.infer<typeof zLoRA>;

export const zAspectRatioID = z.enum(['Free', '21:9', '16:9', '3:2', '4:3', '1:1', '3:4', '2:3', '9:16', '9:21']);
export type AspectRatioID = z.infer<typeof zAspectRatioID>;
export const isAspectRatioID = (v: unknown): v is AspectRatioID => zAspectRatioID.safeParse(v).success;
export const ASPECT_RATIO_MAP: Record<Exclude<AspectRatioID, 'Free'>, { ratio: number; inverseID: AspectRatioID }> = {
  '21:9': { ratio: 21 / 9, inverseID: '9:21' },
  '16:9': { ratio: 16 / 9, inverseID: '9:16' },
  '3:2': { ratio: 3 / 2, inverseID: '2:3' },
  '4:3': { ratio: 4 / 3, inverseID: '4:3' },
  '1:1': { ratio: 1, inverseID: '1:1' },
  '3:4': { ratio: 3 / 4, inverseID: '4:3' },
  '2:3': { ratio: 2 / 3, inverseID: '3:2' },
  '9:16': { ratio: 9 / 16, inverseID: '16:9' },
  '9:21': { ratio: 9 / 21, inverseID: '21:9' },
};

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

const zDimensionsState = z.object({
  width: zParameterImageDimension,
  height: zParameterImageDimension,
  aspectRatio: zAspectRatioConfig,
});

export const MAX_POSITIVE_PROMPT_HISTORY = 100;
const zPositivePromptHistory = z
  .array(zParameterPositivePrompt)
  .transform((arr) => arr.slice(0, MAX_POSITIVE_PROMPT_HISTORY));

export const zInfillMethod = z.enum(['patchmatch', 'lama', 'cv2', 'color', 'tile']);
export type InfillMethod = z.infer<typeof zInfillMethod>;

export const zParamsState = z.object({
  _version: z.literal(2),
  maskBlur: z.number(),
  maskBlurMethod: zParameterMaskBlurMethod,
  canvasCoherenceMode: zParameterCanvasCoherenceMode,
  canvasCoherenceMinDenoise: zParameterStrength,
  canvasCoherenceEdgeSize: z.number(),
  infillMethod: zInfillMethod,
  infillTileSize: z.number(),
  infillPatchmatchDownscaleSize: z.number(),
  infillColorValue: zRgbaColor,
  cfgScale: zParameterCFGScale,
  cfgRescaleMultiplier: zParameterCFGRescaleMultiplier,
  guidance: zParameterGuidance,
  img2imgStrength: zParameterStrength,
  optimizedDenoisingEnabled: z.boolean(),
  iterations: z.number(),
  scheduler: zParameterScheduler,
  upscaleScheduler: zParameterScheduler,
  upscaleCfgScale: zParameterCFGScale,
  seed: zParameterSeed,
  shouldRandomizeSeed: z.boolean(),
  steps: zParameterSteps,
  model: zParameterModel.nullable(),
  vae: zParameterVAEModel.nullable(),
  vaePrecision: zParameterPrecision,
  fluxVAE: zParameterVAEModel.nullable(),
  seamlessXAxis: z.boolean(),
  seamlessYAxis: z.boolean(),
  clipSkip: z.number(),
  shouldUseCpuNoise: z.boolean(),
  colorCompensation: z.boolean(),
  positivePrompt: zParameterPositivePrompt,
  positivePromptHistory: zPositivePromptHistory,
  negativePrompt: zParameterNegativePrompt,
  refinerModel: zParameterSDXLRefinerModel.nullable(),
  refinerSteps: z.number(),
  refinerCFGScale: z.number(),
  refinerScheduler: zParameterScheduler,
  refinerPositiveAestheticScore: z.number(),
  refinerNegativeAestheticScore: z.number(),
  refinerStart: z.number(),
  t5EncoderModel: zParameterT5EncoderModel.nullable(),
  clipEmbedModel: zParameterCLIPEmbedModel.nullable(),
  clipLEmbedModel: zParameterCLIPLEmbedModel.nullable(),
  clipGEmbedModel: zParameterCLIPGEmbedModel.nullable(),
  controlLora: zParameterControlLoRAModel.nullable(),
  // Z-Image model components - can use separate models or extract from a Diffusers source
  zImageVaeModel: zParameterVAEModel.nullable(), // Optional: Separate FLUX VAE
  zImageQwen3EncoderModel: zModelIdentifierField.nullable(), // Optional: Separate Qwen3 Encoder
  zImageQwen3SourceModel: zParameterModel.nullable(), // Diffusers Z-Image model (fallback for VAE/Encoder)
  dimensions: zDimensionsState,
});
export type ParamsState = z.infer<typeof zParamsState>;
export const getInitialParamsState = (): ParamsState => ({
  _version: 2,
  maskBlur: 16,
  maskBlurMethod: 'box',
  canvasCoherenceMode: 'Gaussian Blur',
  canvasCoherenceMinDenoise: 0,
  canvasCoherenceEdgeSize: 16,
  infillMethod: 'lama',
  infillTileSize: 32,
  infillPatchmatchDownscaleSize: 1,
  infillColorValue: { r: 0, g: 0, b: 0, a: 1 },
  cfgScale: 7.5,
  cfgRescaleMultiplier: 0,
  guidance: 4,
  img2imgStrength: 0.75,
  optimizedDenoisingEnabled: true,
  iterations: 1,
  scheduler: 'dpmpp_3m_k',
  upscaleScheduler: 'kdpm_2',
  upscaleCfgScale: 2,
  seed: 0,
  shouldRandomizeSeed: true,
  steps: 30,
  model: null,
  vae: null,
  vaePrecision: 'fp32',
  fluxVAE: null,
  seamlessXAxis: false,
  seamlessYAxis: false,
  clipSkip: 0,
  shouldUseCpuNoise: true,
  colorCompensation: false,
  positivePrompt: '',
  positivePromptHistory: [],
  negativePrompt: null,
  refinerModel: null,
  refinerSteps: 20,
  refinerCFGScale: 7.5,
  refinerScheduler: 'euler',
  refinerPositiveAestheticScore: 6,
  refinerNegativeAestheticScore: 2.5,
  refinerStart: 0.8,
  t5EncoderModel: null,
  clipEmbedModel: null,
  clipLEmbedModel: null,
  clipGEmbedModel: null,
  controlLora: null,
  zImageVaeModel: null,
  zImageQwen3EncoderModel: null,
  zImageQwen3SourceModel: null,
  dimensions: {
    width: 512,
    height: 512,
    aspectRatio: deepClone(DEFAULT_ASPECT_RATIO_CONFIG),
  },
});

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
export const zCanvasState = z.object({
  _version: z.literal(3),
  selectedEntityIdentifier: zCanvasEntityIdentifer.nullable(),
  bookmarkedEntityIdentifier: zCanvasEntityIdentifer.nullable(),
  inpaintMasks: zInpaintMasks,
  rasterLayers: zRasterLayers,
  controlLayers: zControlLayers,
  regionalGuidance: zRegionalGuidance,
  bbox: zBboxState,
});
export type CanvasState = z.infer<typeof zCanvasState>;
export const getInitialCanvasState = (): CanvasState => ({
  _version: 3,
  selectedEntityIdentifier: null,
  bookmarkedEntityIdentifier: null,
  inpaintMasks: { isHidden: false, entities: [] },
  rasterLayers: { isHidden: false, entities: [] },
  controlLayers: { isHidden: false, entities: [] },
  regionalGuidance: { isHidden: false, entities: [] },
  bbox: {
    rect: { x: 0, y: 0, width: 512, height: 512 },
    aspectRatio: deepClone(DEFAULT_ASPECT_RATIO_CONFIG),
    scaleMethod: 'auto',
    scaledSize: { width: 512, height: 512 },
    modelBase: 'sd-1',
  },
});

export const zRefImagesState = z.object({
  selectedEntityId: z.string().nullable(),
  isPanelOpen: z.boolean(),
  entities: z.array(zRefImageState),
});
export type RefImagesState = z.infer<typeof zRefImagesState>;
export const getInitialRefImagesState = (): RefImagesState => ({
  selectedEntityId: null,
  isPanelOpen: false,
  entities: [],
});

export const zCanvasReferenceImageState_OLD = zCanvasEntityBase.extend({
  type: z.literal('reference_image'),
  ipAdapter: z.discriminatedUnion('type', [zIPAdapterConfig, zFLUXReduxConfig]),
});

export const zCanvasMetadata = z.object({
  inpaintMasks: z.array(zCanvasInpaintMaskState),
  rasterLayers: z.array(zCanvasRasterLayerState),
  controlLayers: z.array(zCanvasControlLayerState),
  regionalGuidance: z.array(zCanvasRegionalGuidanceState),
  // referenceImages: z.array(zRefImageState),
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

// Ideally, we'd type `adapter` as `CanvasEntityAdapterBase`, but the generics make this tricky. `CanvasEntityAdapter`
// is a union of all entity adapters and is functionally identical to `CanvasEntityAdapterBase`.
export type LifecycleCallback = (adapter: CanvasEntityAdapter) => Promise<boolean>;
