import { NUMPY_RAND_MAX } from 'app/constants';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { zModelIdentifierField, zSchedulerField } from 'features/nodes/types/common';
import { z } from 'zod';

/**
 * Schemas, types and type guards for parameters.
 *
 * Parameters need schemas if we want to recall them from metadata or some untrusted source.
 *
 * Each parameter needs:
 * - a zod schema
 * - a type alias, inferred from the zod schema
 * - a combo validation/type guard function, which returns true if the value is valid, should
 * simply be the zod schema's safeParse function
 */

// #region Positive prompt
const zParameterPositivePrompt = z.string();
export type ParameterPositivePrompt = z.infer<typeof zParameterPositivePrompt>;
export const isParameterPositivePrompt = (val: unknown): val is ParameterPositivePrompt =>
  zParameterPositivePrompt.safeParse(val).success;
// #endregion

// #region Negative prompt
const zParameterNegativePrompt = z.string();
export type ParameterNegativePrompt = z.infer<typeof zParameterNegativePrompt>;
export const isParameterNegativePrompt = (val: unknown): val is ParameterNegativePrompt =>
  zParameterNegativePrompt.safeParse(val).success;
// #endregion

// #region Positive style prompt (SDXL)
const zParameterPositiveStylePromptSDXL = z.string();
export type ParameterPositiveStylePromptSDXL = z.infer<typeof zParameterPositiveStylePromptSDXL>;
export const isParameterPositiveStylePromptSDXL = (val: unknown): val is ParameterPositiveStylePromptSDXL =>
  zParameterPositiveStylePromptSDXL.safeParse(val).success;
// #endregion

// #region Positive style prompt (SDXL)
const zParameterNegativeStylePromptSDXL = z.string();
export type ParameterNegativeStylePromptSDXL = z.infer<typeof zParameterNegativeStylePromptSDXL>;
export const isParameterNegativeStylePromptSDXL = (val: unknown): val is ParameterNegativeStylePromptSDXL =>
  zParameterNegativeStylePromptSDXL.safeParse(val).success;
// #endregion

// #region Steps
const zParameterSteps = z.number().int().min(1);
export type ParameterSteps = z.infer<typeof zParameterSteps>;
export const isParameterSteps = (val: unknown): val is ParameterSteps => zParameterSteps.safeParse(val).success;
// #endregion

// #region CFG scale parameter
const zParameterCFGScale = z.number().min(1);
export type ParameterCFGScale = z.infer<typeof zParameterCFGScale>;
export const isParameterCFGScale = (val: unknown): val is ParameterCFGScale =>
  zParameterCFGScale.safeParse(val).success;
// #endregion

// #region CFG Rescale Multiplier
const zParameterCFGRescaleMultiplier = z.number().gte(0).lt(1);
export type ParameterCFGRescaleMultiplier = z.infer<typeof zParameterCFGRescaleMultiplier>;
export const isParameterCFGRescaleMultiplier = (val: unknown): val is ParameterCFGRescaleMultiplier =>
  zParameterCFGRescaleMultiplier.safeParse(val).success;
// #endregion

// #region Scheduler
const zParameterScheduler = zSchedulerField;
export type ParameterScheduler = z.infer<typeof zParameterScheduler>;
export const isParameterScheduler = (val: unknown): val is ParameterScheduler =>
  zParameterScheduler.safeParse(val).success;
// #endregion

// #region seed
const zParameterSeed = z.number().int().min(0).max(NUMPY_RAND_MAX);
export type ParameterSeed = z.infer<typeof zParameterSeed>;
export const isParameterSeed = (val: unknown): val is ParameterSeed => zParameterSeed.safeParse(val).success;
// #endregion

// #region Width
const zParameterWidth = z
  .number()
  .min(64)
  .transform((val) => roundToMultiple(val, 8));
export type ParameterWidth = z.infer<typeof zParameterWidth>;
export const isParameterWidth = (val: unknown): val is ParameterWidth => zParameterWidth.safeParse(val).success;
// #endregion

// #region Height
const zParameterHeight = zParameterWidth;
export type ParameterHeight = z.infer<typeof zParameterHeight>;
export const isParameterHeight = (val: unknown): val is ParameterHeight => zParameterHeight.safeParse(val).success;
// #endregion

// #region Model
export const zParameterModel = zModelIdentifierField;
export type ParameterModel = z.infer<typeof zParameterModel>;
// #endregion

// #region SDXL Refiner Model
const zParameterSDXLRefinerModel = zModelIdentifierField;
export type ParameterSDXLRefinerModel = z.infer<typeof zParameterSDXLRefinerModel>;
// #endregion

// #region VAE Model
export const zParameterVAEModel = zModelIdentifierField;
export type ParameterVAEModel = z.infer<typeof zParameterVAEModel>;
// #endregion

// #region LoRA Model
const zParameterLoRAModel = zModelIdentifierField;
export type ParameterLoRAModel = z.infer<typeof zParameterLoRAModel>;
// #endregion

// #region ControlNet Model
const zParameterControlNetModel = zModelIdentifierField;
export type ParameterControlNetModel = z.infer<typeof zParameterControlNetModel>;
// #endregion

// #region IP Adapter Model
const zParameterIPAdapterModel = zModelIdentifierField;
export type ParameterIPAdapterModel = z.infer<typeof zParameterIPAdapterModel>;
// #endregion

// #region T2I Adapter Model
const zParameterT2IAdapterModel = zModelIdentifierField;
export type ParameterT2IAdapterModel = z.infer<typeof zParameterT2IAdapterModel>;
// #endregion

// #region Strength (l2l strength)
const zParameterStrength = z.number().min(0).max(1);
export type ParameterStrength = z.infer<typeof zParameterStrength>;
export const isParameterStrength = (val: unknown): val is ParameterStrength =>
  zParameterStrength.safeParse(val).success;
// #endregion

// #region Precision
const zParameterPrecision = z.enum(['fp16', 'fp32']);
export type ParameterPrecision = z.infer<typeof zParameterPrecision>;
export const isParameterPrecision = (val: unknown): val is ParameterPrecision =>
  zParameterPrecision.safeParse(val).success;
// #endregion

// #region HRF Method
const zParameterHRFMethod = z.enum(['ESRGAN', 'bilinear']);
export type ParameterHRFMethod = z.infer<typeof zParameterHRFMethod>;
export const isParameterHRFMethod = (val: unknown): val is ParameterHRFMethod =>
  zParameterHRFMethod.safeParse(val).success;
// #endregion

// #region HRF Enabled
const zParameterHRFEnabled = z.boolean();
export type ParameterHRFEnabled = z.infer<typeof zParameterHRFEnabled>;
export const isParameterHRFEnabled = (val: unknown): val is boolean =>
  zParameterHRFEnabled.safeParse(val).success && val !== null && val !== undefined;
// #endregion

// #region SDXL Refiner Positive Aesthetic Score
const zParameterSDXLRefinerPositiveAestheticScore = z.number().min(1).max(10);
export type ParameterSDXLRefinerPositiveAestheticScore = z.infer<typeof zParameterSDXLRefinerPositiveAestheticScore>;
export const isParameterSDXLRefinerPositiveAestheticScore = (
  val: unknown
): val is ParameterSDXLRefinerPositiveAestheticScore =>
  zParameterSDXLRefinerPositiveAestheticScore.safeParse(val).success;
// #endregion

// #region SDXL Refiner Negative Aesthetic Score
const zParameterSDXLRefinerNegativeAestheticScore = zParameterSDXLRefinerPositiveAestheticScore;
export type ParameterSDXLRefinerNegativeAestheticScore = z.infer<typeof zParameterSDXLRefinerNegativeAestheticScore>;
export const isParameterSDXLRefinerNegativeAestheticScore = (
  val: unknown
): val is ParameterSDXLRefinerNegativeAestheticScore =>
  zParameterSDXLRefinerNegativeAestheticScore.safeParse(val).success;
// #endregion

// #region SDXL Refiner Start
const zParameterSDXLRefinerStart = z.number().min(0).max(1);
export type ParameterSDXLRefinerStart = z.infer<typeof zParameterSDXLRefinerStart>;
export const isParameterSDXLRefinerStart = (val: unknown): val is ParameterSDXLRefinerStart =>
  zParameterSDXLRefinerStart.safeParse(val).success;
// #endregion

// #region Mask Blur Method
const zParameterMaskBlurMethod = z.enum(['box', 'gaussian']);
export type ParameterMaskBlurMethod = z.infer<typeof zParameterMaskBlurMethod>;
// #endregion

// #region Canvas Coherence Mode
const zParameterCanvasCoherenceMode = z.enum(['Gaussian Blur', 'Box Blur', 'Staged']);
export type ParameterCanvasCoherenceMode = z.infer<typeof zParameterCanvasCoherenceMode>;
export const isParameterCanvasCoherenceMode = (val: unknown): val is ParameterCanvasCoherenceMode =>
  zParameterCanvasCoherenceMode.safeParse(val).success;
// #endregion

// #region LoRA weight
const zLoRAWeight = z.number();
type ParameterLoRAWeight = z.infer<typeof zLoRAWeight>;
export const isParameterLoRAWeight = (val: unknown): val is ParameterLoRAWeight => zLoRAWeight.safeParse(val).success;
// #endregion
