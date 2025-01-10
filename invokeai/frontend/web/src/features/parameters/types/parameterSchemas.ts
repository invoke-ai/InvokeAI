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

/**
 * Helper to create a type guard from a zod schema. The type guard will infer the schema's TS type.
 * @param schema The zod schema to create a type guard from.
 * @returns A type guard function for the schema.
 */
export const buildTypeGuard = <T extends z.ZodTypeAny>(schema: T) => {
  return (val: unknown): val is z.infer<T> => schema.safeParse(val).success;
};

/**
 * Helper to create a zod schema and a type guard from it.
 * @param schema The zod schema to create a type guard from.
 * @returns A tuple containing the zod schema and the type guard function.
 */
const buildParameter = <T extends z.ZodTypeAny>(schema: T) => [schema, buildTypeGuard(schema)] as const;

// #region Positive prompt
export const [zParameterPositivePrompt, isParameterPositivePrompt] = buildParameter(z.string());
export type ParameterPositivePrompt = z.infer<typeof zParameterPositivePrompt>;
// #endregion

// #region Negative prompt
export const [zParameterNegativePrompt, isParameterNegativePrompt] = buildParameter(z.string());
export type ParameterNegativePrompt = z.infer<typeof zParameterNegativePrompt>;
// #endregion

// #region Positive style prompt (SDXL)
export const [zParameterPositiveStylePromptSDXL, isParameterPositiveStylePromptSDXL] = buildParameter(z.string());
export type ParameterPositiveStylePromptSDXL = z.infer<typeof zParameterPositiveStylePromptSDXL>;
// #endregion

// #region Positive style prompt (SDXL)
export const [zParameterNegativeStylePromptSDXL, isParameterNegativeStylePromptSDXL] = buildParameter(z.string());
export type ParameterNegativeStylePromptSDXL = z.infer<typeof zParameterNegativeStylePromptSDXL>;
// #endregion

// #region Steps
export const [zParameterSteps, isParameterSteps] = buildParameter(z.number().int().min(1));
export type ParameterSteps = z.infer<typeof zParameterSteps>;
// #endregion

// #region CFG scale parameter
export const [zParameterCFGScale, isParameterCFGScale] = buildParameter(z.number().min(1));
export type ParameterCFGScale = z.infer<typeof zParameterCFGScale>;
// #endregion

// #region Guidance parameter
export const [zParameterGuidance, isParameterGuidance] = buildParameter(z.number().min(1));
export type ParameterGuidance = z.infer<typeof zParameterGuidance>;
// #endregion

// #region CFG Rescale Multiplier
export const [zParameterCFGRescaleMultiplier, isParameterCFGRescaleMultiplier] = buildParameter(
  z.number().gte(0).lt(1)
);
export type ParameterCFGRescaleMultiplier = z.infer<typeof zParameterCFGRescaleMultiplier>;
// #endregion

// #region Scheduler
export const [zParameterScheduler, isParameterScheduler] = buildParameter(zSchedulerField);
export type ParameterScheduler = z.infer<typeof zParameterScheduler>;
// #endregion

// #region seed
export const [zParameterSeed, isParameterSeed] = buildParameter(z.number().int().min(0).max(NUMPY_RAND_MAX));
export type ParameterSeed = z.infer<typeof zParameterSeed>;
// #endregion

// #region Width
export const [zParameterImageDimension, isParameterImageDimension] = buildParameter(
  z
    .number()
    .min(64)
    .transform((val) => roundToMultiple(val, 8))
);
export type ParameterWidth = z.infer<typeof zParameterImageDimension>;
export const isParameterWidth = isParameterImageDimension;

// #region Height
export type ParameterHeight = z.infer<typeof zParameterImageDimension>;
export const isParameterHeight = isParameterImageDimension;
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

// #region Control Lora Model
export const zParameterControlLoRAModel = zModelIdentifierField;
export type ParameterControlLoRAModel = z.infer<typeof zParameterControlLoRAModel>;
// #endregion

// #region T5Encoder Model
export const zParameterT5EncoderModel = zModelIdentifierField;
export type ParameterT5EncoderModel = z.infer<typeof zParameterT5EncoderModel>;
// #endregion

// #region CLIP embed Model
export const zParameterCLIPEmbedModel = zModelIdentifierField;
export type ParameterCLIPEmbedModel = z.infer<typeof zParameterCLIPEmbedModel>;
// #endregion

// #region CLIP embed Model
export const zParameterCLIPLEmbedModel = zModelIdentifierField;
export type ParameterCLIPLEmbedModel = z.infer<typeof zParameterCLIPLEmbedModel>;
// #endregion

// #region CLIP embed Model
export const zParameterCLIPGEmbedModel = zModelIdentifierField;
export type ParameterCLIPGEmbedModel = z.infer<typeof zParameterCLIPGEmbedModel>;
// #endregion

// #region LoRA Model
const zParameterLoRAModel = zModelIdentifierField;
export type ParameterLoRAModel = z.infer<typeof zParameterLoRAModel>;
// #endregion

// #region VAE Model
export const zParameterSpandrelImageToImageModel = zModelIdentifierField;
export type ParameterSpandrelImageToImageModel = z.infer<typeof zParameterSpandrelImageToImageModel>;
// #endregion

// #region Strength (l2l strength)
export const [zParameterStrength, isParameterStrength] = buildParameter(z.number().min(0).max(1));
export type ParameterStrength = z.infer<typeof zParameterStrength>;
// #endregion

// #region SeamlessX
export const [zParameterSeamlessX, isParameterSeamlessX] = buildParameter(z.boolean());
export type ParameterSeamlessX = z.infer<typeof zParameterSeamlessX>;
// #endregion

// #region SeamlessY
export const [zParameterSeamlessY, isParameterSeamlessY] = buildParameter(z.boolean());
export type ParameterSeamlessY = z.infer<typeof zParameterSeamlessY>;
// #endregion

// #region Precision
export const [zParameterPrecision, isParameterPrecision] = buildParameter(z.enum(['fp16', 'fp32']));
export type ParameterPrecision = z.infer<typeof zParameterPrecision>;
// #endregion

// #region HRF Method
export const [zParameterHRFMethod, isParameterHRFMethod] = buildParameter(z.enum(['ESRGAN', 'bilinear']));
export type ParameterHRFMethod = z.infer<typeof zParameterHRFMethod>;
// #endregion

// #region HRF Enabled
export const [zParameterHRFEnabled, isParameterHRFEnabled] = buildParameter(z.boolean());
export type ParameterHRFEnabled = z.infer<typeof zParameterHRFEnabled>;
// #endregion

// #region SDXL Refiner Positive Aesthetic Score
export const [zParameterSDXLRefinerPositiveAestheticScore, isParameterSDXLRefinerPositiveAestheticScore] =
  buildParameter(z.number().min(1).max(10));
export type ParameterSDXLRefinerPositiveAestheticScore = z.infer<typeof zParameterSDXLRefinerPositiveAestheticScore>;
// #endregion

// #region SDXL Refiner Negative Aesthetic Score
export const [zParameterSDXLRefinerNegativeAestheticScore, isParameterSDXLRefinerNegativeAestheticScore] =
  buildParameter(zParameterSDXLRefinerPositiveAestheticScore);
export type ParameterSDXLRefinerNegativeAestheticScore = z.infer<typeof zParameterSDXLRefinerNegativeAestheticScore>;
// #endregion

// #region SDXL Refiner Start
export const [zParameterSDXLRefinerStart, isParameterSDXLRefinerStart] = buildParameter(z.number().min(0).max(1));
export type ParameterSDXLRefinerStart = z.infer<typeof zParameterSDXLRefinerStart>;
// #endregion

// #region Mask Blur Method
const zParameterMaskBlurMethod = z.enum(['box', 'gaussian']);
export type ParameterMaskBlurMethod = z.infer<typeof zParameterMaskBlurMethod>;
// #endregion

// #region Canvas Coherence Mode
export const [zParameterCanvasCoherenceMode, isParameterCanvasCoherenceMode] = buildParameter(
  z.enum(['Gaussian Blur', 'Box Blur', 'Staged'])
);
export type ParameterCanvasCoherenceMode = z.infer<typeof zParameterCanvasCoherenceMode>;
// #endregion

// #region LoRA weight
export const [zLoRAWeight, isParameterLoRAWeight] = buildParameter(z.number());
export type ParameterLoRAWeight = z.infer<typeof zLoRAWeight>;
// #endregion
