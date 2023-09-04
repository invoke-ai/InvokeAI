import { NUMPY_RAND_MAX } from 'app/constants';
import { z } from 'zod';

/**
 * These zod schemas should match the pydantic node schemas.
 *
 * Parameters only need schemas if we want to recall them from metadata.
 *
 * Each parameter needs:
 * - a zod schema
 * - a type alias, inferred from the zod schema
 * - a combo validation/type guard function, which returns true if the value is valid
 */

/**
 * Zod schema for positive prompt parameter
 */
export const zPositivePrompt = z.string();
/**
 * Type alias for positive prompt parameter, inferred from its zod schema
 */
export type PositivePromptParam = z.infer<typeof zPositivePrompt>;
/**
 * Validates/type-guards a value as a positive prompt parameter
 */
export const isValidPositivePrompt = (
  val: unknown
): val is PositivePromptParam => zPositivePrompt.safeParse(val).success;

/**
 * Zod schema for negative prompt parameter
 */
export const zNegativePrompt = z.string();
/**
 * Type alias for negative prompt parameter, inferred from its zod schema
 */
export type NegativePromptParam = z.infer<typeof zNegativePrompt>;
/**
 * Validates/type-guards a value as a negative prompt parameter
 */
export const isValidNegativePrompt = (
  val: unknown
): val is NegativePromptParam => zNegativePrompt.safeParse(val).success;

/**
 * Zod schema for SDXL positive style prompt parameter
 */
export const zPositiveStylePromptSDXL = z.string();
/**
 * Type alias for SDXL positive style prompt parameter, inferred from its zod schema
 */
export type PositiveStylePromptSDXLParam = z.infer<
  typeof zPositiveStylePromptSDXL
>;
/**
 * Validates/type-guards a value as a SDXL positive style prompt parameter
 */
export const isValidSDXLPositiveStylePrompt = (
  val: unknown
): val is PositiveStylePromptSDXLParam =>
  zPositiveStylePromptSDXL.safeParse(val).success;

/**
 * Zod schema for SDXL negative style prompt parameter
 */
export const zNegativeStylePromptSDXL = z.string();
/**
 * Type alias for SDXL negative style prompt parameter, inferred from its zod schema
 */
export type NegativeStylePromptSDXLParam = z.infer<
  typeof zNegativeStylePromptSDXL
>;
/**
 * Validates/type-guards a value as a SDXL negative style prompt parameter
 */
export const isValidSDXLNegativeStylePrompt = (
  val: unknown
): val is NegativeStylePromptSDXLParam =>
  zNegativeStylePromptSDXL.safeParse(val).success;

/**
 * Zod schema for steps parameter
 */
export const zSteps = z.number().int().min(1);
/**
 * Type alias for steps parameter, inferred from its zod schema
 */
export type StepsParam = z.infer<typeof zSteps>;
/**
 * Validates/type-guards a value as a steps parameter
 */
export const isValidSteps = (val: unknown): val is StepsParam =>
  zSteps.safeParse(val).success;

/**
 * Zod schema for CFG scale parameter
 */
export const zCfgScale = z.number().min(1);
/**
 * Type alias for CFG scale parameter, inferred from its zod schema
 */
export type CfgScaleParam = z.infer<typeof zCfgScale>;
/**
 * Validates/type-guards a value as a CFG scale parameter
 */
export const isValidCfgScale = (val: unknown): val is CfgScaleParam =>
  zCfgScale.safeParse(val).success;

/**
 * Zod schema for scheduler parameter
 */
export const zScheduler = z.enum([
  'euler',
  'deis',
  'ddim',
  'ddpm',
  'dpmpp_2s',
  'dpmpp_2m',
  'dpmpp_2m_sde',
  'dpmpp_sde',
  'heun',
  'kdpm_2',
  'lms',
  'pndm',
  'unipc',
  'euler_k',
  'dpmpp_2s_k',
  'dpmpp_2m_k',
  'dpmpp_2m_sde_k',
  'dpmpp_sde_k',
  'heun_k',
  'lms_k',
  'euler_a',
  'kdpm_2_a',
]);
/**
 * Type alias for scheduler parameter, inferred from its zod schema
 */
export type SchedulerParam = z.infer<typeof zScheduler>;
/**
 * Validates/type-guards a value as a scheduler parameter
 */
export const isValidScheduler = (val: unknown): val is SchedulerParam =>
  zScheduler.safeParse(val).success;

export const SCHEDULER_LABEL_MAP: Record<SchedulerParam, string> = {
  euler: 'Euler',
  deis: 'DEIS',
  ddim: 'DDIM',
  ddpm: 'DDPM',
  dpmpp_sde: 'DPM++ SDE',
  dpmpp_2s: 'DPM++ 2S',
  dpmpp_2m: 'DPM++ 2M',
  dpmpp_2m_sde: 'DPM++ 2M SDE',
  heun: 'Heun',
  kdpm_2: 'KDPM 2',
  lms: 'LMS',
  pndm: 'PNDM',
  unipc: 'UniPC',
  euler_k: 'Euler Karras',
  dpmpp_sde_k: 'DPM++ SDE Karras',
  dpmpp_2s_k: 'DPM++ 2S Karras',
  dpmpp_2m_k: 'DPM++ 2M Karras',
  dpmpp_2m_sde_k: 'DPM++ 2M SDE Karras',
  heun_k: 'Heun Karras',
  lms_k: 'LMS Karras',
  euler_a: 'Euler Ancestral',
  kdpm_2_a: 'KDPM 2 Ancestral',
};

/**
 * Zod schema for seed parameter
 */
export const zSeed = z.number().int().min(0).max(NUMPY_RAND_MAX);
/**
 * Type alias for seed parameter, inferred from its zod schema
 */
export type SeedParam = z.infer<typeof zSeed>;
/**
 * Validates/type-guards a value as a seed parameter
 */
export const isValidSeed = (val: unknown): val is SeedParam =>
  zSeed.safeParse(val).success;

/**
 * Zod schema for width parameter
 */
export const zWidth = z.number().multipleOf(8).min(64);
/**
 * Type alias for width parameter, inferred from its zod schema
 */
export type WidthParam = z.infer<typeof zWidth>;
/**
 * Validates/type-guards a value as a width parameter
 */
export const isValidWidth = (val: unknown): val is WidthParam =>
  zWidth.safeParse(val).success;

/**
 * Zod schema for height parameter
 */
export const zHeight = z.number().multipleOf(8).min(64);
/**
 * Type alias for height parameter, inferred from its zod schema
 */
export type HeightParam = z.infer<typeof zHeight>;
/**
 * Validates/type-guards a value as a height parameter
 */
export const isValidHeight = (val: unknown): val is HeightParam =>
  zHeight.safeParse(val).success;

export const zBaseModel = z.enum(['sd-1', 'sd-2', 'sdxl', 'sdxl-refiner']);

export type BaseModelParam = z.infer<typeof zBaseModel>;

/**
 * Zod schema for main model parameter
 * TODO: Make this a dynamically generated enum?
 */
export const zMainModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
  model_type: z.literal('main'),
});
/**
 * Type alias for main model parameter, inferred from its zod schema
 */
export type MainModelParam = z.infer<typeof zMainModel>;
/**
 * Validates/type-guards a value as a main model parameter
 */
export const isValidMainModel = (val: unknown): val is MainModelParam =>
  zMainModel.safeParse(val).success;

/**
 * Zod schema for SDXL refiner model parameter
 * TODO: Make this a dynamically generated enum?
 */
export const zSDXLRefinerModel = z.object({
  model_name: z.string().min(1),
  base_model: z.literal('sdxl-refiner'),
  model_type: z.literal('main'),
});
/**
 * Type alias for SDXL refiner model parameter, inferred from its zod schema
 */
export type SDXLRefinerModelParam = z.infer<typeof zSDXLRefinerModel>;
/**
 * Validates/type-guards a value as a SDXL refiner model parameter
 */
export const isValidSDXLRefinerModel = (
  val: unknown
): val is SDXLRefinerModelParam => zSDXLRefinerModel.safeParse(val).success;

/**
 * Zod schema for Onnx model parameter
 * TODO: Make this a dynamically generated enum?
 */
export const zOnnxModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
  model_type: z.literal('onnx'),
});
/**
 * Type alias for Onnx model parameter, inferred from its zod schema
 */
export type OnnxModelParam = z.infer<typeof zOnnxModel>;
/**
 * Validates/type-guards a value as a Onnx model parameter
 */
export const isValidOnnxModel = (val: unknown): val is OnnxModelParam =>
  zOnnxModel.safeParse(val).success;

export const zMainOrOnnxModel = z.union([zMainModel, zOnnxModel]);

/**
 * Zod schema for VAE parameter
 */
export const zVaeModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
});
/**
 * Type alias for model parameter, inferred from its zod schema
 */
export type VaeModelParam = z.infer<typeof zVaeModel>;
/**
 * Validates/type-guards a value as a model parameter
 */
export const isValidVaeModel = (val: unknown): val is VaeModelParam =>
  zVaeModel.safeParse(val).success;
/**
 * Zod schema for LoRA
 */
export const zLoRAModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
});
/**
 * Type alias for model parameter, inferred from its zod schema
 */
export type LoRAModelParam = z.infer<typeof zLoRAModel>;
/**
 * Validates/type-guards a value as a model parameter
 */
export const isValidLoRAModel = (val: unknown): val is LoRAModelParam =>
  zLoRAModel.safeParse(val).success;
/**
 * Zod schema for ControlNet models
 */
export const zControlNetModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
});
/**
 * Type alias for model parameter, inferred from its zod schema
 */
export type ControlNetModelParam = z.infer<typeof zLoRAModel>;
/**
 * Validates/type-guards a value as a model parameter
 */
export const isValidControlNetModel = (
  val: unknown
): val is ControlNetModelParam => zControlNetModel.safeParse(val).success;

/**
 * Zod schema for l2l strength parameter
 */
export const zStrength = z.number().min(0).max(1);
/**
 * Type alias for l2l strength parameter, inferred from its zod schema
 */
export type StrengthParam = z.infer<typeof zStrength>;
/**
 * Validates/type-guards a value as a l2l strength parameter
 */
export const isValidStrength = (val: unknown): val is StrengthParam =>
  zStrength.safeParse(val).success;

/**
 * Zod schema for a precision parameter
 */
export const zPrecision = z.enum(['fp16', 'fp32']);
/**
 * Type alias for precision parameter, inferred from its zod schema
 */
export type PrecisionParam = z.infer<typeof zPrecision>;
/**
 * Validates/type-guards a value as a precision parameter
 */
export const isValidPrecision = (val: unknown): val is PrecisionParam =>
  zPrecision.safeParse(val).success;

/**
 * Zod schema for SDXL refiner positive aesthetic score parameter
 */
export const zSDXLRefinerPositiveAestheticScore = z.number().min(1).max(10);
/**
 * Type alias for SDXL refiner aesthetic positive score parameter, inferred from its zod schema
 */
export type SDXLRefinerPositiveAestheticScoreParam = z.infer<
  typeof zSDXLRefinerPositiveAestheticScore
>;
/**
 * Validates/type-guards a value as a SDXL refiner positive aesthetic score parameter
 */
export const isValidSDXLRefinerPositiveAestheticScore = (
  val: unknown
): val is SDXLRefinerPositiveAestheticScoreParam =>
  zSDXLRefinerPositiveAestheticScore.safeParse(val).success;

/**
 * Zod schema for SDXL refiner negative aesthetic score parameter
 */
export const zSDXLRefinerNegativeAestheticScore = z.number().min(1).max(10);
/**
 * Type alias for SDXL refiner aesthetic negative score parameter, inferred from its zod schema
 */
export type SDXLRefinerNegativeAestheticScoreParam = z.infer<
  typeof zSDXLRefinerNegativeAestheticScore
>;
/**
 * Validates/type-guards a value as a SDXL refiner negative aesthetic score parameter
 */
export const isValidSDXLRefinerNegativeAestheticScore = (
  val: unknown
): val is SDXLRefinerNegativeAestheticScoreParam =>
  zSDXLRefinerNegativeAestheticScore.safeParse(val).success;

/**
 * Zod schema for SDXL start parameter
 */
export const zSDXLRefinerstart = z.number().min(0).max(1);
/**
 * Type alias for SDXL start, inferred from its zod schema
 */
export type SDXLRefinerStartParam = z.infer<typeof zSDXLRefinerstart>;
/**
 * Validates/type-guards a value as a SDXL refiner aesthetic score parameter
 */
export const isValidSDXLRefinerStart = (
  val: unknown
): val is SDXLRefinerStartParam => zSDXLRefinerstart.safeParse(val).success;

/**
 * Zod schema for a mask blur method parameter
 */
export const zMaskBlurMethod = z.enum(['box', 'gaussian']);
/**
 * Type alias for mask blur method parameter, inferred from its zod schema
 */
export type MaskBlurMethodParam = z.infer<typeof zMaskBlurMethod>;
/**
 * Validates/type-guards a value as a mask blur method parameter
 */
export const isValidMaskBlurMethod = (
  val: unknown
): val is MaskBlurMethodParam => zMaskBlurMethod.safeParse(val).success;

/**
 * Zod schema for a Canvas Coherence Mode method parameter
 */
export const zCanvasCoherenceMode = z.enum(['unmasked', 'mask', 'edge']);
/**
 * Type alias for Canvas Coherence Mode parameter, inferred from its zod schema
 */
export type CanvasCoherenceModeParam = z.infer<typeof zCanvasCoherenceMode>;
/**
 * Validates/type-guards a value as a mask blur method parameter
 */
export const isValidCoherenceModeParam = (
  val: unknown
): val is CanvasCoherenceModeParam =>
  zCanvasCoherenceMode.safeParse(val).success;

// /**
//  * Zod schema for BaseModelType
//  */
// export const zBaseModelType = z.enum(['sd-1', 'sd-2']);
// /**
//  * Type alias for base model type, inferred from its zod schema. Should be identical to the type alias from OpenAPI.
//  */
// export type BaseModelType = z.infer<typeof zBaseModelType>;
// /**
//  * Validates/type-guards a value as a base model type
//  */
// export const isValidBaseModelType = (val: unknown): val is BaseModelType =>
//   zBaseModelType.safeParse(val).success;
