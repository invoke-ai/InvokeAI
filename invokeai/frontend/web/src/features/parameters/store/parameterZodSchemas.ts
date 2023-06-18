import { NUMPY_RAND_MAX, SCHEDULERS } from 'app/constants';
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
export const zScheduler = z.enum(SCHEDULERS);
/**
 * Type alias for scheduler parameter, inferred from its zod schema
 */
export type SchedulerParam = z.infer<typeof zScheduler>;
/**
 * Validates/type-guards a value as a scheduler parameter
 */
export const isValidScheduler = (val: unknown): val is SchedulerParam =>
  zScheduler.safeParse(val).success;

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

/**
 * Zod schema for model parameter
 * TODO: Make this a dynamically generated enum?
 */
export const zModel = z.string();
/**
 * Type alias for model parameter, inferred from its zod schema
 */
export type ModelParam = z.infer<typeof zModel>;
/**
 * Validates/type-guards a value as a model parameter
 */
export const isValidModel = (val: unknown): val is ModelParam =>
  zModel.safeParse(val).success;

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
