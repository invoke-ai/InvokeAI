import { z } from 'zod';

// #region Field data schemas
export const zImageField = z.object({
  image_name: z.string().trim().min(1),
});

export const zBoardField = z.object({
  board_id: z.string().trim().min(1),
});

export const zColorField = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
  a: z.number().int().min(0).max(255),
});

export const zSchedulerField = z.enum([
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
  'lcm',
  'TCD',
]);
// #endregion

// #region Model-related schemas
const zBaseModel = z.enum(['any', 'sd-1', 'sd-2', 'sdxl', 'sdxl-refiner']);
const zModelName = z.string().min(3);
export const zModelIdentifier = z.object({
  model_name: zModelName,
  base_model: zBaseModel,
});
export type ModelIdentifier = z.infer<typeof zModelIdentifier>;

export const zMainModelField = z.object({
  model_name: zModelName,
  base_model: zBaseModel,
  model_type: z.literal('main'),
});

export const zVAEModelField = zModelIdentifier;
export const zLoRAModelField = zModelIdentifier;
export const zControlNetModelField = zModelIdentifier;
export const zIPAdapterModelField = zModelIdentifier;
export const zT2IAdapterModelField = zModelIdentifier;
// #endregion
