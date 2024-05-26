import { z } from 'zod';

import type { ModelIdentifier as ModelIdentifierV2 } from './v2/common';
import { zModelIdentifier as zModelIdentifierV2 } from './v2/common';

// #region Field data schemas
export const zImageField = z.object({
  image_name: z.string().trim().min(1),
});
export type ImageField = z.infer<typeof zImageField>;

export const zBoardField = z.object({
  board_id: z.string().trim().min(1),
});
export type BoardField = z.infer<typeof zBoardField>;

export const zColorField = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
  a: z.number().int().min(0).max(255),
});
export type ColorField = z.infer<typeof zColorField>;

export const zClassification = z.enum(['stable', 'beta', 'prototype']);
export type Classification = z.infer<typeof zClassification>;

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
  'tcd',
]);
export type SchedulerField = z.infer<typeof zSchedulerField>;
// #endregion

// #region Model-related schemas
const zBaseModel = z.enum(['any', 'sd-1', 'sd-2', 'sdxl', 'sdxl-refiner']);
const zModelType = z.enum([
  'main',
  'vae',
  'lora',
  'controlnet',
  't2i_adapter',
  'ip_adapter',
  'embedding',
  'onnx',
  'clip_vision',
]);
const zSubModelType = z.enum([
  'unet',
  'text_encoder',
  'text_encoder_2',
  'tokenizer',
  'tokenizer_2',
  'vae',
  'vae_decoder',
  'vae_encoder',
  'scheduler',
  'safety_checker',
]);
export const zModelIdentifierField = z.object({
  key: z.string().min(1),
  hash: z.string().min(1),
  name: z.string().min(1),
  base: zBaseModel,
  type: zModelType,
  submodel_type: zSubModelType.nullish(),
});
export const isModelIdentifier = (field: unknown): field is ModelIdentifierField =>
  zModelIdentifierField.safeParse(field).success;
export const isModelIdentifierV2 = (field: unknown): field is ModelIdentifierV2 =>
  zModelIdentifierV2.safeParse(field).success;
export type ModelIdentifierField = z.infer<typeof zModelIdentifierField>;
// #endregion

// #region Control Adapters
export const zControlField = z.object({
  image: zImageField,
  control_model: zModelIdentifierField,
  control_weight: z.union([z.number(), z.array(z.number())]).optional(),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
  control_mode: z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']).optional(),
  resize_mode: z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']).optional(),
});
export type ControlField = z.infer<typeof zControlField>;

export const zIPAdapterField = z.object({
  image: zImageField,
  ip_adapter_model: zModelIdentifierField,
  weight: z.number(),
  method: z.enum(['full', 'style', 'composition']),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
});
export type IPAdapterField = z.infer<typeof zIPAdapterField>;

export const zT2IAdapterField = z.object({
  image: zImageField,
  t2i_adapter_model: zModelIdentifierField,
  weight: z.union([z.number(), z.array(z.number())]).optional(),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
  resize_mode: z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']).optional(),
});
export type T2IAdapterField = z.infer<typeof zT2IAdapterField>;
// #endregion

// #region ProgressImage
export const zProgressImage = z.object({
  dataURL: z.string(),
  width: z.number().int(),
  height: z.number().int(),
});
export type ProgressImage = z.infer<typeof zProgressImage>;
// #endregion

// #region ImageOutput
const zImageOutput = z.object({
  image: zImageField,
  width: z.number().int().gt(0),
  height: z.number().int().gt(0),
  type: z.literal('image_output'),
});
export type ImageOutput = z.infer<typeof zImageOutput>;
export const isImageOutput = (output: unknown): output is ImageOutput => zImageOutput.safeParse(output).success;
// #endregion
