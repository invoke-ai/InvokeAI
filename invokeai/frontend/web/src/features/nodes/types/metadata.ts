import { z } from 'zod';

import {
  zControlField,
  zIPAdapterField,
  zLoRAModelField,
  zMainModelField,
  zSDXLRefinerModelField,
  zT2IAdapterField,
  zVAEModelField,
} from './common';

// #region Metadata-optimized versions of schemas
// TODO: It's possible that `deepPartial` will be deprecated:
// - https://github.com/colinhacks/zod/issues/2106
// - https://github.com/colinhacks/zod/issues/2854
export const zLoRAMetadataItem = z.object({
  lora: zLoRAModelField.deepPartial(),
  weight: z.number(),
});
const zControlNetMetadataItem = zControlField.deepPartial();
const zIPAdapterMetadataItem = zIPAdapterField.deepPartial();
const zT2IAdapterMetadataItem = zT2IAdapterField.deepPartial();
const zSDXLRefinerModelMetadataItem = zSDXLRefinerModelField.deepPartial();
const zModelMetadataItem = zMainModelField.deepPartial();
const zVAEModelMetadataItem = zVAEModelField.deepPartial();
export type LoRAMetadataItem = z.infer<typeof zLoRAMetadataItem>;
export type ControlNetMetadataItem = z.infer<typeof zControlNetMetadataItem>;
export type IPAdapterMetadataItem = z.infer<typeof zIPAdapterMetadataItem>;
export type T2IAdapterMetadataItem = z.infer<typeof zT2IAdapterMetadataItem>;
export type SDXLRefinerModelMetadataItem = z.infer<typeof zSDXLRefinerModelMetadataItem>;
export type ModelMetadataItem = z.infer<typeof zModelMetadataItem>;
export type VAEModelMetadataItem = z.infer<typeof zVAEModelMetadataItem>;
// #endregion

// #region CoreMetadata
export const zCoreMetadata = z
  .object({
    app_version: z.string().nullish().catch(null),
    generation_mode: z.string().nullish().catch(null),
    created_by: z.string().nullish().catch(null),
    positive_prompt: z.string().nullish().catch(null),
    negative_prompt: z.string().nullish().catch(null),
    width: z.number().int().nullish().catch(null),
    height: z.number().int().nullish().catch(null),
    seed: z.number().int().nullish().catch(null),
    rand_device: z.string().nullish().catch(null),
    cfg_scale: z.number().nullish().catch(null),
    cfg_rescale_multiplier: z.number().nullish().catch(null),
    steps: z.number().int().nullish().catch(null),
    scheduler: z.string().nullish().catch(null),
    clip_skip: z.number().int().nullish().catch(null),
    model: zModelMetadataItem.nullish().catch(null),
    controlnets: z.array(zControlNetMetadataItem).nullish().catch(null),
    ipAdapters: z.array(zIPAdapterMetadataItem).nullish().catch(null),
    t2iAdapters: z.array(zT2IAdapterMetadataItem).nullish().catch(null),
    loras: z.array(zLoRAMetadataItem).nullish().catch(null),
    vae: zVAEModelMetadataItem.nullish().catch(null),
    strength: z.number().nullish().catch(null),
    hrf_enabled: z.boolean().nullish().catch(null),
    hrf_strength: z.number().nullish().catch(null),
    hrf_method: z.string().nullish().catch(null),
    init_image: z.string().nullish().catch(null),
    positive_style_prompt: z.string().nullish().catch(null),
    negative_style_prompt: z.string().nullish().catch(null),
    refiner_model: zSDXLRefinerModelMetadataItem.nullish().catch(null),
    refiner_cfg_scale: z.number().nullish().catch(null),
    refiner_steps: z.number().int().nullish().catch(null),
    refiner_scheduler: z.string().nullish().catch(null),
    refiner_positive_aesthetic_score: z.number().nullish().catch(null),
    refiner_negative_aesthetic_score: z.number().nullish().catch(null),
    refiner_start: z.number().nullish().catch(null),
  })
  .passthrough();
export type CoreMetadata = z.infer<typeof zCoreMetadata>;

// #endregion
