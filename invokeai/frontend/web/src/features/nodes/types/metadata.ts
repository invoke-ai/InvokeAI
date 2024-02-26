import { z } from 'zod';

import { zControlField, zIPAdapterField, zModelFieldBase, zT2IAdapterField } from './common';

export const zLoRAWeight = z.number().nullish();
// #region Metadata-optimized versions of schemas
// TODO: It's possible that `deepPartial` will be deprecated:
// - https://github.com/colinhacks/zod/issues/2106
// - https://github.com/colinhacks/zod/issues/2854
export const zLoRAMetadataItem = z.object({
  lora: zModelFieldBase.deepPartial(),
  weight: zLoRAWeight,
});
const zControlNetMetadataItem = zControlField.merge(z.object({ control_model: z.unknown() })).deepPartial();
const zIPAdapterMetadataItem = zIPAdapterField.merge(z.object({ ip_adapter_model: z.unknown() })).deepPartial();
const zT2IAdapterMetadataItem = zT2IAdapterField.merge(z.object({ t2i_adapter_model: z.unknown() })).deepPartial();
const zSDXLRefinerModelMetadataItem = z.unknown();
const zModelMetadataItem = z.unknown();
const zVAEModelMetadataItem = z.unknown();
export type LoRAMetadataItem = z.infer<typeof zLoRAMetadataItem>;
export type ControlNetMetadataItem = z.infer<typeof zControlNetMetadataItem>;
export type IPAdapterMetadataItem = z.infer<typeof zIPAdapterMetadataItem>;
export type T2IAdapterMetadataItem = z.infer<typeof zT2IAdapterMetadataItem>;
export type SDXLRefinerModelMetadataItem = z.infer<typeof zSDXLRefinerModelMetadataItem>;
export type ModelMetadataItem = z.infer<typeof zModelMetadataItem>;
export type VAEModelMetadataItem = z.infer<typeof zVAEModelMetadataItem>;
// #endregion

