import type { BaseModelType } from 'features/nodes/types/common';

export type PromptModelCapabilities = {
  supportsAttentionWeights: boolean;
  attentionWeightsLabel: string;
};

const ATTENTION_WEIGHT_BASES = new Set<BaseModelType>(['sd-1', 'sd-2', 'sdxl']);

export const getPromptModelCapabilities = (base: BaseModelType | null | undefined): PromptModelCapabilities => {
  const supportsAttentionWeights = !!base && ATTENTION_WEIGHT_BASES.has(base);

  return {
    supportsAttentionWeights,
    attentionWeightsLabel: supportsAttentionWeights
      ? 'Prompt weights are supported for this model.'
      : 'Prompt weight syntax may be treated as literal text by this model.',
  };
};
