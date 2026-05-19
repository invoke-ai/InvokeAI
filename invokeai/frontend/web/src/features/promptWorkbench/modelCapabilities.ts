import type { BaseModelType } from 'features/nodes/types/common';

export type PromptModelCapabilities = {
  supportsAttentionWeights: boolean;
  attentionWeightsLabelKey: string;
};

const ATTENTION_WEIGHT_BASES = new Set<BaseModelType>(['sd-1', 'sd-2', 'sdxl']);

export const getPromptModelCapabilities = (base: BaseModelType | null | undefined): PromptModelCapabilities => {
  const supportsAttentionWeights = !!base && ATTENTION_WEIGHT_BASES.has(base);

  return {
    supportsAttentionWeights,
    attentionWeightsLabelKey: supportsAttentionWeights
      ? 'promptWorkbench.weight.supportedDescription'
      : 'promptWorkbench.weight.literalDescription',
  };
};
