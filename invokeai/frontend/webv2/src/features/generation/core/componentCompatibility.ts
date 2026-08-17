import type { ComponentModelConfig, GenerateModelConfig, VaeModelConfig } from './types';

export type GenerateComponentCandidate = {
  base: string;
  format?: string;
  key?: string;
  submodels?: Record<string, unknown> | null;
  type: string;
  variant?: unknown;
};

export type GenerateComponentFilter = (model: GenerateComponentCandidate) => boolean;

export const KLEIN_TO_QWEN3_VARIANT: Record<string, string> = {
  klein_4b: 'qwen3_4b',
  klein_4b_base: 'qwen3_4b',
  klein_9b: 'qwen3_8b',
  klein_9b_base: 'qwen3_8b',
};

const SDNQ_PIPELINE_COMPONENTS = ['transformer', 'vae', 'text_encoder', 'tokenizer'] as const;
const SDNQ_FLUX1_COMPONENTS = [...SDNQ_PIPELINE_COMPONENTS, 'text_encoder_2', 'tokenizer_2'] as const;

const hasSubmodels = (model: GenerateComponentCandidate, required: readonly string[]): boolean => {
  if (model.format !== 'sdnq_quantized' || !model.submodels) {
    return false;
  }

  return required.every((submodel) => Boolean(model.submodels?.[submodel]));
};

export const isSelfContainedSDNQPipeline = (model: GenerateComponentCandidate): boolean =>
  hasSubmodels(model, SDNQ_PIPELINE_COMPONENTS);

export const isSelfContainedSDNQFlux1Pipeline = (model: GenerateComponentCandidate): boolean =>
  hasSubmodels(model, SDNQ_FLUX1_COMPONENTS);

export const isBundledMainForBase =
  (base: string): GenerateComponentFilter =>
  (model) => {
    if (model.type !== 'main' || model.base !== base) {
      return false;
    }
    if (model.format === 'diffusers') {
      return true;
    }
    if (base === 'flux') {
      return isSelfContainedSDNQFlux1Pipeline(model);
    }
    return (base === 'flux2' || base === 'z-image') && isSelfContainedSDNQPipeline(model);
  };

export const getCompatibleSelectedComponentKey = (
  value: ComponentModelConfig | null,
  filter?: GenerateComponentFilter
): string | null => (value && (!filter || filter(value)) ? value.key : null);

export const isDiffusersMainForBase =
  (base: string): GenerateComponentFilter =>
  (model) =>
    model.type === 'main' && model.base === base && model.format === 'diffusers';

export const isVaeForBases =
  (bases: readonly string[]): GenerateComponentFilter =>
  (model) =>
    model.type === 'vae' && bases.length > 0 && bases.includes(model.base);

export const isClipVariant =
  (variant: string): GenerateComponentFilter =>
  (model) =>
    model.type === 'clip_embed' && model.variant === variant;

export const isAnimaQwen3Encoder: GenerateComponentFilter = (model) =>
  model.type === 'qwen3_encoder' && model.variant === 'qwen3_06b';

export const isNonAnimaQwen3Encoder: GenerateComponentFilter = (model) =>
  model.type === 'qwen3_encoder' && model.variant !== 'qwen3_06b';

export const isFlux2MistralEncoder: GenerateComponentFilter = (model) => model.type === 'mistral_encoder';

export const isFlux2Qwen3EncoderForModel = (selectedModel: GenerateModelConfig): GenerateComponentFilter => {
  if (selectedModel.variant === 'dev') {
    return () => false;
  }

  const requiredVariant =
    typeof selectedModel.variant === 'string' ? KLEIN_TO_QWEN3_VARIANT[selectedModel.variant] : null;

  return (model) => {
    if (!isNonAnimaQwen3Encoder(model)) {
      return false;
    }

    return requiredVariant ? model.variant === requiredVariant : true;
  };
};

export const isFlux2DiffusersSourceForModel = (selectedModel: GenerateModelConfig): GenerateComponentFilter => {
  const selectedVariant = typeof selectedModel.variant === 'string' ? selectedModel.variant : null;
  const requiredVariant = selectedVariant ? KLEIN_TO_QWEN3_VARIANT[selectedVariant] : null;

  return (model) => {
    if (!isBundledMainForBase('flux2')(model)) {
      return false;
    }

    const sourceVariant = typeof model.variant === 'string' ? model.variant : null;

    if (selectedVariant === 'dev') {
      return sourceVariant === 'dev';
    }

    if (!requiredVariant) {
      return true;
    }

    return sourceVariant ? KLEIN_TO_QWEN3_VARIANT[sourceVariant] === requiredVariant : false;
  };
};

export const isCompatibleDiffusersComponentSourceForModel = (
  selectedModel: GenerateModelConfig,
  source: GenerateComponentCandidate
): boolean => {
  if (selectedModel.type === 'external_image_generator') {
    return false;
  }

  if (selectedModel.base === 'flux2') {
    return isFlux2DiffusersSourceForModel(selectedModel)(source);
  }

  return isBundledMainForBase(selectedModel.base)(source);
};

export const getCompatibleDiffusersComponentSource = <T extends GenerateComponentCandidate>(
  selectedModel: GenerateModelConfig,
  source: T | null | undefined
): T | undefined =>
  source && isCompatibleDiffusersComponentSourceForModel(selectedModel, source) ? source : undefined;

export const isAnimaVae = isVaeForBases(['anima', 'qwen-image', 'flux']);

/**
 * Krea-2 decodes with the Qwen-Image VAE (16-channel), which is why its graph reuses
 * `qwen_image_l2i`. `anima` is accepted alongside `qwen-image` because the same physical VAE
 * is registered under either base depending on which family it was installed for — matching
 * `krea2_model_loader`'s own `ui_model_base=[QwenImage, Anima]`. Restricting this to
 * `qwen-image` hides a working VAE that the backend would have accepted.
 */
export const isKrea2Vae = isVaeForBases(['qwen-image', 'anima']);

export const isVaeCompatibleWithGenerateModel = (model: GenerateModelConfig, vae: VaeModelConfig): boolean => {
  if (model.type === 'external_image_generator') {
    return false;
  }

  switch (model.base) {
    case 'anima':
      return isAnimaVae(vae);
    case 'z-image':
      return isVaeForBases(['flux'])(vae);
    case 'qwen-image':
      return isVaeForBases(['qwen-image'])(vae);
    case 'krea-2':
      return isKrea2Vae(vae);
    case 'flux2':
      return isVaeForBases(['flux2'])(vae);
    default:
      return vae.type === 'vae' && vae.base === model.base;
  }
};
