import type { ComponentModelConfig, GenerateModelConfig, VaeModelConfig } from './types';

import { getArchitectureCapabilityRow } from './architectureCapabilities';

export type GenerateComponentCandidate = {
  base: string;
  /** Which Ideogram 4 transformer branch a single file holds. Only `ideogram-4` mains carry it. */
  branch?: unknown;
  format?: string;
  key?: string;
  /** VAE latent width. Only `wan` ships more than one, and its two are different decoders. */
  latent_channels?: number | null;
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

export const isClipVariant =
  (variant: string): GenerateComponentFilter =>
  (model) =>
    model.type === 'clip_embed' && model.variant === variant;

export const isAnimaQwen3Encoder: GenerateComponentFilter = (model) =>
  model.type === 'qwen3_encoder' && model.variant === 'qwen3_06b';

export const isNonAnimaQwen3Encoder: GenerateComponentFilter = (model) =>
  model.type === 'qwen3_encoder' && model.variant !== 'qwen3_06b';

/** Ministral and Mistral encoders require distinct variants. */
const MINISTRAL_3B_VARIANT = 'ministral3_3b';

export const isFlux2MistralEncoder: GenerateComponentFilter = (model) =>
  model.type === 'mistral_encoder' && model.variant !== MINISTRAL_3B_VARIANT;

export const isErnieImageMistralEncoder: GenerateComponentFilter = (model) =>
  model.type === 'mistral_encoder' && model.variant === MINISTRAL_3B_VARIANT;

/** Krea's 4B encoder is incompatible with Ideogram's 8B encoder. */
export const isKrea2Qwen3VlEncoder: GenerateComponentFilter = (model) =>
  model.type === 'qwen3_vl_encoder' && model.variant === 'qwen3_vl_4b';

export const isIdeogram4Qwen3VlEncoder: GenerateComponentFilter = (model) =>
  model.type === 'qwen3_vl_encoder' && model.variant === 'qwen3_vl_8b';

/** Exclude only single-file unconditional branches; Diffusers bundles both branches. */
export const isIdeogram4UnconditionalBranch: GenerateComponentFilter = (model) =>
  model.type === 'main' &&
  model.base === 'ideogram-4' &&
  model.format === 'checkpoint' &&
  model.branch === 'unconditional';

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

/** Backend rows own VAE compatibility; missing rows fail closed. */
const acceptsVae = (base: string, model: GenerateComponentCandidate, variant?: unknown): boolean => {
  if (model.type !== 'vae') {
    return false;
  }

  // With the variant: Wan A14B and TI2V-5B decode with different VAEs, and only the variant row says so.
  const row = getArchitectureCapabilityRow(base, variant);

  if (!row) {
    return false;
  }

  // A null `vae` block is the backend saying "its own base, no constraints" — what `accepts_vae`
  // answers for an architecture that declares no facet. It is a declaration, not a missing one.
  if (!row.vae) {
    return model.base === base;
  }

  return row.vae.accepted.some(
    (accepted) =>
      accepted.base === model.base &&
      (accepted.latent_channels === null || model.latent_channels === accepted.latent_channels)
  );
};

export const isVaeAcceptedByBase =
  (base: string, variant?: unknown): GenerateComponentFilter =>
  (model) =>
    acceptsVae(base, model, variant);

/** Picker, validator, and compiler share the served VAE rule. */
export const isVaeCompatibleWithGenerateModel = (model: GenerateModelConfig, vae: VaeModelConfig): boolean => {
  if (model.type === 'external_image_generator') {
    return false;
  }

  return acceptsVae(model.base, vae, model.variant);
};
