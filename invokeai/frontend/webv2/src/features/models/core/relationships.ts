import type { ModelBase, ModelConfig, ModelTaxonomyType } from './types';

/**
 * Compatibility rules for the bidirectional "related models" links. A link is
 * a curation hint ("this LoRA pairs well with this checkpoint"), so the rule
 * only permits pairs that can actually appear in one pipeline.
 */

/** The two fields the compatibility rule reads. */
export type RelatableModel = Pick<ModelConfig, 'base' | 'type'>;

/**
 * Bases that carry no architecture meaning. `any` is the backend's null value
 * for models with no base association (encoders, CLIP, Spandrel, ...), NOT a
 * universal-compatibility wildcard; `external` and `unknown` say nothing
 * about what a model pairs with.
 */
export const NULL_BASES: ReadonlySet<string> = new Set(['any', 'external', 'unknown']);

/**
 * Curated allowances: `any`-based helper types mapped to the concrete bases
 * whose pipelines actually consume them, per the backend model loaders.
 */
export const NULL_BASE_ALLOWANCES: Readonly<Partial<Record<ModelTaxonomyType, ReadonlySet<ModelBase>>>> = {
  /** CLIP text encoders (flux_model_loader, sd3_model_loader). */
  clip_embed: new Set(['flux', 'sd-3']),
  /** IP-adapter image encoders (ip_adapter config bases). */
  clip_vision: new Set(['flux', 'sd-1', 'sd-2', 'sdxl']),
  /** PiD decoder captioning (pid_decoder config bases; prototype). */
  gemma2_encoder: new Set(['flux', 'flux2', 'qwen-image', 'sd-3', 'sdxl']),
  /** FLUX.2 dev text encoder (flux2_dev_model_loader). */
  mistral_encoder: new Set(['flux2']),
  qwen3_encoder: new Set(['anima', 'flux2', 'z-image']),
  qwen3_vl_encoder: new Set(['krea-2']),
  qwen_vl_encoder: new Set(['qwen-image']),
  /** FLUX Redux image encoder (flux_redux). */
  siglip: new Set(['flux']),
  t5_encoder: new Set(['flux', 'sd-3']),
  wan_t5_encoder: new Set(['wan']),
};

/**
 * Types offered when linking related models. The concrete-based helper types,
 * plus every `any`-based type with a curated allowance.
 */
export const LINKABLE_TYPES: readonly ModelTaxonomyType[] = [
  'main',
  'lora',
  'embedding',
  'vae',
  'controlnet',
  't2i_adapter',
  'ip_adapter',
  ...(Object.keys(NULL_BASE_ALLOWANCES) as ModelTaxonomyType[]),
];

const LINKABLE_TYPE_SET: ReadonlySet<ModelTaxonomyType> = new Set(LINKABLE_TYPES);

export const isLinkableType = (type: ModelTaxonomyType): boolean => LINKABLE_TYPE_SET.has(type);

const isAllowedHelperFor = (helper: RelatableModel, host: RelatableModel): boolean =>
  helper.base === 'any' &&
  !NULL_BASES.has(String(host.base)) &&
  (NULL_BASE_ALLOWANCES[helper.type]?.has(host.base) ?? false);

/**
 * Symmetric base-compatibility for related-model links: concrete bases must
 * match exactly; a null base never wildcards — an `any`-based model links only
 * to concrete bases its type has a curated allowance for.
 */
export const isBaseCompatible = (a: RelatableModel, b: RelatableModel): boolean => {
  if (!NULL_BASES.has(String(a.base)) && !NULL_BASES.has(String(b.base))) {
    return a.base === b.base;
  }

  return isAllowedHelperFor(a, b) || isAllowedHelperFor(b, a);
};
