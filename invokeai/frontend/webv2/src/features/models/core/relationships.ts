import type { ModelBase, ModelConfig, ModelTaxonomyType } from './types';

/** Related-model links are symmetric curation hints and permit only models usable in one pipeline. */

/** The two fields the compatibility rule reads. */
export type RelatableModel = Pick<ModelConfig, 'base' | 'type'>;

/**
 * any denotes no base association, not universal compatibility; external and unknown convey no architecture
 * compatibility.
 */
export const NULL_BASES: ReadonlySet<string> = new Set(['any', 'external', 'unknown']);

/** Cite consuming backend invocations for each any-based helper allowance; config metadata alone is insufficient. */
export const NULL_BASE_ALLOWANCES: Readonly<Partial<Record<ModelTaxonomyType, ReadonlySet<ModelBase>>>> = {
  /** CLIP text encoders (flux_model_loader, sd3_model_loader). */
  clip_embed: new Set(['flux', 'sd-3']),
  /** IP-adapter image encoders (ip_adapter accepts sd-1/sdxl only; flux_ip_adapter). */
  clip_vision: new Set(['flux', 'sd-1', 'sdxl']),
  /** PiD decode (flux/flux2/qwen_image/z_image_pid_decode; sd-3/sdxl pid_decoder configs; prototype). */
  gemma2_encoder: new Set(['flux', 'flux2', 'qwen-image', 'sd-3', 'sdxl', 'z-image']),
  /** FLUX.2 dev text encoder (flux2_dev_model_loader). */
  mistral_encoder: new Set(['flux2']),
  /** anima_model_loader, flux2_klein_model_loader, z_image_model_loader. */
  qwen3_encoder: new Set(['anima', 'flux2', 'z-image']),
  /** krea2_model_loader. */
  qwen3_vl_encoder: new Set(['krea-2']),
  /** qwen_image_model_loader. */
  qwen_vl_encoder: new Set(['qwen-image']),
  /** FLUX Redux image encoder (flux_redux). */
  siglip: new Set(['flux']),
  /** flux_model_loader, sd3_model_loader. */
  t5_encoder: new Set(['flux', 'sd-3']),
  /** wan_model_loader. */
  wan_t5_encoder: new Set(['wan']),
};

/** Cross-base helper allowances cite consuming backend invocations, like NULL_BASE_ALLOWANCES. */
export const CROSS_BASE_ALLOWANCES: Readonly<
  Partial<Record<ModelTaxonomyType, Readonly<Partial<Record<ModelBase, ReadonlySet<ModelBase>>>>>>
> = {
  /** The 16-channel VAEs are shared across backbones. */
  vae: {
    /** krea2_model_loader (accepts QwenImage and Anima VAEs). */
    anima: new Set(['krea-2']),
    /** z_image_model_loader; flux2_klein_model_loader; anima_model_loader ("A FLUX VAE can also be used"). */
    flux: new Set(['anima', 'flux2', 'z-image']),
    /** krea2_model_loader; anima_model_loader ("Wan 2.1 / QwenImage VAE"). */
    'qwen-image': new Set(['anima', 'krea-2']),
    /** anima_model_loader ("Wan 2.1 / QwenImage VAE"). */
    wan: new Set(['anima']),
  },
  /** z_image_pid_decode reuses the FLUX decoder (assert_pid_decoder_matches_base). */
  pid_decoder: {
    flux: new Set(['z-image']),
  },
};

export const LINKABLE_TYPES: readonly ModelTaxonomyType[] = [
  'main',
  'lora',
  'embedding',
  'vae',
  'controlnet',
  'control_lora',
  't2i_adapter',
  'ip_adapter',
  'flux_redux',
  'pid_decoder',
  ...(Object.keys(NULL_BASE_ALLOWANCES) as ModelTaxonomyType[]),
];

const LINKABLE_TYPE_SET: ReadonlySet<ModelTaxonomyType> = new Set(LINKABLE_TYPES);

export const isLinkableType = (type: ModelTaxonomyType): boolean => LINKABLE_TYPE_SET.has(type);

/** Only concrete bases or explicitly allowed any-based helper types can receive new links. */
export const hasLinkableBase = (model: RelatableModel): boolean =>
  !NULL_BASES.has(String(model.base)) || (model.base === 'any' && NULL_BASE_ALLOWANCES[model.type] !== undefined);

const isAllowedHelperFor = (helper: RelatableModel, host: RelatableModel): boolean =>
  helper.base === 'any' &&
  !NULL_BASES.has(String(host.base)) &&
  (NULL_BASE_ALLOWANCES[helper.type]?.has(host.base) ?? false);

const isAllowedCrossBaseHelperFor = (helper: RelatableModel, host: RelatableModel): boolean =>
  CROSS_BASE_ALLOWANCES[helper.type]?.[helper.base]?.has(host.base) ?? false;

/**
 * Match concrete bases unless a helper allowance permits crossing; any-based models require explicit concrete-host
 * allowances.
 */
export const isBaseCompatible = (a: RelatableModel, b: RelatableModel): boolean => {
  if (!NULL_BASES.has(String(a.base)) && !NULL_BASES.has(String(b.base))) {
    return a.base === b.base || isAllowedCrossBaseHelperFor(a, b) || isAllowedCrossBaseHelperFor(b, a);
  }

  return isAllowedHelperFor(a, b) || isAllowedHelperFor(b, a);
};

/** Single-slot types cannot meaningfully link to their own type; stackable adapters, LoRAs, and embeddings can. */
const SINGLETON_LINK_TYPES: ReadonlySet<ModelTaxonomyType> = new Set<ModelTaxonomyType>([
  'main',
  'vae',
  'controlnet',
  'control_lora',
  'clip_embed',
  'clip_vision',
  'siglip',
  'flux_redux',
  'pid_decoder',
  't5_encoder',
  'wan_t5_encoder',
  'qwen3_encoder',
  'qwen_vl_encoder',
  'qwen3_vl_encoder',
  'mistral_encoder',
  'gemma2_encoder',
  'gemma4_encoder',
]);

/** The full candidate rule for the link picker: compatible bases, and no same-type pair of a single-slot type. */
export const isLinkablePair = (a: RelatableModel, b: RelatableModel): boolean =>
  isBaseCompatible(a, b) && !(a.type === b.type && SINGLETON_LINK_TYPES.has(a.type));
