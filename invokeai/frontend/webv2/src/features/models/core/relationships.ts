import type { ModelBase, ModelConfig, ModelTaxonomyType } from './types';

import { architectureAcceptsVae, someArchitectureVariantAcceptsVae } from './architectureCapabilities';

/** Related-model links are symmetric curation hints and permit only models usable in one pipeline. */

/** The fields the compatibility rule reads. A model counts as each of its types: the primary one and its roles. */
export type RelatableModel = Pick<ModelConfig, 'base' | 'latent_channels' | 'roles' | 'type' | 'variant'>;

const typesOf = (model: RelatableModel): ModelTaxonomyType[] => [model.type, ...(model.roles ?? []).map((r) => r.type)];

/**
 * any denotes no base association, not universal compatibility; external and unknown convey no architecture
 * compatibility.
 */
export const NULL_BASES: ReadonlySet<string> = new Set(['any', 'external', 'unknown']);

const hasConcreteBase = (model: RelatableModel): boolean => !NULL_BASES.has(String(model.base));

/** Cite consuming backend invocations for each any-based helper allowance; config metadata alone is insufficient. */
export const NULL_BASE_ALLOWANCES: Readonly<Partial<Record<ModelTaxonomyType, ReadonlySet<ModelBase>>>> = {
  /** CLIP text encoders (flux_model_loader, sd3_model_loader). */
  clip_embed: new Set(['flux', 'sd-3']),
  /** IP-adapter image encoders (ip_adapter accepts sd-1/sdxl only; flux_ip_adapter). */
  clip_vision: new Set(['flux', 'sd-1', 'sdxl']),
  /** PiD decode (flux/flux2/qwen_image/z_image_pid_decode; sd-3/sdxl pid_decoder configs; prototype). */
  gemma2_encoder: new Set(['flux', 'flux2', 'flux3', 'moss-image', 'qwen-image', 'sd-3', 'sdxl', 'z-image']),
  /** FLUX.2 dev text encoder (flux2_dev_model_loader). */
  mistral_encoder: new Set(['flux2']),
  /** anima_model_loader, flux2_klein_model_loader, z_image_model_loader. */
  qwen3_encoder: new Set(['anima', 'flux2', 'z-image']),
  /** krea2_model_loader, qwen_image_2_1_model_loader, flux3_model_loader. */
  qwen3_vl_encoder: new Set(['flux3', 'krea-2', 'qwen-image-2-1']),
  /** qwen_image_2_1_model_loader (mmproj of a GGUF 8B encoder), krea2_model_loader (4B). */
  qwen3_vl_vision_tower: new Set(['krea-2', 'qwen-image-2-1']),
  /** anima_model_loader (Anima-3.8B's semantic connector), moss_image_model_loader. */
  qwen3_5_encoder: new Set(['anima', 'moss-image']),
  /** qwen_image_model_loader. */
  qwen_vl_encoder: new Set(['qwen-image']),
  /** FLUX Redux image encoder (flux_redux). */
  siglip: new Set(['flux']),
  /** flux_model_loader, sd3_model_loader. */
  t5_encoder: new Set(['flux', 'sd-3']),
  /** wan_model_loader. */
  wan_t5_encoder: new Set(['wan']),
};

/**
 * Cross-base helper allowances cite consuming backend invocations, like NULL_BASE_ALLOWANCES. VAEs have none: the
 * served architecture table answers which VAEs each architecture decodes with (see `isVaePair`).
 */
export const CROSS_BASE_ALLOWANCES: Readonly<
  Partial<Record<ModelTaxonomyType, Readonly<Partial<Record<ModelBase, ReadonlySet<ModelBase>>>>>>
> = {
  /** z_image_pid_decode reuses the FLUX decoder, MOSS-Image flux2_pid_decode's (assert_pid_decoder_matches_base). */
  pid_decoder: {
    flux: new Set(['z-image']),
    /** flux2_pid_decode decodes FLUX.3 and MOSS-Image latents, which are FLUX.2 latents. */
    flux2: new Set(['flux3', 'moss-image']),
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

/** Whether any type the model counts as can be linked, e.g. a Text LLM that also serves as a Qwen3.5 encoder. */
export const isLinkableModel = (model: Pick<RelatableModel, 'roles' | 'type'>): boolean =>
  typesOf({ ...model, base: 'any' }).some(isLinkableType);

/** Only concrete bases or explicitly allowed any-based helper types can receive new links. */
export const hasLinkableBase = (model: RelatableModel): boolean =>
  hasConcreteBase(model) ||
  (model.base === 'any' && typesOf(model).some((type) => NULL_BASE_ALLOWANCES[type] !== undefined));

const isAllowedHelperFor = (helper: RelatableModel, host: RelatableModel): boolean =>
  helper.base === 'any' &&
  hasConcreteBase(host) &&
  typesOf(helper).some((type) => NULL_BASE_ALLOWANCES[type]?.has(host.base) ?? false);

const isAllowedCrossBaseHelperFor = (helper: RelatableModel, host: RelatableModel): boolean =>
  typesOf(helper).some((type) => CROSS_BASE_ALLOWANCES[type]?.[helper.base]?.has(host.base) ?? false);

/**
 * A VAE and a model of another type, both of concrete bases: linkable only as the served table says, so the answer
 * waits for that table to load.
 */
export const isVaePair = (a: RelatableModel, b: RelatableModel): boolean =>
  (a.type === 'vae') !== (b.type === 'vae') && hasConcreteBase(a) && hasConcreteBase(b);

/**
 * The host's architecture decodes with the VAE. A main's variant selects its row; other hosts' variants (a Wan LoRA's
 * `5b`) are not main variants, so they take any VAE some variant of the base decodes with.
 */
const isAcceptedVaePair = (a: RelatableModel, b: RelatableModel): boolean => {
  const [vae, host] = a.type === 'vae' ? [a, b] : [b, a];
  const candidate = { base: vae.base, latent_channels: vae.latent_channels };

  return host.type === 'main'
    ? architectureAcceptsVae({ base: host.base, variant: host.variant }, candidate)
    : someArchitectureVariantAcceptsVae(host.base, candidate);
};

/**
 * Match concrete bases unless a helper allowance or, for a VAE, the served table permits crossing; any-based models
 * require explicit concrete-host allowances. VAE pairs fail closed until the table has loaded.
 */
export const isBaseCompatible = (a: RelatableModel, b: RelatableModel): boolean => {
  if (isVaePair(a, b)) {
    return isAcceptedVaePair(a, b);
  }

  if (hasConcreteBase(a) && hasConcreteBase(b)) {
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
  'qwen3_vl_vision_tower',
  'qwen3_5_encoder',
  'mistral_encoder',
  'gemma2_encoder',
  'gemma4_encoder',
]);

/** The full candidate rule for the link picker: compatible bases, and no same-type pair of a single-slot type. */
export const isLinkablePair = (a: RelatableModel, b: RelatableModel): boolean =>
  isBaseCompatible(a, b) && !typesOf(a).some((type) => SINGLETON_LINK_TYPES.has(type) && typesOf(b).includes(type));
