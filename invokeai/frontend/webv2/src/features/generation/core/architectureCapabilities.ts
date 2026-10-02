/**
 * Backend architecture/variant rows are authoritative. Handwritten DTOs are fixture-pinned; push updates into this
 * React-free registry.
 */

import { createListenerChannel } from '@platform/state/externalStoreCore';

import type { BaseGenerationConfig, GuidanceLabel, NegativePromptUsage, SchedulerSetId } from './generationConfig';

/** A capabilities row exactly as the backend serialises it. snake_case, unions kept open. */
export interface ArchitectureCapabilitiesRow {
  base: string;
  /** Null for the architecture's own row. A variant row overrides it. */
  variant: string | null;
  features: {
    negative_prompt: { visible: boolean; usage: NegativePromptUsage };
    dimension_grid: number;
    guidance_label: GuidanceLabel | (string & {});
    /** The floor the denoise node enforces on the guidance field; `0` where it enforces none. */
    guidance_min: number;
    /** The ceiling the denoise node enforces, or `null` where it enforces none. */
    guidance_max: number | null;
    /** A variant row may carry its own. */
    scheduler_set: SchedulerSetId | null;
    scheduler_applies_to_graph: boolean;
    control_kinds: string[];
    max_reference_images: number;
    reference_images_require_variant: string | null;
    supports_regional_guidance: boolean;
    regional_negative: boolean;
    clip_skip_max: number | null;
    supports_seamless: boolean;
    supports_cfg_rescale: boolean;
    sd_vae_override: boolean;
    color_compensation: boolean;
    vae_precision: boolean;
  };
  defaults: {
    vae: string | null;
    vae_precision: 'fp16' | 'fp32' | null;
    scheduler: string | null;
    steps: number | null;
    cfg_scale: number | null;
    cfg_rescale_multiplier: number | null;
    width: number | null;
    height: number | null;
    guidance: number | null;
    cpu_only: boolean | null;
    fp8_storage: boolean | null;
  } | null;
  /** Null restricts VAE compatibility to the model's own base; channel restrictions are optional. */
  vae: {
    accepted: { base: string; latent_channels: number | null }[];
  } | null;
}

const FALLBACK_STEPS = 30;
const FALLBACK_CFG_SCALE = 7;
const FALLBACK_SCHEDULER = 'euler_a';
const FALLBACK_OPTIMAL_SIDE = 1024;

/** guidance_label selects the field; CFG-off markers are not guidance values. */
const guidanceValue = (defaults: NonNullable<ArchitectureCapabilitiesRow['defaults']>, label: GuidanceLabel): number =>
  (label === 'Guidance' ? (defaults.guidance ?? defaults.cfg_scale) : (defaults.cfg_scale ?? defaults.guidance)) ??
  FALLBACK_CFG_SCALE;

/** Derive optimal side from pixel area, including nonsquare defaults. */
const optimalSide = (defaults: NonNullable<ArchitectureCapabilitiesRow['defaults']>): number => {
  const { width, height } = defaults;
  if (width && height) {
    return Math.round(Math.sqrt(width * height));
  }
  return width ?? height ?? FALLBACK_OPTIMAL_SIDE;
};

/** Map one row onto the shape generation policy already speaks. Pure. */
export const toBaseGenerationConfig = (row: ArchitectureCapabilitiesRow): BaseGenerationConfig => {
  const { features } = row;
  const defaults = row.defaults;
  const guidanceLabel: GuidanceLabel = features.guidance_label === 'Guidance' ? 'Guidance' : 'CFG';

  return {
    dimensions: {
      grid: features.dimension_grid,
      optimalSide: defaults ? optimalSide(defaults) : FALLBACK_OPTIMAL_SIDE,
    },
    defaults: {
      steps: defaults?.steps ?? FALLBACK_STEPS,
      cfgScale: defaults ? guidanceValue(defaults, guidanceLabel) : FALLBACK_CFG_SCALE,
      scheduler: defaults?.scheduler ?? FALLBACK_SCHEDULER,
    },
    schedulerSet: features.scheduler_set ?? 'standard',
    schedulerAppliesToGraph: features.scheduler_applies_to_graph,
    guidanceLabel,
    // Nullish fallbacks support older backend fields without producing NaN bounds.
    guidance: { min: features.guidance_min ?? 1, max: features.guidance_max ?? null },
    negativePrompt: features.negative_prompt,
    ui: {
      sdVaeOverride: features.sd_vae_override,
      colorCompensation: features.color_compensation,
      vaePrecision: features.vae_precision,
      seamless: features.supports_seamless,
      cfgRescale: features.supports_cfg_rescale,
      // `undefined`, not `null`: consumers branch on falsiness and on presence.
      clipSkipMax: features.clip_skip_max ?? undefined,
    },
  };
};

const key = (base: string, variant: string | null): string => `${base}\u0000${variant ?? ''}`;

let rows: readonly ArchitectureCapabilitiesRow[] | null = null;
let byKey = new Map<string, ArchitectureCapabilitiesRow>();
let configByKey = new Map<string, BaseGenerationConfig>();
let revision = 0;

/** Synchronous readers observe the registry revision signal. */
const channel = createListenerChannel();

/** Subscribe to table replacements. Returns an unsubscribe function. */
export const onArchitectureCapabilitiesChanged = channel.subscribe;

/** Identity of the table currently held; `0` while there is none. */
export const getArchitectureCapabilitiesRevision = (): number => revision;

/** Called by `data/` once the table has been fetched. */
export const setArchitectureCapabilities = (next: readonly ArchitectureCapabilitiesRow[]): void => {
  // Validate and map before atomic publication; errors must not expose a loaded empty map.
  const nextByKey = new Map(next.map((row) => [key(row.base, row.variant), row]));
  // Memoised so policy accessors stay O(1) and hand back referentially stable objects.
  const nextConfigByKey = new Map(next.map((row) => [key(row.base, row.variant), toBaseGenerationConfig(row)]));

  rows = next;
  byKey = nextByKey;
  configByKey = nextConfigByKey;
  revision += 1;
  channel.notify();
};

/** Drop the table. The store calls this on account change; tests call it to isolate. */
export const resetArchitectureCapabilities = (): void => {
  if (rows === null) {
    return;
  }

  rows = null;
  byKey = new Map();
  configByKey = new Map();
  revision = 0;
  channel.notify();
};

export const hasArchitectureCapabilities = (): boolean => rows !== null;

/** `(base, variant)` if that variant answers differently, else the architecture's own row. */
export const getArchitectureCapabilityRow = (
  base: string,
  variant?: unknown
): ArchitectureCapabilitiesRow | undefined => {
  if (typeof variant === 'string' && variant.length > 0) {
    const exact = byKey.get(key(base, variant));
    if (exact) {
      return exact;
    }
  }
  return byKey.get(key(base, null));
};

export const getArchitectureGenerationConfig = (base: string, variant?: unknown): BaseGenerationConfig | undefined => {
  if (typeof variant === 'string' && variant.length > 0) {
    const exact = configByKey.get(key(base, variant));
    if (exact) {
      return exact;
    }
  }
  return configByKey.get(key(base, null));
};

/** Look up the variant first; omitting it uses the base row. */
export const getArchitectureFeatures = (
  base: string,
  variant?: unknown
): ArchitectureCapabilitiesRow['features'] | undefined => getArchitectureCapabilityRow(base, variant)?.features;
