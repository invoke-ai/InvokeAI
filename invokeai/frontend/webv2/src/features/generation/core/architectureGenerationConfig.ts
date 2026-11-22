/** Generation's reading of the Models-owned architecture table: each served row mapped onto its policy shape. */

import type { ArchitectureCapabilitiesRow } from '@features/models/architectures';

import { getArchitectureCapabilityRow } from '@features/models/architectures';

import type { BaseGenerationConfig, GuidanceLabel } from './generationConfig';

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

// Keyed by the served row, so a config is stable for as long as its row is held and every table replacement or reset
// leaves the old configs unreachable.
const configByRow = new WeakMap<ArchitectureCapabilitiesRow, BaseGenerationConfig>();

/** `toBaseGenerationConfig` of the row `getArchitectureCapabilityRow` answers, memoised so render paths get stable objects. */
export const getArchitectureGenerationConfig = (base: string, variant?: unknown): BaseGenerationConfig | undefined => {
  const row = getArchitectureCapabilityRow(base, variant);

  if (!row) {
    return undefined;
  }

  let config = configByRow.get(row);

  if (!config) {
    config = toBaseGenerationConfig(row);
    configByRow.set(row, config);
  }

  return config;
};
