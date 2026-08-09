import type { ModelBase, ModelConfig } from './types';

// Model-base identity registry: labels, colors, and scalar display facts only.
// Generation behavior keyed by base lives in @features/generation/settings.

/** Fallback display casing for open-union taxonomy values ("t5_encoder" -> "T5 Encoder"). */
export const toTitleCase = (value: string): string =>
  value.replaceAll(/[_-]+/g, ' ').replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1));

export type ModelBaseColorPalette = 'blue' | 'cyan' | 'gray' | 'green' | 'orange' | 'pink' | 'purple' | 'red' | 'teal';

export interface ModelBaseInfo {
  base: ModelBase;
  /** Short form, for badges and filter chips. */
  label: string;
  /** Spelled-out form for list group headers; falls back to `label`. */
  longLabel?: string;
  colorPalette: ModelBaseColorPalette;
  description?: string;
  supportsDiffusersConversion?: boolean;
}

export const MODEL_BASES = {
  'sd-1': {
    base: 'sd-1',
    label: 'SD 1.x',
    longLabel: 'Stable Diffusion 1.x',
    colorPalette: 'green',
    supportsDiffusersConversion: true,
  },
  'sd-2': {
    base: 'sd-2',
    label: 'SD 2.x',
    longLabel: 'Stable Diffusion 2.x',
    colorPalette: 'teal',
    supportsDiffusersConversion: true,
  },
  sdxl: {
    base: 'sdxl',
    label: 'SDXL',
    longLabel: 'Stable Diffusion XL',
    colorPalette: 'blue',
    supportsDiffusersConversion: true,
  },
  'sdxl-refiner': {
    base: 'sdxl-refiner',
    label: 'SDXL Refiner',
    longLabel: 'Stable Diffusion XL Refiner',
    colorPalette: 'blue',
  },
  'sd-3': {
    base: 'sd-3',
    label: 'SD 3.x',
    longLabel: 'Stable Diffusion 3.x',
    colorPalette: 'purple',
  },
  flux: {
    base: 'flux',
    label: 'FLUX',
    colorPalette: 'teal',
  },
  flux2: {
    base: 'flux2',
    label: 'FLUX.2',
    colorPalette: 'cyan',
  },
  cogview4: {
    base: 'cogview4',
    label: 'CogView4',
    colorPalette: 'red',
  },
  'qwen-image': {
    base: 'qwen-image',
    label: 'Qwen Image',
    colorPalette: 'cyan',
  },
  'z-image': {
    base: 'z-image',
    label: 'Z-Image',
    colorPalette: 'orange',
  },
  'ideogram-4': {
    base: 'ideogram-4',
    label: 'Ideogram 4',
    colorPalette: 'pink',
  },
  'krea-2': {
    base: 'krea-2',
    label: 'Krea-2',
    colorPalette: 'pink',
  },
  anima: {
    base: 'anima',
    label: 'Anima',
    colorPalette: 'pink',
  },
  wan: {
    base: 'wan',
    label: 'Wan 2.2',
    colorPalette: 'cyan',
    description: 'Video architecture used for image generation at a single frame.',
  },
  any: {
    base: 'any',
    label: 'Any',
    colorPalette: 'gray',
  },
  external: {
    base: 'external',
    label: 'External',
    colorPalette: 'gray',
  },
  unknown: {
    base: 'unknown',
    label: 'Unknown',
    colorPalette: 'gray',
  },
} satisfies Record<string, ModelBaseInfo>;

export type KnownModelBase = keyof typeof MODEL_BASES;

export const KNOWN_MODEL_BASES = Object.keys(MODEL_BASES) as KnownModelBase[];

// Unknown bases are display-safe here, but generation support is decided in baseGenerationPolicies.ts.
export const getModelBaseInfo = (base: ModelBase): ModelBaseInfo =>
  (MODEL_BASES as Record<string, ModelBaseInfo>)[base] ?? {
    base,
    label: toTitleCase(base),
    colorPalette: 'gray',
  };

export const getModelBaseLabel = (base: ModelBase): string => getModelBaseInfo(base).label;

/** Spelled-out base name for list group headers; short bases read the same either way. */
export const getModelBaseLongLabel = (base: ModelBase): string => {
  const info = getModelBaseInfo(base);

  return info.longLabel ?? info.label;
};

export const getModelBaseColorPalette = (base: ModelBase): ModelBaseColorPalette => getModelBaseInfo(base).colorPalette;

export const isKnownModelBase = (base: ModelBase): base is KnownModelBase => base in MODEL_BASES;

export const isConvertibleToDiffusers = (model: Pick<ModelConfig, 'base' | 'format' | 'type'>): boolean =>
  model.format === 'checkpoint' &&
  model.type === 'main' &&
  Boolean(getModelBaseInfo(model.base).supportsDiffusersConversion);
