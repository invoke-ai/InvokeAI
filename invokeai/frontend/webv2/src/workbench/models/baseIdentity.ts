import type { ColorPalette } from '@chakra-ui/react';
import type { ModelBase, ModelConfig } from './types';

// Model-base identity registry: labels, colors, and scalar display facts only.
// Generation behavior keyed by base lives in @workbench/generation/baseGenerationPolicies.

export interface ModelBaseInfo {
  base: ModelBase;
  label: string;
  colorPalette: ColorPalette;
  description?: string;
  supportsDiffusersConversion?: boolean;
}

export const MODEL_BASES = {
  'sd-1': {
    base: 'sd-1',
    label: 'SD 1.x',
    colorPalette: 'green',
    supportsDiffusersConversion: true,
  },
  'sd-2': {
    base: 'sd-2',
    label: 'SD 2.x',
    colorPalette: 'teal',
    supportsDiffusersConversion: true,
  },
  sdxl: {
    base: 'sdxl',
    label: 'SDXL',
    colorPalette: 'blue',
    supportsDiffusersConversion: true,
  },
  'sdxl-refiner': {
    base: 'sdxl-refiner',
    label: 'SDXL Refiner',
    colorPalette: 'blue',
  },
  'sd-3': {
    base: 'sd-3',
    label: 'SD 3.x',
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
  anima: {
    base: 'anima',
    label: 'Anima',
    colorPalette: 'pink',
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

const toTitleCase = (value: string): string =>
  value.replaceAll(/[_-]+/g, ' ').replace(/\w\S*/g, (word) => word[0].toUpperCase() + word.slice(1));

// Unknown bases are display-safe here, but generation support is decided in baseGenerationPolicies.ts.
export const getModelBaseInfo = (base: ModelBase): ModelBaseInfo =>
  (MODEL_BASES as Record<string, ModelBaseInfo>)[base] ?? {
    base,
    label: toTitleCase(base),
    colorPalette: 'gray',
  };

export const getModelBaseLabel = (base: ModelBase): string => getModelBaseInfo(base).label;

export const getModelBaseColorPalette = (base: ModelBase): string => getModelBaseInfo(base).colorPalette;

export const isKnownModelBase = (base: ModelBase): base is KnownModelBase => base in MODEL_BASES;

export const isConvertibleToDiffusers = (model: ModelConfig): boolean =>
  model.format === 'checkpoint' &&
  model.type === 'main' &&
  Boolean(getModelBaseInfo(model.base).supportsDiffusersConversion);
