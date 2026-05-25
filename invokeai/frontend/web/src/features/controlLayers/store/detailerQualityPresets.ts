import type { DetailerQuality, ParamsState } from 'features/controlLayers/store/types';

export type DetailerQualityPreset = Pick<
  ParamsState,
  | 'detailerTargetSize'
  | 'detailerMaxUpscale'
  | 'detailerMaxProcessSize'
  | 'detailerCropPadding'
  | 'detailerStrength'
  | 'detailerSteps'
  | 'detailerCfgScale'
  | 'detailerDenoiseMaskFeather'
  | 'detailerPasteMaskExpand'
  | 'detailerPasteMaskFeather'
  | 'detailerSamModel'
>;

export type DetailerQualityPresetKey = keyof DetailerQualityPreset;
export type DetailerQualityPresetState = Pick<ParamsState, 'detailerQuality' | DetailerQualityPresetKey>;

export const DETAILER_QUALITY_PRESETS = {
  fast: {
    detailerTargetSize: 512,
    detailerMaxUpscale: 4,
    detailerMaxProcessSize: 512,
    detailerCropPadding: 64,
    detailerStrength: 0.24,
    detailerSteps: 8,
    detailerCfgScale: 4,
    detailerDenoiseMaskFeather: 4,
    detailerPasteMaskExpand: 0,
    detailerPasteMaskFeather: 6,
    detailerSamModel: 'segment-anything-2-small',
  },
  balanced: {
    detailerTargetSize: 768,
    detailerMaxUpscale: 8,
    detailerMaxProcessSize: 1024,
    detailerCropPadding: 48,
    detailerStrength: 0.26,
    detailerSteps: 14,
    detailerCfgScale: 4.5,
    detailerDenoiseMaskFeather: 6,
    detailerPasteMaskExpand: 0,
    detailerPasteMaskFeather: 6,
    detailerSamModel: 'segment-anything-2-base',
  },
  high: {
    detailerTargetSize: 1024,
    detailerMaxUpscale: 12,
    detailerMaxProcessSize: 1024,
    detailerCropPadding: 32,
    detailerStrength: 0.28,
    detailerSteps: 18,
    detailerCfgScale: 5,
    detailerDenoiseMaskFeather: 6,
    detailerPasteMaskExpand: 0,
    detailerPasteMaskFeather: 6,
    detailerSamModel: 'segment-anything-2-large',
  },
} satisfies Record<DetailerQuality, DetailerQualityPreset>;

export const DETAILER_QUALITY_PRESET_KEYS = Object.keys(
  DETAILER_QUALITY_PRESETS.balanced
) as DetailerQualityPresetKey[];

export const isDetailerQualityPresetAdjusted = (params: DetailerQualityPresetState) => {
  const preset = DETAILER_QUALITY_PRESETS[params.detailerQuality];

  return DETAILER_QUALITY_PRESET_KEYS.some((key) => params[key] !== preset[key]);
};
