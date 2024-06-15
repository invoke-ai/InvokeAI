import type { PayloadAction } from '@reduxjs/toolkit';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterMaskBlurMethod,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterPrecision,
  ParameterScheduler,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import type { RgbaColor } from 'react-colorful';

export interface GenerationState {
  _version: 2;
  cfgScale: ParameterCFGScale;
  cfgRescaleMultiplier: ParameterCFGRescaleMultiplier;
  img2imgStrength: ParameterStrength;
  infillMethod: string;
  iterations: number;
  scheduler: ParameterScheduler;
  maskBlur: number;
  maskBlurMethod: ParameterMaskBlurMethod;
  canvasCoherenceMode: ParameterCanvasCoherenceMode;
  canvasCoherenceMinDenoise: ParameterStrength;
  canvasCoherenceEdgeSize: number;
  seed: ParameterSeed;
  shouldRandomizeSeed: boolean;
  steps: ParameterSteps;
  model: ParameterModel | null;
  vae: ParameterVAEModel | null;
  vaePrecision: ParameterPrecision;
  seamlessXAxis: boolean;
  seamlessYAxis: boolean;
  clipSkip: number;
  shouldUseCpuNoise: boolean;
  shouldShowAdvancedOptions: boolean;
  infillTileSize: number;
  infillPatchmatchDownscaleSize: number;
  infillMosaicTileWidth: number;
  infillMosaicTileHeight: number;
  infillMosaicMinColor: RgbaColor;
  infillMosaicMaxColor: RgbaColor;
  infillColorValue: RgbaColor;
  positivePrompt: ParameterPositivePrompt;
  negativePrompt: ParameterNegativePrompt;
  positivePrompt2: ParameterPositiveStylePromptSDXL;
  negativePrompt2: ParameterNegativeStylePromptSDXL;
  shouldConcatPrompts: boolean;
}

export type PayloadActionWithOptimalDimension<T = void> = PayloadAction<T, string, { optimalDimension: number }>;
