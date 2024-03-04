import type { PayloadAction } from '@reduxjs/toolkit';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterHeight,
  ParameterMaskBlurMethod,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterPositivePrompt,
  ParameterPrecision,
  ParameterScheduler,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterVAEModel,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';

export interface GenerationState {
  _version: 1;
  cfgScale: ParameterCFGScale;
  cfgRescaleMultiplier: ParameterCFGRescaleMultiplier;
  height: ParameterHeight;
  img2imgStrength: ParameterStrength;
  infillMethod: string;
  initialImage?: { imageName: string; width: number; height: number };
  iterations: number;
  positivePrompt: ParameterPositivePrompt;
  negativePrompt: ParameterNegativePrompt;
  scheduler: ParameterScheduler;
  maskBlur: number;
  maskBlurMethod: ParameterMaskBlurMethod;
  canvasCoherenceMode: ParameterCanvasCoherenceMode;
  canvasCoherenceMinDenoise: ParameterStrength;
  canvasCoherenceEdgeSize: number;
  seed: ParameterSeed;
  shouldFitToWidthHeight: boolean;
  shouldRandomizeSeed: boolean;
  steps: ParameterSteps;
  infillTileSize: number;
  infillPatchmatchDownscaleSize: number;
  width: ParameterWidth;
  model: ParameterModel | null;
  vae: ParameterVAEModel | null;
  vaePrecision: ParameterPrecision;
  seamlessXAxis: boolean;
  seamlessYAxis: boolean;
  clipSkip: number;
  shouldUseCpuNoise: boolean;
  shouldShowAdvancedOptions: boolean;
  aspectRatio: AspectRatioState;
  triggerPhrases: string[];
}

export type PayloadActionWithOptimalDimension<T = void> = PayloadAction<T, string, { optimalDimension: number }>;
