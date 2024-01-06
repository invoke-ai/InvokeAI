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
  cfgScale: ParameterCFGScale;
  cfgRescaleMultiplier: ParameterCFGRescaleMultiplier;
  height: ParameterHeight;
  img2imgStrength: ParameterStrength;
  infillMethod: string;
  initialImage?: { imageName: string; width: number; height: number };
  iterations: number;
  perlin: number;
  positivePrompt: ParameterPositivePrompt;
  negativePrompt: ParameterNegativePrompt;
  scheduler: ParameterScheduler;
  maskBlur: number;
  maskBlurMethod: ParameterMaskBlurMethod;
  canvasCoherenceMode: ParameterCanvasCoherenceMode;
  canvasCoherenceSteps: number;
  canvasCoherenceStrength: ParameterStrength;
  seed: ParameterSeed;
  seedWeights: string;
  shouldFitToWidthHeight: boolean;
  shouldGenerateVariations: boolean;
  shouldRandomizeSeed: boolean;
  steps: ParameterSteps;
  threshold: number;
  infillTileSize: number;
  infillPatchmatchDownscaleSize: number;
  variationAmount: number;
  width: ParameterWidth;
  shouldUseSymmetry: boolean;
  horizontalSymmetrySteps: number;
  verticalSymmetrySteps: number;
  model: ParameterModel | null;
  vae: ParameterVAEModel | null;
  vaePrecision: ParameterPrecision;
  seamlessXAxis: boolean;
  seamlessYAxis: boolean;
  clipSkip: number;
  shouldUseCpuNoise: boolean;
  shouldShowAdvancedOptions: boolean;
  aspectRatio: AspectRatioState;
}

export type PayloadActionWithOptimalDimension<T = void> = PayloadAction<
  T,
  string,
  { optimalDimension: number }
>;
