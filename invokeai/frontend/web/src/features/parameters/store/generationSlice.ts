import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { configChanged } from 'features/system/store/configSlice';
import { clamp } from 'lodash-es';
import { ImageDTO } from 'services/api/types';

import { isAnyControlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import {
  ParameterCanvasCoherenceMode,
  ParameterCFGScale,
  ParameterHeight,
  ParameterHRFMethod,
  ParameterModel,
  ParameterMaskBlurMethod,
  ParameterNegativePrompt,
  ParameterPositivePrompt,
  ParameterPrecision,
  ParameterScheduler,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterVAEModel,
  ParameterWidth,
  zParameterModel,
  ParameterCFGRescaleMultiplier,
} from 'features/parameters/types/parameterSchemas';

export interface GenerationState {
  hrfEnabled: boolean;
  hrfStrength: ParameterStrength;
  hrfMethod: ParameterHRFMethod;
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
  aspectRatio: number | null;
  shouldLockAspectRatio: boolean;
}

export const initialGenerationState: GenerationState = {
  hrfStrength: 0.45,
  hrfEnabled: false,
  hrfMethod: 'ESRGAN',
  cfgScale: 7.5,
  cfgRescaleMultiplier: 0,
  height: 512,
  img2imgStrength: 0.75,
  infillMethod: 'patchmatch',
  iterations: 1,
  perlin: 0,
  positivePrompt: '',
  negativePrompt: '',
  scheduler: 'euler',
  maskBlur: 16,
  maskBlurMethod: 'box',
  canvasCoherenceMode: 'unmasked',
  canvasCoherenceSteps: 20,
  canvasCoherenceStrength: 0.3,
  seed: 0,
  seedWeights: '',
  shouldFitToWidthHeight: true,
  shouldGenerateVariations: false,
  shouldRandomizeSeed: true,
  steps: 50,
  threshold: 0,
  infillTileSize: 32,
  infillPatchmatchDownscaleSize: 1,
  variationAmount: 0.1,
  width: 512,
  shouldUseSymmetry: false,
  horizontalSymmetrySteps: 0,
  verticalSymmetrySteps: 0,
  model: null,
  vae: null,
  vaePrecision: 'fp32',
  seamlessXAxis: false,
  seamlessYAxis: false,
  clipSkip: 0,
  shouldUseCpuNoise: true,
  shouldShowAdvancedOptions: false,
  aspectRatio: null,
  shouldLockAspectRatio: false,
};

const initialState: GenerationState = initialGenerationState;

export const generationSlice = createSlice({
  name: 'generation',
  initialState,
  reducers: {
    setPositivePrompt: (state, action: PayloadAction<string>) => {
      state.positivePrompt = action.payload;
    },
    setNegativePrompt: (state, action: PayloadAction<string>) => {
      state.negativePrompt = action.payload;
    },
    setIterations: (state, action: PayloadAction<number>) => {
      state.iterations = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    clampSymmetrySteps: (state) => {
      state.horizontalSymmetrySteps = clamp(
        state.horizontalSymmetrySteps,
        0,
        state.steps
      );
      state.verticalSymmetrySteps = clamp(
        state.verticalSymmetrySteps,
        0,
        state.steps
      );
    },
    setCfgScale: (state, action: PayloadAction<ParameterCFGScale>) => {
      state.cfgScale = action.payload;
    },
    setCfgRescaleMultiplier: (
      state,
      action: PayloadAction<ParameterCFGRescaleMultiplier>
    ) => {
      state.cfgRescaleMultiplier = action.payload;
    },
    setThreshold: (state, action: PayloadAction<number>) => {
      state.threshold = action.payload;
    },
    setPerlin: (state, action: PayloadAction<number>) => {
      state.perlin = action.payload;
    },
    setHeight: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    setWidth: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    toggleSize: (state) => {
      const [width, height] = [state.width, state.height];
      state.width = height;
      state.height = width;
    },
    setScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
      state.scheduler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
      state.shouldRandomizeSeed = false;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setSeamlessXAxis: (state, action: PayloadAction<boolean>) => {
      state.seamlessXAxis = action.payload;
    },
    setSeamlessYAxis: (state, action: PayloadAction<boolean>) => {
      state.seamlessYAxis = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    setShouldGenerateVariations: (state, action: PayloadAction<boolean>) => {
      state.shouldGenerateVariations = action.payload;
    },
    setVariationAmount: (state, action: PayloadAction<number>) => {
      state.variationAmount = action.payload;
    },
    setSeedWeights: (state, action: PayloadAction<string>) => {
      state.seedWeights = action.payload;
      state.shouldGenerateVariations = true;
      state.variationAmount = 0;
    },
    resetParametersState: (state) => {
      return {
        ...state,
        ...initialGenerationState,
      };
    },
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
    clearInitialImage: (state) => {
      state.initialImage = undefined;
    },
    setMaskBlur: (state, action: PayloadAction<number>) => {
      state.maskBlur = action.payload;
    },
    setMaskBlurMethod: (
      state,
      action: PayloadAction<ParameterMaskBlurMethod>
    ) => {
      state.maskBlurMethod = action.payload;
    },
    setCanvasCoherenceMode: (
      state,
      action: PayloadAction<ParameterCanvasCoherenceMode>
    ) => {
      state.canvasCoherenceMode = action.payload;
    },
    setCanvasCoherenceSteps: (state, action: PayloadAction<number>) => {
      state.canvasCoherenceSteps = action.payload;
    },
    setCanvasCoherenceStrength: (state, action: PayloadAction<number>) => {
      state.canvasCoherenceStrength = action.payload;
    },
    setInfillMethod: (state, action: PayloadAction<string>) => {
      state.infillMethod = action.payload;
    },
    setInfillTileSize: (state, action: PayloadAction<number>) => {
      state.infillTileSize = action.payload;
    },
    setInfillPatchmatchDownscaleSize: (
      state,
      action: PayloadAction<number>
    ) => {
      state.infillPatchmatchDownscaleSize = action.payload;
    },
    setShouldUseSymmetry: (state, action: PayloadAction<boolean>) => {
      state.shouldUseSymmetry = action.payload;
    },
    setHorizontalSymmetrySteps: (state, action: PayloadAction<number>) => {
      state.horizontalSymmetrySteps = action.payload;
    },
    setVerticalSymmetrySteps: (state, action: PayloadAction<number>) => {
      state.verticalSymmetrySteps = action.payload;
    },
    initialImageChanged: (state, action: PayloadAction<ImageDTO>) => {
      const { image_name, width, height } = action.payload;
      state.initialImage = { imageName: image_name, width, height };
    },
    modelChanged: (state, action: PayloadAction<ParameterModel | null>) => {
      state.model = action.payload;

      if (state.model === null) {
        return;
      }

      // Clamp ClipSkip Based On Selected Model
      const { maxClip } = CLIP_SKIP_MAP[state.model.base_model];
      state.clipSkip = clamp(state.clipSkip, 0, maxClip);
    },
    vaeSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      // null is a valid VAE!
      state.vae = action.payload;
    },
    vaePrecisionChanged: (state, action: PayloadAction<ParameterPrecision>) => {
      state.vaePrecision = action.payload;
    },
    setClipSkip: (state, action: PayloadAction<number>) => {
      state.clipSkip = action.payload;
    },
    setHrfStrength: (state, action: PayloadAction<number>) => {
      state.hrfStrength = action.payload;
    },
    setHrfEnabled: (state, action: PayloadAction<boolean>) => {
      state.hrfEnabled = action.payload;
    },
    setHrfMethod: (state, action: PayloadAction<ParameterHRFMethod>) => {
      state.hrfMethod = action.payload;
    },
    shouldUseCpuNoiseChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCpuNoise = action.payload;
    },
    setAspectRatio: (state, action: PayloadAction<number | null>) => {
      const newAspectRatio = action.payload;
      state.aspectRatio = newAspectRatio;
      if (newAspectRatio) {
        state.height = roundToMultiple(state.width / newAspectRatio, 8);
      }
    },
    setShouldLockAspectRatio: (state, action: PayloadAction<boolean>) => {
      state.shouldLockAspectRatio = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(configChanged, (state, action) => {
      const defaultModel = action.payload.sd?.defaultModel;

      if (defaultModel && !state.model) {
        const [base_model, model_type, model_name] = defaultModel.split('/');

        const result = zParameterModel.safeParse({
          model_name,
          base_model,
          model_type,
        });

        if (result.success) {
          state.model = result.data;
        }
      }
    });

    // TODO: This is a temp fix to reduce issues with T2I adapter having a different downscaling
    // factor than the UNet. Hopefully we get an upstream fix in diffusers.
    builder.addMatcher(isAnyControlAdapterAdded, (state, action) => {
      if (action.payload.type === 't2i_adapter') {
        state.width = roundToMultiple(state.width, 64);
        state.height = roundToMultiple(state.height, 64);
      }
    });
  },
});

export const {
  clampSymmetrySteps,
  clearInitialImage,
  resetParametersState,
  resetSeed,
  setCfgScale,
  setCfgRescaleMultiplier,
  setWidth,
  setHeight,
  toggleSize,
  setImg2imgStrength,
  setInfillMethod,
  setIterations,
  setPerlin,
  setPositivePrompt,
  setNegativePrompt,
  setScheduler,
  setMaskBlur,
  setMaskBlurMethod,
  setCanvasCoherenceMode,
  setCanvasCoherenceSteps,
  setCanvasCoherenceStrength,
  setSeed,
  setSeedWeights,
  setShouldFitToWidthHeight,
  setShouldGenerateVariations,
  setShouldRandomizeSeed,
  setSteps,
  setThreshold,
  setInfillTileSize,
  setInfillPatchmatchDownscaleSize,
  setVariationAmount,
  setShouldUseSymmetry,
  setHorizontalSymmetrySteps,
  setVerticalSymmetrySteps,
  initialImageChanged,
  modelChanged,
  vaeSelected,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setClipSkip,
  setHrfEnabled,
  setHrfStrength,
  setHrfMethod,
  shouldUseCpuNoiseChanged,
  setAspectRatio,
  setShouldLockAspectRatio,
  vaePrecisionChanged,
} = generationSlice.actions;

export default generationSlice.reducer;
