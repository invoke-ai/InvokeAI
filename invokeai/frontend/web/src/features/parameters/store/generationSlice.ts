import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { DEFAULT_SCHEDULER_NAME } from 'app/constants';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { configChanged } from 'features/system/store/configSlice';
import {
  setAspectRatio,
  setShouldShowAdvancedOptions,
} from 'features/ui/store/uiSlice';
import { clamp } from 'lodash-es';
import { ImageDTO } from 'services/api/types';
import { clipSkipMap } from '../components/Parameters/Advanced/ParamClipSkip';
import {
  CfgScaleParam,
  HeightParam,
  MainModelParam,
  NegativePromptParam,
  PositivePromptParam,
  SchedulerParam,
  SeedParam,
  StepsParam,
  StrengthParam,
  VaeModelParam,
  WidthParam,
  zMainModel,
} from './parameterZodSchemas';

export interface GenerationState {
  cfgScale: CfgScaleParam;
  height: HeightParam;
  img2imgStrength: StrengthParam;
  infillMethod: string;
  initialImage?: { imageName: string; width: number; height: number };
  iterations: number;
  perlin: number;
  positivePrompt: PositivePromptParam;
  negativePrompt: NegativePromptParam;
  scheduler: SchedulerParam;
  seamBlur: number;
  seamSize: number;
  seamSteps: number;
  seamStrength: number;
  seed: SeedParam;
  seedWeights: string;
  shouldFitToWidthHeight: boolean;
  shouldGenerateVariations: boolean;
  shouldRandomizeSeed: boolean;
  shouldUseNoiseSettings: boolean;
  steps: StepsParam;
  threshold: number;
  tileSize: number;
  variationAmount: number;
  width: WidthParam;
  shouldUseSymmetry: boolean;
  horizontalSymmetrySteps: number;
  verticalSymmetrySteps: number;
  model: MainModelParam | null;
  vae: VaeModelParam | null;
  seamlessXAxis: boolean;
  seamlessYAxis: boolean;
  clipSkip: number;
  shouldUseCpuNoise: boolean;
}

export const initialGenerationState: GenerationState = {
  cfgScale: 7.5,
  height: 512,
  img2imgStrength: 0.75,
  infillMethod: 'patchmatch',
  iterations: 1,
  perlin: 0,
  positivePrompt: '',
  negativePrompt: '',
  scheduler: DEFAULT_SCHEDULER_NAME,
  seamBlur: 16,
  seamSize: 96,
  seamSteps: 30,
  seamStrength: 0.7,
  seed: 0,
  seedWeights: '',
  shouldFitToWidthHeight: true,
  shouldGenerateVariations: false,
  shouldRandomizeSeed: true,
  shouldUseNoiseSettings: false,
  steps: 50,
  threshold: 0,
  tileSize: 32,
  variationAmount: 0.1,
  width: 512,
  shouldUseSymmetry: false,
  horizontalSymmetrySteps: 0,
  verticalSymmetrySteps: 0,
  model: null,
  vae: null,
  seamlessXAxis: false,
  seamlessYAxis: false,
  clipSkip: 0,
  shouldUseCpuNoise: true,
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
    setCfgScale: (state, action: PayloadAction<number>) => {
      state.cfgScale = action.payload;
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
    setScheduler: (state, action: PayloadAction<SchedulerParam>) => {
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
    setSeamSize: (state, action: PayloadAction<number>) => {
      state.seamSize = action.payload;
    },
    setSeamBlur: (state, action: PayloadAction<number>) => {
      state.seamBlur = action.payload;
    },
    setSeamStrength: (state, action: PayloadAction<number>) => {
      state.seamStrength = action.payload;
    },
    setSeamSteps: (state, action: PayloadAction<number>) => {
      state.seamSteps = action.payload;
    },
    setTileSize: (state, action: PayloadAction<number>) => {
      state.tileSize = action.payload;
    },
    setInfillMethod: (state, action: PayloadAction<string>) => {
      state.infillMethod = action.payload;
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
    setShouldUseNoiseSettings: (state, action: PayloadAction<boolean>) => {
      state.shouldUseNoiseSettings = action.payload;
    },
    initialImageChanged: (state, action: PayloadAction<ImageDTO>) => {
      const { image_name, width, height } = action.payload;
      state.initialImage = { imageName: image_name, width, height };
    },
    modelSelected: (state, action: PayloadAction<string>) => {
      const [base_model, type, name] = action.payload.split('/');

      state.model = zMainModel.parse({
        id: action.payload,
        base_model,
        name,
        type,
      });

      // Clamp ClipSkip Based On Selected Model
      const { maxClip } = clipSkipMap[state.model.base_model];
      state.clipSkip = clamp(state.clipSkip, 0, maxClip);
    },
    modelChanged: (state, action: PayloadAction<MainModelParam>) => {
      state.model = action.payload;
    },
    vaeSelected: (state, action: PayloadAction<VaeModelParam | null>) => {
      state.vae = action.payload;
    },
    setClipSkip: (state, action: PayloadAction<number>) => {
      state.clipSkip = action.payload;
    },
    shouldUseCpuNoiseChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCpuNoise = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(configChanged, (state, action) => {
      const defaultModel = action.payload.sd?.defaultModel;

      if (defaultModel && !state.model) {
        const [base_model, model_type, model_name] = defaultModel.split('/');
        state.model = zMainModel.parse({
          id: defaultModel,
          name: model_name,
          base_model,
        });
      }
    });
    builder.addCase(setShouldShowAdvancedOptions, (state, action) => {
      const advancedOptionsStatus = action.payload;
      if (!advancedOptionsStatus) state.clipSkip = 0;
    });
    builder.addCase(setAspectRatio, (state, action) => {
      const ratio = action.payload;
      if (ratio) {
        state.height = roundToMultiple(state.width / ratio, 8);
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
  setSeamBlur,
  setSeamSize,
  setSeamSteps,
  setSeamStrength,
  setSeed,
  setSeedWeights,
  setShouldFitToWidthHeight,
  setShouldGenerateVariations,
  setShouldRandomizeSeed,
  setSteps,
  setThreshold,
  setTileSize,
  setVariationAmount,
  setShouldUseSymmetry,
  setHorizontalSymmetrySteps,
  setVerticalSymmetrySteps,
  initialImageChanged,
  modelChanged,
  vaeSelected,
  setShouldUseNoiseSettings,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setClipSkip,
  shouldUseCpuNoiseChanged,
} = generationSlice.actions;

export default generationSlice.reducer;
