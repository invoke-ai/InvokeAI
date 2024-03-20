import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { isAnyControlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterModel,
  ParameterPrecision,
  ParameterScheduler,
  ParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { configChanged } from 'features/system/store/configSlice';
import { clamp } from 'lodash-es';
import type { RgbaColor } from 'react-colorful';
import type { ImageDTO } from 'services/api/types';

import type { GenerationState } from './types';

const initialGenerationState: GenerationState = {
  _version: 2,
  cfgScale: 7.5,
  cfgRescaleMultiplier: 0,
  height: 512,
  img2imgStrength: 0.75,
  infillMethod: 'patchmatch',
  iterations: 1,
  positivePrompt: '',
  negativePrompt: '',
  scheduler: 'euler',
  maskBlur: 16,
  maskBlurMethod: 'box',
  canvasCoherenceMode: 'Gaussian Blur',
  canvasCoherenceMinDenoise: 0,
  canvasCoherenceEdgeSize: 16,
  seed: 0,
  shouldFitToWidthHeight: true,
  shouldRandomizeSeed: true,
  steps: 50,
  infillTileSize: 32,
  infillPatchmatchDownscaleSize: 1,
  width: 512,
  model: null,
  vae: null,
  vaePrecision: 'fp32',
  seamlessXAxis: false,
  seamlessYAxis: false,
  clipSkip: 0,
  shouldUseCpuNoise: true,
  shouldShowAdvancedOptions: false,
  aspectRatio: { ...initialAspectRatioState },
  infillMosaicTileWidth: 64,
  infillMosaicTileHeight: 64,
  infillMosaicMinColor: { r: 0, g: 0, b: 0, a: 1 },
  infillMosaicMaxColor: { r: 255, g: 255, b: 255, a: 1 },
};

export const generationSlice = createSlice({
  name: 'generation',
  initialState: initialGenerationState,
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
    setCfgScale: (state, action: PayloadAction<ParameterCFGScale>) => {
      state.cfgScale = action.payload;
    },
    setCfgRescaleMultiplier: (state, action: PayloadAction<ParameterCFGRescaleMultiplier>) => {
      state.cfgRescaleMultiplier = action.payload;
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
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
    clearInitialImage: (state) => {
      state.initialImage = undefined;
    },
    setMaskBlur: (state, action: PayloadAction<number>) => {
      state.maskBlur = action.payload;
    },
    setCanvasCoherenceMode: (state, action: PayloadAction<ParameterCanvasCoherenceMode>) => {
      state.canvasCoherenceMode = action.payload;
    },
    setCanvasCoherenceEdgeSize: (state, action: PayloadAction<number>) => {
      state.canvasCoherenceEdgeSize = action.payload;
    },
    setCanvasCoherenceMinDenoise: (state, action: PayloadAction<number>) => {
      state.canvasCoherenceMinDenoise = action.payload;
    },
    setInfillMethod: (state, action: PayloadAction<string>) => {
      state.infillMethod = action.payload;
    },
    setInfillTileSize: (state, action: PayloadAction<number>) => {
      state.infillTileSize = action.payload;
    },
    setInfillPatchmatchDownscaleSize: (state, action: PayloadAction<number>) => {
      state.infillPatchmatchDownscaleSize = action.payload;
    },
    initialImageChanged: (state, action: PayloadAction<ImageDTO>) => {
      const { image_name, width, height } = action.payload;
      state.initialImage = { imageName: image_name, width, height };
    },
    modelChanged: {
      reducer: (
        state,
        action: PayloadAction<ParameterModel | null, string, { previousModel?: ParameterModel | null }>
      ) => {
        const newModel = action.payload;
        state.model = newModel;

        if (newModel === null) {
          return;
        }

        // Clamp ClipSkip Based On Selected Model
        // TODO(psyche): remove this special handling when https://github.com/invoke-ai/InvokeAI/issues/4583 is resolved
        // WIP PR here: https://github.com/invoke-ai/InvokeAI/pull/4624
        if (newModel.base === 'sdxl') {
          // We don't support clip skip for SDXL yet - it's not in the graphs
          state.clipSkip = 0;
        } else {
          const { maxClip } = CLIP_SKIP_MAP[newModel.base];
          state.clipSkip = clamp(state.clipSkip, 0, maxClip);
        }

        if (action.meta.previousModel?.base === newModel.base) {
          // The base model hasn't changed, we don't need to optimize the size
          return;
        }

        const optimalDimension = getOptimalDimension(newModel);
        if (getIsSizeOptimal(state.width, state.height, optimalDimension)) {
          return;
        }
        const { width, height } = calculateNewSize(state.aspectRatio.value, optimalDimension * optimalDimension);
        state.width = width;
        state.height = height;
      },
      prepare: (payload: ParameterModel | null, previousModel?: ParameterModel | null) => ({
        payload,
        meta: {
          previousModel,
        },
      }),
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
    shouldUseCpuNoiseChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCpuNoise = action.payload;
    },
    widthChanged: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    heightChanged: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    widthRecalled: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
      state.aspectRatio.value = action.payload / state.height;
      state.aspectRatio.id = 'Free';
      state.aspectRatio.isLocked = false;
    },
    heightRecalled: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
      state.aspectRatio.value = state.width / action.payload;
      state.aspectRatio.id = 'Free';
      state.aspectRatio.isLocked = false;
    },
    aspectRatioChanged: (state, action: PayloadAction<AspectRatioState>) => {
      state.aspectRatio = action.payload;
    },
    setInfillMosaicTileWidth: (state, action: PayloadAction<number>) => {
      state.infillMosaicTileWidth = action.payload;
    },
    setInfillMosaicTileHeight: (state, action: PayloadAction<number>) => {
      state.infillMosaicTileHeight = action.payload;
    },
    setInfillMosaicMinColor: (state, action: PayloadAction<RgbaColor>) => {
      state.infillMosaicMinColor = action.payload;
    },
    setInfillMosaicMaxColor: (state, action: PayloadAction<RgbaColor>) => {
      state.infillMosaicMaxColor = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(configChanged, (state, action) => {
      if (action.payload.sd?.scheduler) {
        state.scheduler = action.payload.sd.scheduler;
      }
      if (action.payload.sd?.vaePrecision) {
        state.vaePrecision = action.payload.sd.vaePrecision;
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
  selectors: {
    selectOptimalDimension: (slice) => getOptimalDimension(slice.model),
  },
});

export const {
  clearInitialImage,
  setCfgScale,
  setCfgRescaleMultiplier,
  setImg2imgStrength,
  setInfillMethod,
  setIterations,
  setPositivePrompt,
  setNegativePrompt,
  setScheduler,
  setMaskBlur,
  setCanvasCoherenceMode,
  setCanvasCoherenceEdgeSize,
  setCanvasCoherenceMinDenoise,
  setSeed,
  setShouldFitToWidthHeight,
  setShouldRandomizeSeed,
  setSteps,
  setInfillTileSize,
  setInfillPatchmatchDownscaleSize,
  initialImageChanged,
  modelChanged,
  vaeSelected,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setClipSkip,
  shouldUseCpuNoiseChanged,
  vaePrecisionChanged,
  aspectRatioChanged,
  widthChanged,
  heightChanged,
  widthRecalled,
  heightRecalled,
  setInfillMosaicTileWidth,
  setInfillMosaicTileHeight,
  setInfillMosaicMinColor,
  setInfillMosaicMaxColor,
} = generationSlice.actions;

export const { selectOptimalDimension } = generationSlice.selectors;

export const selectGenerationSlice = (state: RootState) => state.generation;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateGenerationState = (state: any): GenerationState => {
  if (!('_version' in state)) {
    state._version = 1;
    state.aspectRatio = initialAspectRatioState;
  }
  if (state._version === 1) {
    // The signature of the model has changed, so we need to reset it
    state._version = 2;
    state.model = null;
    state.canvasCoherenceMode = initialGenerationState.canvasCoherenceMode;
  }
  return state;
};

export const generationPersistConfig: PersistConfig<GenerationState> = {
  name: generationSlice.name,
  initialState: initialGenerationState,
  migrate: migrateGenerationState,
  persistDenylist: [],
};
