import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import { clamp } from 'es-toolkit/compat';
import type { AspectRatioID, InfillMethod, ParamsState, RgbaColor } from 'features/controlLayers/store/types';
import {
  ASPECT_RATIO_MAP,
  DEFAULT_ASPECT_RATIO_CONFIG,
  MAX_POSITIVE_PROMPT_HISTORY,
  zParamsState,
} from 'features/controlLayers/store/types';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import {
  SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS,
  SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS,
  SUPPORTS_REF_IMAGES_BASE_MODELS,
} from 'features/modelManagerV2/models';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterCLIPEmbedModel,
  ParameterCLIPGEmbedModel,
  ParameterCLIPLEmbedModel,
  ParameterControlLoRAModel,
  ParameterGuidance,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterPositivePrompt,
  ParameterPrecision,
  ParameterScheduler,
  ParameterSDXLRefinerModel,
  ParameterT5EncoderModel,
  ParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { getGridSize, getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';

import { modelChanged } from './actions';
import { selectActiveCanvasId, selectActiveTab } from './selectors';

export const getInitialParamsState = (): ParamsState => ({
  maskBlur: 16,
  maskBlurMethod: 'box',
  canvasCoherenceMode: 'Gaussian Blur',
  canvasCoherenceMinDenoise: 0,
  canvasCoherenceEdgeSize: 16,
  infillMethod: 'lama',
  infillTileSize: 32,
  infillPatchmatchDownscaleSize: 1,
  infillColorValue: { r: 0, g: 0, b: 0, a: 1 },
  cfgScale: 7.5,
  cfgRescaleMultiplier: 0,
  guidance: 4,
  img2imgStrength: 0.75,
  optimizedDenoisingEnabled: true,
  iterations: 1,
  scheduler: 'dpmpp_3m_k',
  upscaleScheduler: 'kdpm_2',
  upscaleCfgScale: 2,
  seed: 0,
  shouldRandomizeSeed: true,
  steps: 30,
  model: null,
  vae: null,
  vaePrecision: 'fp32',
  fluxVAE: null,
  seamlessXAxis: false,
  seamlessYAxis: false,
  clipSkip: 0,
  shouldUseCpuNoise: true,
  positivePrompt: '',
  positivePromptHistory: [],
  negativePrompt: null,
  refinerModel: null,
  refinerSteps: 20,
  refinerCFGScale: 7.5,
  refinerScheduler: 'euler',
  refinerPositiveAestheticScore: 6,
  refinerNegativeAestheticScore: 2.5,
  refinerStart: 0.8,
  t5EncoderModel: null,
  clipEmbedModel: null,
  clipLEmbedModel: null,
  clipGEmbedModel: null,
  controlLora: null,
  dimensions: {
    width: 512,
    height: 512,
    aspectRatio: deepClone(DEFAULT_ASPECT_RATIO_CONFIG),
  },
});

export const paramsState = createSlice({
  name: 'params',
  initialState: {} as ParamsState,
  reducers: {
    setIterations: (state, action: PayloadAction<number>) => {
      state.iterations = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    setCfgScale: (state, action: PayloadAction<ParameterCFGScale>) => {
      state.cfgScale = action.payload;
    },
    setUpscaleCfgScale: (state, action: PayloadAction<ParameterCFGScale>) => {
      state.upscaleCfgScale = action.payload;
    },
    setGuidance: (state, action: PayloadAction<ParameterGuidance>) => {
      state.guidance = action.payload;
    },
    setCfgRescaleMultiplier: (state, action: PayloadAction<ParameterCFGRescaleMultiplier>) => {
      state.cfgRescaleMultiplier = action.payload;
    },
    setScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
      state.scheduler = action.payload;
    },
    setUpscaleScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
      state.upscaleScheduler = action.payload;
    },

    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
      state.shouldRandomizeSeed = false;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setOptimizedDenoisingEnabled: (state, action: PayloadAction<boolean>) => {
      state.optimizedDenoisingEnabled = action.payload;
    },
    setSeamlessXAxis: (state, action: PayloadAction<boolean>) => {
      state.seamlessXAxis = action.payload;
    },
    setSeamlessYAxis: (state, action: PayloadAction<boolean>) => {
      state.seamlessYAxis = action.payload;
    },
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
    vaeSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      // null is a valid VAE!
      const result = zParamsState.shape.vae.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.vae = result.data;
    },
    fluxVAESelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.fluxVAE.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.fluxVAE = result.data;
    },
    t5EncoderModelSelected: (state, action: PayloadAction<ParameterT5EncoderModel | null>) => {
      const result = zParamsState.shape.t5EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.t5EncoderModel = result.data;
    },
    controlLoRAModelSelected: (state, action: PayloadAction<ParameterControlLoRAModel | null>) => {
      const result = zParamsState.shape.controlLora.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.controlLora = result.data;
    },
    clipEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPEmbedModel | null>) => {
      const result = zParamsState.shape.clipEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipEmbedModel = result.data;
    },
    clipLEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPLEmbedModel | null>) => {
      const result = zParamsState.shape.clipLEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipLEmbedModel = result.data;
    },
    clipGEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPGEmbedModel | null>) => {
      const result = zParamsState.shape.clipGEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipGEmbedModel = result.data;
    },
    vaePrecisionChanged: (state, action: PayloadAction<ParameterPrecision>) => {
      state.vaePrecision = action.payload;
    },
    setClipSkip: (state, action: PayloadAction<number>) => {
      applyClipSkip(state, state.model, action.payload);
    },
    shouldUseCpuNoiseChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCpuNoise = action.payload;
    },
    positivePromptChanged: (state, action: PayloadAction<ParameterPositivePrompt>) => {
      state.positivePrompt = action.payload;
    },
    positivePromptAddedToHistory: (state, action: PayloadAction<ParameterPositivePrompt>) => {
      const prompt = action.payload.trim();
      if (prompt.length === 0) {
        return;
      }

      state.positivePromptHistory = [prompt, ...state.positivePromptHistory.filter((p) => p !== prompt)];

      if (state.positivePromptHistory.length > MAX_POSITIVE_PROMPT_HISTORY) {
        state.positivePromptHistory = state.positivePromptHistory.slice(0, MAX_POSITIVE_PROMPT_HISTORY);
      }
    },
    promptRemovedFromHistory: (state, action: PayloadAction<string>) => {
      state.positivePromptHistory = state.positivePromptHistory.filter((p) => p !== action.payload);
    },
    promptHistoryCleared: (state) => {
      state.positivePromptHistory = [];
    },
    negativePromptChanged: (state, action: PayloadAction<ParameterNegativePrompt>) => {
      state.negativePrompt = action.payload;
    },
    refinerModelChanged: (state, action: PayloadAction<ParameterSDXLRefinerModel | null>) => {
      const result = zParamsState.shape.refinerModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.refinerModel = result.data;
    },
    setRefinerSteps: (state, action: PayloadAction<number>) => {
      state.refinerSteps = action.payload;
    },
    setRefinerCFGScale: (state, action: PayloadAction<number>) => {
      state.refinerCFGScale = action.payload;
    },
    setRefinerScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
      state.refinerScheduler = action.payload;
    },
    setRefinerPositiveAestheticScore: (state, action: PayloadAction<number>) => {
      state.refinerPositiveAestheticScore = action.payload;
    },
    setRefinerNegativeAestheticScore: (state, action: PayloadAction<number>) => {
      state.refinerNegativeAestheticScore = action.payload;
    },
    setRefinerStart: (state, action: PayloadAction<number>) => {
      state.refinerStart = action.payload;
    },
    setInfillMethod: (state, action: PayloadAction<InfillMethod>) => {
      state.infillMethod = action.payload;
    },
    setInfillTileSize: (state, action: PayloadAction<number>) => {
      state.infillTileSize = action.payload;
    },
    setInfillPatchmatchDownscaleSize: (state, action: PayloadAction<number>) => {
      state.infillPatchmatchDownscaleSize = action.payload;
    },
    setInfillColorValue: (state, action: PayloadAction<RgbaColor>) => {
      state.infillColorValue = action.payload;
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

    //#region Dimensions
    sizeRecalled: (state, action: PayloadAction<{ width: number; height: number }>) => {
      const { width, height } = action.payload;
      const gridSize = getGridSize(state.model?.base);
      state.dimensions.width = Math.max(roundDownToMultiple(width, gridSize), 64);
      state.dimensions.height = Math.max(roundDownToMultiple(height, gridSize), 64);
      state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
      state.dimensions.aspectRatio.id = 'Free';
      state.dimensions.aspectRatio.isLocked = true;
    },
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      const gridSize = getGridSize(state.model?.base);
      state.dimensions.width = clamp ? Math.max(roundDownToMultiple(width, gridSize), 64) : width;

      if (state.dimensions.aspectRatio.isLocked) {
        state.dimensions.height = roundToMultiple(
          state.dimensions.width / state.dimensions.aspectRatio.value,
          gridSize
        );
      }

      if (updateAspectRatio || !state.dimensions.aspectRatio.isLocked) {
        state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
        state.dimensions.aspectRatio.id = 'Free';
        state.dimensions.aspectRatio.isLocked = false;
      }
    },
    heightChanged: (state, action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { height, updateAspectRatio, clamp } = action.payload;
      const gridSize = getGridSize(state.model?.base);
      state.dimensions.height = clamp ? Math.max(roundDownToMultiple(height, gridSize), 64) : height;

      if (state.dimensions.aspectRatio.isLocked) {
        state.dimensions.width = roundToMultiple(
          state.dimensions.height * state.dimensions.aspectRatio.value,
          gridSize
        );
      }

      if (updateAspectRatio || !state.dimensions.aspectRatio.isLocked) {
        state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
        state.dimensions.aspectRatio.id = 'Free';
        state.dimensions.aspectRatio.isLocked = false;
      }
    },
    aspectRatioLockToggled: (state) => {
      state.dimensions.aspectRatio.isLocked = !state.dimensions.aspectRatio.isLocked;
    },
    aspectRatioIdChanged: (state, action: PayloadAction<{ id: AspectRatioID }>) => {
      const { id } = action.payload;
      state.dimensions.aspectRatio.id = id;
      if (id === 'Free') {
        state.dimensions.aspectRatio.isLocked = false;
      } else {
        state.dimensions.aspectRatio.isLocked = true;
        state.dimensions.aspectRatio.value = ASPECT_RATIO_MAP[id].ratio;
        const { width, height } = calculateNewSize(
          state.dimensions.aspectRatio.value,
          state.dimensions.width * state.dimensions.height,
          state.model?.base
        );
        state.dimensions.width = width;
        state.dimensions.height = height;
      }
    },
    dimensionsSwapped: (state) => {
      state.dimensions.aspectRatio.value = 1 / state.dimensions.aspectRatio.value;
      if (state.dimensions.aspectRatio.id === 'Free') {
        const newWidth = state.dimensions.height;
        const newHeight = state.dimensions.width;
        state.dimensions.width = newWidth;
        state.dimensions.height = newHeight;
      } else {
        const { width, height } = calculateNewSize(
          state.dimensions.aspectRatio.value,
          state.dimensions.width * state.dimensions.height,
          state.model?.base
        );
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.id = ASPECT_RATIO_MAP[state.dimensions.aspectRatio.id].inverseID;
      }
    },
    sizeOptimized: (state) => {
      const optimalDimension = getOptimalDimension(state.model?.base);
      if (state.dimensions.aspectRatio.isLocked) {
        const { width, height } = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.model?.base
        );
        state.dimensions.width = width;
        state.dimensions.height = height;
      } else {
        state.dimensions.aspectRatio = deepClone(DEFAULT_ASPECT_RATIO_CONFIG);
        state.dimensions.width = optimalDimension;
        state.dimensions.height = optimalDimension;
      }
    },
    syncedToOptimalDimension: (state) => {
      const optimalDimension = getOptimalDimension(state.model?.base);

      if (!getIsSizeOptimal(state.dimensions.width, state.dimensions.height, state.model?.base)) {
        const bboxDims = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.model?.base
        );
        state.dimensions.width = bboxDims.width;
        state.dimensions.height = bboxDims.height;
      }
    },
    paramsReset: (state) => resetState(state),
  },
  extraReducers(builder) {
    builder.addCase(modelChanged, (state, action) => {
      const { previousModel } = action.payload;
      const result = zParamsState.shape.model.safeParse(action.payload.model);
      if (!result.success) {
        return;
      }
      const model = result.data;
      state.model = model;

      // If the model base changes (e.g. SD1.5 -> SDXL), we need to change a few things
      if (model === null || previousModel?.base === model.base) {
        return;
      }

      applyClipSkip(state, model, state.clipSkip);
    });
  },
});

const applyClipSkip = (state: { clipSkip: number }, model: ParameterModel | null, clipSkip: number) => {
  if (model === null) {
    return;
  }

  const maxClip = getModelMaxClipSkip(model);

  state.clipSkip = clamp(clipSkip, 0, maxClip ?? 0);
};

const hasModelClipSkip = (model: ParameterModel | null) => {
  if (model === null) {
    return false;
  }

  return getModelMaxClipSkip(model) ?? 0 > 0;
};

const getModelMaxClipSkip = (model: ParameterModel) => {
  if (model.base === 'sdxl') {
    // We don't support user-defined CLIP skip for SDXL because it doesn't do anything useful
    return 0;
  }

  return CLIP_SKIP_MAP[model.base]?.maxClip;
};

const resetState = (state: ParamsState): ParamsState => {
  // When a new session is requested, we need to keep the current model selections, plus dependent state
  // like VAE precision. Everything else gets reset to default.
  const oldState = deepClone(state);
  const newState = getInitialParamsState();
  newState.dimensions = oldState.dimensions;
  newState.model = oldState.model;
  newState.vae = oldState.vae;
  newState.fluxVAE = oldState.fluxVAE;
  newState.vaePrecision = oldState.vaePrecision;
  newState.t5EncoderModel = oldState.t5EncoderModel;
  newState.clipEmbedModel = oldState.clipEmbedModel;
  newState.refinerModel = oldState.refinerModel;
  return newState;
};

export const isTabParamsStateAction = isAnyOf(...Object.values(paramsState.actions), modelChanged);

export const {
  setInfillMethod,
  setInfillTileSize,
  setInfillPatchmatchDownscaleSize,
  setInfillColorValue,
  setMaskBlur,
  setCanvasCoherenceMode,
  setCanvasCoherenceEdgeSize,
  setCanvasCoherenceMinDenoise,
  setIterations,
  setSteps,
  setCfgScale,
  setCfgRescaleMultiplier,
  setGuidance,
  setScheduler,
  setUpscaleScheduler,
  setUpscaleCfgScale,
  setSeed,
  setImg2imgStrength,
  setOptimizedDenoisingEnabled,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setShouldRandomizeSeed,
  vaeSelected,
  fluxVAESelected,
  vaePrecisionChanged,
  t5EncoderModelSelected,
  clipEmbedModelSelected,
  clipLEmbedModelSelected,
  clipGEmbedModelSelected,
  setClipSkip,
  shouldUseCpuNoiseChanged,
  positivePromptChanged,
  positivePromptAddedToHistory,
  promptRemovedFromHistory,
  promptHistoryCleared,
  negativePromptChanged,
  refinerModelChanged,
  setRefinerSteps,
  setRefinerCFGScale,
  setRefinerScheduler,
  setRefinerPositiveAestheticScore,
  setRefinerNegativeAestheticScore,
  setRefinerStart,

  // Dimensions
  sizeRecalled,
  widthChanged,
  heightChanged,
  aspectRatioLockToggled,
  aspectRatioIdChanged,
  dimensionsSwapped,
  sizeOptimized,
  syncedToOptimalDimension,

  paramsReset,
} = paramsState.actions;

const initialTabParamsState = getInitialParamsState();

export const selectActiveTabParams = (state: RootState) => {
  const tab = selectActiveTab(state);
  const canvasId = selectActiveCanvasId(state);

  switch (tab) {
    case 'generate':
      return state.tab.generate.params;
    case 'canvas':
      return state.canvas.canvases[canvasId]!.params.params;
    case 'upscaling':
      return state.tab.upscaling.params;
    default:
      // Fallback for global controls in other tabs
      return initialTabParamsState;
  }
};

const buildActiveTabParamsSelector =
  <T>(selector: Selector<ParamsState, T>) =>
  (state: RootState) =>
    selector(selectActiveTabParams(state));

export const selectBase = buildActiveTabParamsSelector((params) => params.model?.base);
export const selectIsSDXL = buildActiveTabParamsSelector((params) => params.model?.base === 'sdxl');
export const selectIsFLUX = buildActiveTabParamsSelector((params) => params.model?.base === 'flux');
export const selectIsSD3 = buildActiveTabParamsSelector((params) => params.model?.base === 'sd-3');
export const selectIsCogView4 = buildActiveTabParamsSelector((params) => params.model?.base === 'cogview4');
export const selectIsFluxKontext = buildActiveTabParamsSelector((params) => {
  if (params.model?.base === 'flux' && params.model?.name.toLowerCase().includes('kontext')) {
    return true;
  }
  return false;
});

export const selectModel = buildActiveTabParamsSelector((params) => params.model);
export const selectModelKey = buildActiveTabParamsSelector((params) => params.model?.key);
export const selectVAE = buildActiveTabParamsSelector((params) => params.vae);
export const selectFLUXVAE = buildActiveTabParamsSelector((params) => params.fluxVAE);
export const selectVAEKey = buildActiveTabParamsSelector((params) => params.vae?.key);
export const selectT5EncoderModel = buildActiveTabParamsSelector((params) => params.t5EncoderModel);
export const selectCLIPEmbedModel = buildActiveTabParamsSelector((params) => params.clipEmbedModel);
export const selectCLIPLEmbedModel = buildActiveTabParamsSelector((params) => params.clipLEmbedModel);

export const selectCLIPGEmbedModel = buildActiveTabParamsSelector((params) => params.clipGEmbedModel);

export const selectCFGScale = buildActiveTabParamsSelector((params) => params.cfgScale);
export const selectGuidance = buildActiveTabParamsSelector((params) => params.guidance);
export const selectSteps = buildActiveTabParamsSelector((params) => params.steps);
export const selectCFGRescaleMultiplier = buildActiveTabParamsSelector((params) => params.cfgRescaleMultiplier);
export const selectCLIPSkip = buildActiveTabParamsSelector((params) => params.clipSkip);
export const selectHasModelCLIPSkip = buildActiveTabParamsSelector((params) => hasModelClipSkip(params.model));
export const selectCanvasCoherenceEdgeSize = buildActiveTabParamsSelector((params) => params.canvasCoherenceEdgeSize);
export const selectCanvasCoherenceMinDenoise = buildActiveTabParamsSelector(
  (params) => params.canvasCoherenceMinDenoise
);
export const selectCanvasCoherenceMode = buildActiveTabParamsSelector((params) => params.canvasCoherenceMode);
export const selectMaskBlur = buildActiveTabParamsSelector((params) => params.maskBlur);
export const selectInfillMethod = buildActiveTabParamsSelector((params) => params.infillMethod);
export const selectInfillTileSize = buildActiveTabParamsSelector((params) => params.infillTileSize);
export const selectInfillPatchmatchDownscaleSize = buildActiveTabParamsSelector(
  (params) => params.infillPatchmatchDownscaleSize
);
export const selectInfillColorValue = buildActiveTabParamsSelector((params) => params.infillColorValue);
export const selectImg2imgStrength = buildActiveTabParamsSelector((params) => params.img2imgStrength);
export const selectOptimizedDenoisingEnabled = buildActiveTabParamsSelector(
  (params) => params.optimizedDenoisingEnabled
);
export const selectPositivePrompt = buildActiveTabParamsSelector((params) => params.positivePrompt);
export const selectNegativePrompt = buildActiveTabParamsSelector((params) => params.negativePrompt);
export const selectNegativePromptWithFallback = buildActiveTabParamsSelector((params) => params.negativePrompt ?? '');
export const selectHasNegativePrompt = buildActiveTabParamsSelector((params) => params.negativePrompt !== null);
export const selectModelSupportsNegativePrompt = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsRefImages = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsOptimizedDenoising = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS.includes(model.base)
);
export const selectScheduler = buildActiveTabParamsSelector((params) => params.scheduler);
export const selectSeamlessXAxis = buildActiveTabParamsSelector((params) => params.seamlessXAxis);
export const selectSeamlessYAxis = buildActiveTabParamsSelector((params) => params.seamlessYAxis);
export const selectSeed = buildActiveTabParamsSelector((params) => params.seed);
export const selectShouldRandomizeSeed = buildActiveTabParamsSelector((params) => params.shouldRandomizeSeed);
export const selectVAEPrecision = buildActiveTabParamsSelector((params) => params.vaePrecision);
export const selectIterations = buildActiveTabParamsSelector((params) => params.iterations);
export const selectShouldUseCPUNoise = buildActiveTabParamsSelector((params) => params.shouldUseCpuNoise);

export const selectUpscaleScheduler = buildActiveTabParamsSelector((params) => params.upscaleScheduler);
export const selectUpscaleCfgScale = buildActiveTabParamsSelector((params) => params.upscaleCfgScale);

export const selectPositivePromptHistory = buildActiveTabParamsSelector((params) => params.positivePromptHistory);
export const selectRefinerCFGScale = buildActiveTabParamsSelector((params) => params.refinerCFGScale);
export const selectRefinerModel = buildActiveTabParamsSelector((params) => params.refinerModel);
export const selectIsRefinerModelSelected = buildActiveTabParamsSelector((params) => Boolean(params.refinerModel));
export const selectRefinerPositiveAestheticScore = buildActiveTabParamsSelector(
  (params) => params.refinerPositiveAestheticScore
);
export const selectRefinerNegativeAestheticScore = buildActiveTabParamsSelector(
  (params) => params.refinerNegativeAestheticScore
);
export const selectRefinerScheduler = buildActiveTabParamsSelector((params) => params.refinerScheduler);
export const selectRefinerStart = buildActiveTabParamsSelector((params) => params.refinerStart);
export const selectRefinerSteps = buildActiveTabParamsSelector((params) => params.refinerSteps);

export const selectWidth = buildActiveTabParamsSelector((params) => params.dimensions.width);
export const selectHeight = buildActiveTabParamsSelector((params) => params.dimensions.height);
export const selectAspectRatioID = buildActiveTabParamsSelector((params) => params.dimensions.aspectRatio.id);
export const selectAspectRatioValue = buildActiveTabParamsSelector((params) => params.dimensions.aspectRatio.value);
export const selectAspectRatioIsLocked = buildActiveTabParamsSelector(
  (params) => params.dimensions.aspectRatio.isLocked
);
export const selectOptimalDimension = buildActiveTabParamsSelector((params) =>
  getOptimalDimension(params.model?.base ?? null)
);
export const selectGridSize = buildActiveTabParamsSelector((params) => getGridSize(params.model?.base ?? null));

export const selectMainModelConfig = createSelector(
  selectModelConfigsQuery,
  selectActiveTabParams,
  (modelConfigs, { model }) => {
    if (!modelConfigs.data) {
      return null;
    }
    if (!model) {
      return null;
    }
    const modelConfig = modelConfigsAdapterSelectors.selectById(modelConfigs.data, model.key);
    if (!modelConfig) {
      return null;
    }
    if (!isNonRefinerMainModelConfig(modelConfig)) {
      return null;
    }
    return modelConfig;
  }
);
