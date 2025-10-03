import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { extractTabActionContext } from 'app/store/util';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import { isPlainObject } from 'es-toolkit';
import { clamp } from 'es-toolkit/compat';
import type { AspectRatioID, ParamsState, RgbaColor, TabParamsState } from 'features/controlLayers/store/types';
import {
  ASPECT_RATIO_MAP,
  CHATGPT_ASPECT_RATIOS,
  DEFAULT_ASPECT_RATIO_CONFIG,
  FLUX_KONTEXT_ASPECT_RATIOS,
  GEMINI_2_5_ASPECT_RATIOS,
  IMAGEN_ASPECT_RATIOS,
  isChatGPT4oAspectRatioID,
  isFluxKontextAspectRatioID,
  isGemini2_5AspectRatioID,
  isImagenAspectRatioID,
  MAX_POSITIVE_PROMPT_HISTORY,
  zParamsState,
  zTabParamsState,
} from 'features/controlLayers/store/types';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import {
  API_BASE_MODELS,
  CLIP_SKIP_MAP,
  SUPPORTS_ASPECT_RATIO_BASE_MODELS,
  SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS,
  SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS,
  SUPPORTS_PIXEL_DIMENSIONS_BASE_MODELS,
  SUPPORTS_REF_IMAGES_BASE_MODELS,
  SUPPORTS_SEED_BASE_MODELS,
} from 'features/parameters/types/constants';
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
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { modelChanged } from './actions';
import { selectActiveCanvasId } from './selectors';

export const getInitialTabParamsState = (): TabParamsState => ({
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

const getInitialParamsState = (): ParamsState => ({
  _version: 3,
  generate: getInitialTabParamsState(),
  upscaling: getInitialTabParamsState(),
  video: getInitialTabParamsState(),
});

const paramsSlice = createSlice({
  name: 'params',
  initialState: getInitialParamsState,
  reducers: {},
  extraReducers(builder) {
    builder.addDefaultCase((state, action) => {
      const context = extractTabActionContext(action);

      if (!context) {
        return;
      }

      switch (context.tab) {
        case 'generate':
          state.generate = tabParamsState.reducer(state.generate, action);
          break;
        case 'upscaling':
          state.upscaling = tabParamsState.reducer(state.upscaling, action);
          break;
        case 'video':
          state.video = tabParamsState.reducer(state.video, action);
          break;
      }
    });
  },
});

export const tabParamsState = createSlice({
  name: 'tabParams',
  initialState: {} as TabParamsState,
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
      const result = zTabParamsState.shape.vae.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.vae = result.data;
    },
    fluxVAESelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zTabParamsState.shape.fluxVAE.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.fluxVAE = result.data;
    },
    t5EncoderModelSelected: (state, action: PayloadAction<ParameterT5EncoderModel | null>) => {
      const result = zTabParamsState.shape.t5EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.t5EncoderModel = result.data;
    },
    controlLoRAModelSelected: (state, action: PayloadAction<ParameterControlLoRAModel | null>) => {
      const result = zTabParamsState.shape.controlLora.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.controlLora = result.data;
    },
    clipEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPEmbedModel | null>) => {
      const result = zTabParamsState.shape.clipEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipEmbedModel = result.data;
    },
    clipLEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPLEmbedModel | null>) => {
      const result = zTabParamsState.shape.clipLEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipLEmbedModel = result.data;
    },
    clipGEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPGEmbedModel | null>) => {
      const result = zTabParamsState.shape.clipGEmbedModel.safeParse(action.payload);
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
      const result = zTabParamsState.shape.refinerModel.safeParse(action.payload);
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
    setInfillMethod: (state, action: PayloadAction<string>) => {
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
      } else if ((state.model?.base === 'imagen3' || state.model?.base === 'imagen4') && isImagenAspectRatioID(id)) {
        const { width, height } = IMAGEN_ASPECT_RATIOS[id];
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
        state.dimensions.aspectRatio.isLocked = true;
      } else if (state.model?.base === 'chatgpt-4o' && isChatGPT4oAspectRatioID(id)) {
        const { width, height } = CHATGPT_ASPECT_RATIOS[id];
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
        state.dimensions.aspectRatio.isLocked = true;
      } else if (state.model?.base === 'gemini-2.5' && isGemini2_5AspectRatioID(id)) {
        const { width, height } = GEMINI_2_5_ASPECT_RATIOS[id];
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
        state.dimensions.aspectRatio.isLocked = true;
      } else if (state.model?.base === 'flux-kontext' && isFluxKontextAspectRatioID(id)) {
        const { width, height } = FLUX_KONTEXT_ASPECT_RATIOS[id];
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
        state.dimensions.aspectRatio.isLocked = true;
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
      const result = zTabParamsState.shape.model.safeParse(action.payload.model);
      if (!result.success) {
        return;
      }
      const model = result.data;
      state.model = model;

      // If the model base changes (e.g. SD1.5 -> SDXL), we need to change a few things
      if (model === null || previousModel?.base === model.base) {
        return;
      }

      if (API_BASE_MODELS.includes(model.base)) {
        state.dimensions.aspectRatio.isLocked = true;
        state.dimensions.aspectRatio.value = 1;
        state.dimensions.aspectRatio.id = '1:1';
        state.dimensions.width = 1024;
        state.dimensions.height = 1024;
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

const resetState = (state: TabParamsState): TabParamsState => {
  // When a new session is requested, we need to keep the current model selections, plus dependent state
  // like VAE precision. Everything else gets reset to default.
  const oldState = deepClone(state);
  const newState = getInitialTabParamsState();
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

export const isTabParamsStateAction = isAnyOf(...Object.values(tabParamsState.actions), modelChanged);

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
} = tabParamsState.actions;

export const paramsSliceConfig: SliceConfig<typeof paramsSlice> = {
  slice: paramsSlice,
  schema: zParamsState,
  getInitialState: getInitialParamsState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));

      if (!('_version' in state)) {
        // v0 -> v1, add _version and remove x/y from dimensions, lifting width/height to top level
        state._version = 1;
        state.dimensions.width = state.dimensions.rect.width;
        state.dimensions.height = state.dimensions.rect.height;
      }

      if (state._version === 1) {
        // v1 -> v2, add positive prompt history
        state._version = 2;
        state.positivePromptHistory = [];
      }

      if (state._version === 2) {
        // Migrate from v2 to v3: slice represented shared params -> slice represents multiple tabs/canvases params
        state = {
          _version: 3,
          generate: { ...state },
          upscaling: { ...state },
          video: { ...state },
        };
      }

      return zParamsState.parse(state);
    },
  },
};

const initialTabParamsState = getInitialTabParamsState();

export const selectActiveParams = (state: RootState) => {
  const tab = selectActiveTab(state);
  const canvasId = selectActiveCanvasId(state);

  switch (tab) {
    case 'generate':
      return state.params.generate;
    case 'canvas': {
      const params = state.canvas.canvases[canvasId]?.params;
      assert(params, 'Params must exist for a canvas once the canvas has been created');
      return params;
    }
    case 'upscaling':
      return state.params.upscaling;
    case 'video':
      return state.params.video;
  }

  // Fallback for global controls
  return initialTabParamsState;
};

const buildActiveParamsSelector =
  <T>(selector: Selector<TabParamsState, T>) =>
  (state: RootState) =>
    selector(selectActiveParams(state));

export const selectBase = buildActiveParamsSelector((params) => params.model?.base);
export const selectIsSDXL = buildActiveParamsSelector((params) => params.model?.base === 'sdxl');
export const selectIsFLUX = buildActiveParamsSelector((params) => params.model?.base === 'flux');
export const selectIsSD3 = buildActiveParamsSelector((params) => params.model?.base === 'sd-3');
export const selectIsCogView4 = buildActiveParamsSelector((params) => params.model?.base === 'cogview4');
export const selectIsImagen3 = buildActiveParamsSelector((params) => params.model?.base === 'imagen3');
export const selectIsImagen4 = buildActiveParamsSelector((params) => params.model?.base === 'imagen4');
export const selectIsFluxKontext = buildActiveParamsSelector((params) => {
  if (params.model?.base === 'flux-kontext') {
    return true;
  }
  if (params.model?.base === 'flux' && params.model?.name.toLowerCase().includes('kontext')) {
    return true;
  }
  return false;
});
export const selectIsChatGPT4o = buildActiveParamsSelector((params) => params.model?.base === 'chatgpt-4o');
export const selectIsGemini2_5 = buildActiveParamsSelector((params) => params.model?.base === 'gemini-2.5');

export const selectModel = buildActiveParamsSelector((params) => params.model);
export const selectModelKey = buildActiveParamsSelector((params) => params.model?.key);
export const selectVAE = buildActiveParamsSelector((params) => params.vae);
export const selectFLUXVAE = buildActiveParamsSelector((params) => params.fluxVAE);
export const selectVAEKey = buildActiveParamsSelector((params) => params.vae?.key);
export const selectT5EncoderModel = buildActiveParamsSelector((params) => params.t5EncoderModel);
export const selectCLIPEmbedModel = buildActiveParamsSelector((params) => params.clipEmbedModel);
export const selectCLIPLEmbedModel = buildActiveParamsSelector((params) => params.clipLEmbedModel);

export const selectCLIPGEmbedModel = buildActiveParamsSelector((params) => params.clipGEmbedModel);

export const selectCFGScale = buildActiveParamsSelector((params) => params.cfgScale);
export const selectGuidance = buildActiveParamsSelector((params) => params.guidance);
export const selectSteps = buildActiveParamsSelector((params) => params.steps);
export const selectCFGRescaleMultiplier = buildActiveParamsSelector((params) => params.cfgRescaleMultiplier);
export const selectCLIPSkip = buildActiveParamsSelector((params) => params.clipSkip);
export const selectHasModelCLIPSkip = buildActiveParamsSelector((params) => hasModelClipSkip(params.model));
export const selectCanvasCoherenceEdgeSize = buildActiveParamsSelector((params) => params.canvasCoherenceEdgeSize);
export const selectCanvasCoherenceMinDenoise = buildActiveParamsSelector((params) => params.canvasCoherenceMinDenoise);
export const selectCanvasCoherenceMode = buildActiveParamsSelector((params) => params.canvasCoherenceMode);
export const selectMaskBlur = buildActiveParamsSelector((params) => params.maskBlur);
export const selectInfillMethod = buildActiveParamsSelector((params) => params.infillMethod);
export const selectInfillTileSize = buildActiveParamsSelector((params) => params.infillTileSize);
export const selectInfillPatchmatchDownscaleSize = buildActiveParamsSelector(
  (params) => params.infillPatchmatchDownscaleSize
);
export const selectInfillColorValue = buildActiveParamsSelector((params) => params.infillColorValue);
export const selectImg2imgStrength = buildActiveParamsSelector((params) => params.img2imgStrength);
export const selectOptimizedDenoisingEnabled = buildActiveParamsSelector((params) => params.optimizedDenoisingEnabled);
export const selectPositivePrompt = buildActiveParamsSelector((params) => params.positivePrompt);
export const selectNegativePrompt = buildActiveParamsSelector((params) => params.negativePrompt);
export const selectNegativePromptWithFallback = buildActiveParamsSelector((params) => params.negativePrompt ?? '');
export const selectHasNegativePrompt = buildActiveParamsSelector((params) => params.negativePrompt !== null);
export const selectModelSupportsNegativePrompt = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsSeed = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_SEED_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsRefImages = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsAspectRatio = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_ASPECT_RATIO_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsPixelDimensions = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_PIXEL_DIMENSIONS_BASE_MODELS.includes(model.base)
);
export const selectIsApiBaseModel = createSelector(
  selectModel,
  (model) => !!model && API_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsOptimizedDenoising = createSelector(
  selectModel,
  (model) => !!model && SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS.includes(model.base)
);
export const selectScheduler = buildActiveParamsSelector((params) => params.scheduler);
export const selectSeamlessXAxis = buildActiveParamsSelector((params) => params.seamlessXAxis);
export const selectSeamlessYAxis = buildActiveParamsSelector((params) => params.seamlessYAxis);
export const selectSeed = buildActiveParamsSelector((params) => params.seed);
export const selectShouldRandomizeSeed = buildActiveParamsSelector((params) => params.shouldRandomizeSeed);
export const selectVAEPrecision = buildActiveParamsSelector((params) => params.vaePrecision);
export const selectIterations = buildActiveParamsSelector((params) => params.iterations);
export const selectShouldUseCPUNoise = buildActiveParamsSelector((params) => params.shouldUseCpuNoise);

export const selectUpscaleScheduler = buildActiveParamsSelector((params) => params.upscaleScheduler);
export const selectUpscaleCfgScale = buildActiveParamsSelector((params) => params.upscaleCfgScale);

export const selectPositivePromptHistory = buildActiveParamsSelector((params) => params.positivePromptHistory);
export const selectRefinerCFGScale = buildActiveParamsSelector((params) => params.refinerCFGScale);
export const selectRefinerModel = buildActiveParamsSelector((params) => params.refinerModel);
export const selectIsRefinerModelSelected = buildActiveParamsSelector((params) => Boolean(params.refinerModel));
export const selectRefinerPositiveAestheticScore = buildActiveParamsSelector(
  (params) => params.refinerPositiveAestheticScore
);
export const selectRefinerNegativeAestheticScore = buildActiveParamsSelector(
  (params) => params.refinerNegativeAestheticScore
);
export const selectRefinerScheduler = buildActiveParamsSelector((params) => params.refinerScheduler);
export const selectRefinerStart = buildActiveParamsSelector((params) => params.refinerStart);
export const selectRefinerSteps = buildActiveParamsSelector((params) => params.refinerSteps);

export const selectWidth = buildActiveParamsSelector((params) => params.dimensions.width);
export const selectHeight = buildActiveParamsSelector((params) => params.dimensions.height);
export const selectAspectRatioID = buildActiveParamsSelector((params) => params.dimensions.aspectRatio.id);
export const selectAspectRatioValue = buildActiveParamsSelector((params) => params.dimensions.aspectRatio.value);
export const selectAspectRatioIsLocked = buildActiveParamsSelector((params) => params.dimensions.aspectRatio.isLocked);
export const selectOptimalDimension = buildActiveParamsSelector((params) =>
  getOptimalDimension(params.model?.base ?? null)
);
export const selectGridSize = buildActiveParamsSelector((params) => getGridSize(params.model?.base ?? null));

export const selectMainModelConfig = createSelector(
  selectModelConfigsQuery,
  selectActiveParams,
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
