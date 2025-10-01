import type { ActionCreatorWithPayload, Selector, UnknownAction } from '@reduxjs/toolkit';
import { createSelector, createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { AppDispatch, RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import { isPlainObject } from 'es-toolkit';
import { clamp } from 'es-toolkit/compat';
import type {
  AspectRatioID,
  CanvasInstanceParams,
  InstanceParams,
  ParamsPayloadAction,
  ParamsState,
  RgbaColor,
} from 'features/controlLayers/store/types';
import {
  ASPECT_RATIO_MAP,
  CHATGPT_ASPECT_RATIOS,
  DEFAULT_ASPECT_RATIO_CONFIG,
  FLUX_KONTEXT_ASPECT_RATIOS,
  GEMINI_2_5_ASPECT_RATIOS,
  getInitialCanvasInstanceParamsState,
  getInitialInstanceParamsState,
  getInitialParamsState,
  IMAGEN_ASPECT_RATIOS,
  isChatGPT4oAspectRatioID,
  isFluxKontextAspectRatioID,
  isGemini2_5AspectRatioID,
  isImagenAspectRatioID,
  isParamsTab,
  MAX_POSITIVE_PROMPT_HISTORY,
  zInstanceParams,
  zParamsState,
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
import type { TabName } from 'features/ui/store/uiTypes';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { modelChanged } from './actions';
import {
  canvasAdding,
  canvasDeleted,
  canvasInitialized,
  canvasMultiCanvasMigrated,
  MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
} from './canvasSlice';
import { selectActiveCanvasId } from './selectors';

const paramsSlice = createSlice({
  name: 'params',
  initialState: getInitialParamsState,
  reducers: {},
  extraReducers(builder) {
    builder.addCase(canvasAdding, (state, action) => {
      const canvasParams = getInitialCanvasInstanceParamsState(action.payload.canvasId);
      state.canvases[canvasParams.canvasId] = canvasParams;
    });
    builder.addCase(canvasDeleted, (state, action) => {
      delete state.canvases[action.payload.canvasId];
    });
    builder.addCase(canvasMultiCanvasMigrated, (state, action) => {
      const canvasParams = state.canvases[MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER];
      if (!canvasParams) {
        return;
      }
      canvasParams.canvasId = action.payload.canvasId;
      state.canvases[canvasParams.canvasId] = canvasParams;
      delete state.canvases[MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER];
    });
    builder.addCase(canvasInitialized, (state, action) => {
      const canvasId = action.payload.canvasId;
      if (!state.canvases[canvasId]) {
        state.canvases[canvasId] = getInitialCanvasInstanceParamsState(canvasId);
      }
    });
  },
});

const instanceParamsFragment = createSlice({
  name: 'params',
  initialState: {} as InstanceParams,
  reducers: {
    setIterations: (state, action: ParamsPayloadAction<number>) => {
      state.iterations = action.payload.value;
    },
    setSteps: (state, action: ParamsPayloadAction<number>) => {
      state.steps = action.payload.value;
    },
    setCfgScale: (state, action: ParamsPayloadAction<ParameterCFGScale>) => {
      state.cfgScale = action.payload.value;
    },
    setUpscaleCfgScale: (state, action: ParamsPayloadAction<ParameterCFGScale>) => {
      state.upscaleCfgScale = action.payload.value;
    },
    setGuidance: (state, action: ParamsPayloadAction<ParameterGuidance>) => {
      state.guidance = action.payload.value;
    },
    setCfgRescaleMultiplier: (state, action: ParamsPayloadAction<ParameterCFGRescaleMultiplier>) => {
      state.cfgRescaleMultiplier = action.payload.value;
    },
    setScheduler: (state, action: ParamsPayloadAction<ParameterScheduler>) => {
      state.scheduler = action.payload.value;
    },
    setUpscaleScheduler: (state, action: ParamsPayloadAction<ParameterScheduler>) => {
      state.upscaleScheduler = action.payload.value;
    },

    setSeed: (state, action: ParamsPayloadAction<number>) => {
      state.seed = action.payload.value;
      state.shouldRandomizeSeed = false;
    },
    setImg2imgStrength: (state, action: ParamsPayloadAction<number>) => {
      state.img2imgStrength = action.payload.value;
    },
    setOptimizedDenoisingEnabled: (state, action: ParamsPayloadAction<boolean>) => {
      state.optimizedDenoisingEnabled = action.payload.value;
    },
    setSeamlessXAxis: (state, action: ParamsPayloadAction<boolean>) => {
      state.seamlessXAxis = action.payload.value;
    },
    setSeamlessYAxis: (state, action: ParamsPayloadAction<boolean>) => {
      state.seamlessYAxis = action.payload.value;
    },
    setShouldRandomizeSeed: (state, action: ParamsPayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload.value;
    },
    vaeSelected: (state, action: ParamsPayloadAction<ParameterVAEModel | null>) => {
      // null is a valid VAE!
      const result = zInstanceParams.shape.vae.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.vae = result.data;
    },
    fluxVAESelected: (state, action: ParamsPayloadAction<ParameterVAEModel | null>) => {
      const result = zInstanceParams.shape.fluxVAE.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.fluxVAE = result.data;
    },
    t5EncoderModelSelected: (state, action: ParamsPayloadAction<ParameterT5EncoderModel | null>) => {
      const result = zInstanceParams.shape.t5EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.t5EncoderModel = result.data;
    },
    controlLoRAModelSelected: (state, action: ParamsPayloadAction<ParameterControlLoRAModel | null>) => {
      const result = zInstanceParams.shape.controlLora.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.controlLora = result.data;
    },
    clipEmbedModelSelected: (state, action: ParamsPayloadAction<ParameterCLIPEmbedModel | null>) => {
      const result = zInstanceParams.shape.clipEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipEmbedModel = result.data;
    },
    clipLEmbedModelSelected: (state, action: ParamsPayloadAction<ParameterCLIPLEmbedModel | null>) => {
      const result = zInstanceParams.shape.clipLEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipLEmbedModel = result.data;
    },
    clipGEmbedModelSelected: (state, action: ParamsPayloadAction<ParameterCLIPGEmbedModel | null>) => {
      const result = zInstanceParams.shape.clipGEmbedModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.clipGEmbedModel = result.data;
    },
    vaePrecisionChanged: (state, action: ParamsPayloadAction<ParameterPrecision>) => {
      state.vaePrecision = action.payload.value;
    },
    setClipSkip: (state, action: ParamsPayloadAction<number>) => {
      applyClipSkip(state, state.model, action.payload.value);
    },
    shouldUseCpuNoiseChanged: (state, action: ParamsPayloadAction<boolean>) => {
      state.shouldUseCpuNoise = action.payload.value;
    },
    positivePromptChanged: (state, action: ParamsPayloadAction<ParameterPositivePrompt>) => {
      state.positivePrompt = action.payload.value;
    },
    positivePromptAddedToHistory: (state, action: ParamsPayloadAction<ParameterPositivePrompt>) => {
      const prompt = action.payload.value.trim();
      if (prompt.length === 0) {
        return;
      }

      state.positivePromptHistory = [prompt, ...state.positivePromptHistory.filter((p) => p !== prompt)];

      if (state.positivePromptHistory.length > MAX_POSITIVE_PROMPT_HISTORY) {
        state.positivePromptHistory = state.positivePromptHistory.slice(0, MAX_POSITIVE_PROMPT_HISTORY);
      }
    },
    promptRemovedFromHistory: (state, action: ParamsPayloadAction<string>) => {
      state.positivePromptHistory = state.positivePromptHistory.filter((p) => p !== action.payload.value);
    },
    promptHistoryCleared: (state, _action: ParamsPayloadAction) => {
      state.positivePromptHistory = [];
    },
    negativePromptChanged: (state, action: ParamsPayloadAction<ParameterNegativePrompt>) => {
      state.negativePrompt = action.payload.value;
    },
    refinerModelChanged: (state, action: ParamsPayloadAction<ParameterSDXLRefinerModel | null>) => {
      const result = zInstanceParams.shape.refinerModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.refinerModel = result.data;
    },
    setRefinerSteps: (state, action: ParamsPayloadAction<number>) => {
      state.refinerSteps = action.payload.value;
    },
    setRefinerCFGScale: (state, action: ParamsPayloadAction<number>) => {
      state.refinerCFGScale = action.payload.value;
    },
    setRefinerScheduler: (state, action: ParamsPayloadAction<ParameterScheduler>) => {
      state.refinerScheduler = action.payload.value;
    },
    setRefinerPositiveAestheticScore: (state, action: ParamsPayloadAction<number>) => {
      state.refinerPositiveAestheticScore = action.payload.value;
    },
    setRefinerNegativeAestheticScore: (state, action: ParamsPayloadAction<number>) => {
      state.refinerNegativeAestheticScore = action.payload.value;
    },
    setRefinerStart: (state, action: ParamsPayloadAction<number>) => {
      state.refinerStart = action.payload.value;
    },
    setInfillMethod: (state, action: ParamsPayloadAction<string>) => {
      state.infillMethod = action.payload.value;
    },
    setInfillTileSize: (state, action: ParamsPayloadAction<number>) => {
      state.infillTileSize = action.payload.value;
    },
    setInfillPatchmatchDownscaleSize: (state, action: ParamsPayloadAction<number>) => {
      state.infillPatchmatchDownscaleSize = action.payload.value;
    },
    setInfillColorValue: (state, action: ParamsPayloadAction<RgbaColor>) => {
      state.infillColorValue = action.payload.value;
    },
    setMaskBlur: (state, action: ParamsPayloadAction<number>) => {
      state.maskBlur = action.payload.value;
    },
    setCanvasCoherenceMode: (state, action: ParamsPayloadAction<ParameterCanvasCoherenceMode>) => {
      state.canvasCoherenceMode = action.payload.value;
    },
    setCanvasCoherenceEdgeSize: (state, action: ParamsPayloadAction<number>) => {
      state.canvasCoherenceEdgeSize = action.payload.value;
    },
    setCanvasCoherenceMinDenoise: (state, action: ParamsPayloadAction<number>) => {
      state.canvasCoherenceMinDenoise = action.payload.value;
    },

    //#region Dimensions
    sizeRecalled: (state, action: ParamsPayloadAction<{ width: number; height: number }>) => {
      const { width, height } = action.payload.value;
      const gridSize = getGridSize(state.model?.base);
      state.dimensions.width = Math.max(roundDownToMultiple(width, gridSize), 64);
      state.dimensions.height = Math.max(roundDownToMultiple(height, gridSize), 64);
      state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
      state.dimensions.aspectRatio.id = 'Free';
      state.dimensions.aspectRatio.isLocked = true;
    },
    widthChanged: (
      state,
      action: ParamsPayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>
    ) => {
      const { width, updateAspectRatio, clamp } = action.payload.value;
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
    heightChanged: (
      state,
      action: ParamsPayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>
    ) => {
      const { height, updateAspectRatio, clamp } = action.payload.value;
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
    aspectRatioLockToggled: (state, _action: ParamsPayloadAction) => {
      state.dimensions.aspectRatio.isLocked = !state.dimensions.aspectRatio.isLocked;
    },
    aspectRatioIdChanged: (state, action: ParamsPayloadAction<{ id: AspectRatioID }>) => {
      const { id } = action.payload.value;
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
    dimensionsSwapped: (state, _action: ParamsPayloadAction) => {
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
    sizeOptimized: (state, _action: ParamsPayloadAction) => {
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
    syncedToOptimalDimension: (state, _action: ParamsPayloadAction) => {
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
    paramsReset: (state, _action: ParamsPayloadAction) => resetState(state),
  },
  extraReducers(builder) {
    builder.addCase(modelChanged, (state, action) => {
      const { previousModel } = action.payload.value;
      const result = zInstanceParams.shape.model.safeParse(action.payload.value.model);
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

const resetState = (state: InstanceParams): InstanceParams => {
  // When a new session is requested, we need to keep the current model selections, plus dependent state
  // like VAE precision. Everything else gets reset to default.
  const oldState = deepClone(state);
  const newState = getInitialInstanceParamsState();
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
} = instanceParamsFragment.actions;

const instanceParamsActions = { ...instanceParamsFragment.actions, modelChanged };

type InstanceParamsAction = typeof instanceParamsActions;
type InstanceParamsActionCreator = InstanceParamsAction[keyof InstanceParamsAction];
type PayloadOf<AC> = AC extends ActionCreatorWithPayload<infer P> ? P : never;
type ValueOf<AC> = PayloadOf<AC> extends { value: infer V } ? V : never;

export const useParamsDispatch = () => {
  const dispatch = useAppDispatch();
  const activeTab = useAppSelector(selectActiveTab);
  const activeCanvasId = useAppSelector(selectActiveCanvasId);

  const paramsDispatch = <AC extends InstanceParamsActionCreator>(
    actionCreator: AC,
    ...rest: ValueOf<AC> extends never ? [] : [value: ValueOf<AC>]
  ): void => {
    const value = rest[0];

    dispatchParamsAction(actionCreator, activeTab, activeCanvasId, value, dispatch);
  };

  return useCallback(paramsDispatch, [activeTab, activeCanvasId, dispatch]);
};

export const toStore = (state: RootState, dispatch: AppDispatch) => ({ getState: () => state, dispatch });

export const paramsDispatch = <AC extends InstanceParamsActionCreator>(
  store: { getState: () => RootState; dispatch: AppDispatch },
  actionCreator: AC,
  ...rest: ValueOf<AC> extends never ? [] : [value: ValueOf<AC>]
): void => {
  const state = store.getState();
  const activeTab = selectActiveTab(state);
  const activeCanvasId = selectActiveCanvasId(state);
  const value = rest[0];

  dispatchParamsAction(actionCreator, activeTab, activeCanvasId, value, store.dispatch);
};

const dispatchParamsAction = <AC extends InstanceParamsActionCreator>(
  actionCreator: AC,
  tab: TabName,
  canvasId: string,
  value: ValueOf<AC> | undefined,
  dispatch: AppDispatch
) => {
  if (isParamsTab(tab)) {
    // Type information simplified to help TS compiler cope with too complex type
    const payload = value !== undefined ? { tab, canvasId, value } : ({ tab, canvasId } as unknown);
    const action = (actionCreator as ActionCreatorWithPayload<unknown>)(payload);

    dispatch(action);
  }
};

const isInstanceParamsAction = isAnyOf(...Object.values(instanceParamsActions));

export const paramsSliceReducer = (state: ParamsState, action: UnknownAction): ParamsState => {
  state = paramsSlice.reducer(state, action);

  if (!isInstanceParamsAction(action)) {
    return state;
  }

  const { tab, canvasId } = action.payload;

  switch (tab) {
    case 'generate':
      return {
        ...state,
        generate: instanceParamsFragment.reducer(state.generate, action),
      };
    case 'canvas':
      return {
        ...state,
        canvases: {
          ...state.canvases,
          [canvasId]: { canvasId, ...instanceParamsFragment.reducer(state.canvases[canvasId], action) },
        },
      };
    case 'upscaling':
      return {
        ...state,
        upscaling: instanceParamsFragment.reducer(state.upscaling, action),
      };
    case 'video':
      return {
        ...state,
        upscaling: instanceParamsFragment.reducer(state.video, action),
      };
  }
};

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
        const canvasParams = {
          canvasId: MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
          ...state,
        } as CanvasInstanceParams;

        state = {
          _version: 3,
          generate: { ...state },
          canvases: { [canvasParams.canvasId]: canvasParams },
          upscaling: { ...state },
          video: { ...state },
        };
      }

      return zParamsState.parse(state);
    },
  },
};

const initialInstanceParamsState = getInitialInstanceParamsState();

export const selectActiveParams = (state: RootState) => {
  const tab = selectActiveTab(state);
  const canvasId = selectActiveCanvasId(state);

  switch (tab) {
    case 'generate':
      return state.params.generate;
    case 'canvas': {
      const params = state.params.canvases[canvasId];
      assert(params, 'Params must exist for a canvas once the canvas has been created');
      return params;
    }
    case 'upscaling':
      return state.params.upscaling;
    case 'video':
      return state.params.video;
  }

  // Fallback for global controls
  return initialInstanceParamsState;
};

const buildActiveParamsSelector =
  <T>(selector: Selector<InstanceParams, T>) =>
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
