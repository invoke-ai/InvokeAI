import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import { isPlainObject } from 'es-toolkit';
import { clamp } from 'es-toolkit/compat';
import { logout } from 'features/auth/store/authSlice';
import type { AspectRatioID, InfillMethod, ParamsState, RgbaColor } from 'features/controlLayers/store/types';
import {
  ASPECT_RATIO_MAP,
  DEFAULT_ASPECT_RATIO_CONFIG,
  getInitialParamsState,
  MAX_POSITIVE_PROMPT_HISTORY,
  zParamsState,
} from 'features/controlLayers/store/types';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import {
  SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS,
  SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS,
  SUPPORTS_REF_IMAGES_BASE_MODELS,
} from 'features/modelManagerV2/models';
import type { BaseModelType } from 'features/nodes/types/common';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterCLIPEmbedModel,
  ParameterCLIPGEmbedModel,
  ParameterCLIPLEmbedModel,
  ParameterControlLoRAModel,
  ParameterFluxDypePreset,
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
import { getExternalPanelControl, hasExternalPanelControl } from 'features/parameters/util/externalPanelSchema';
import { getGridSize, getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfigWithExternal } from 'services/api/types';
import { isExternalApiModelConfig, isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const slice = createSlice({
  name: 'params',
  initialState: getInitialParamsState(),
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
    setFluxScheduler: (state, action: PayloadAction<'euler' | 'heun' | 'lcm'>) => {
      state.fluxScheduler = action.payload;
    },
    setFluxDypePreset: (state, action: PayloadAction<ParameterFluxDypePreset>) => {
      state.fluxDypePreset = action.payload;
    },
    setFluxDypeScale: (state, action: PayloadAction<number>) => {
      state.fluxDypeScale = action.payload;
    },
    setFluxDypeExponent: (state, action: PayloadAction<number>) => {
      state.fluxDypeExponent = action.payload;
    },
    setZImageScheduler: (state, action: PayloadAction<'euler' | 'heun' | 'lcm'>) => {
      state.zImageScheduler = action.payload;
    },
    setZImageShift: (state, action: PayloadAction<number | null>) => {
      state.zImageShift = action.payload;
    },
    setZImageSeedVarianceEnabled: (state, action: PayloadAction<boolean>) => {
      state.zImageSeedVarianceEnabled = action.payload;
    },
    setZImageSeedVarianceStrength: (state, action: PayloadAction<number>) => {
      state.zImageSeedVarianceStrength = action.payload;
    },
    setZImageSeedVarianceRandomizePercent: (state, action: PayloadAction<number>) => {
      state.zImageSeedVarianceRandomizePercent = action.payload;
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
    modelChanged: (
      state,
      action: PayloadAction<{ model: ParameterModel | null; previousModel?: ParameterModel | null }>
    ) => {
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
    zImageVaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.zImageVaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.zImageVaeModel = result.data;
    },
    zImageQwen3EncoderModelSelected: (
      state,
      action: PayloadAction<{ key: string; name: string; base: string } | null>
    ) => {
      const result = zParamsState.shape.zImageQwen3EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.zImageQwen3EncoderModel = result.data;
    },
    zImageQwen3SourceModelSelected: (state, action: PayloadAction<ParameterModel | null>) => {
      const result = zParamsState.shape.zImageQwen3SourceModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.zImageQwen3SourceModel = result.data;
    },
    animaVaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.animaVaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.animaVaeModel = result.data;
    },
    animaQwen3EncoderModelSelected: (state, action: PayloadAction<ParameterT5EncoderModel | null>) => {
      const result = zParamsState.shape.animaQwen3EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.animaQwen3EncoderModel = result.data;
    },
    animaT5EncoderModelSelected: (state, action: PayloadAction<ParameterT5EncoderModel | null>) => {
      const result = zParamsState.shape.animaT5EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.animaT5EncoderModel = result.data;
    },
    setAnimaScheduler: (
      state,
      action: PayloadAction<'euler' | 'heun' | 'dpmpp_2m' | 'dpmpp_2m_sde' | 'er_sde' | 'lcm'>
    ) => {
      state.animaScheduler = action.payload;
    },
    kleinVaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.kleinVaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.kleinVaeModel = result.data;
    },
    kleinQwen3EncoderModelSelected: (
      state,
      action: PayloadAction<{ key: string; name: string; base: string } | null>
    ) => {
      const result = zParamsState.shape.kleinQwen3EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.kleinQwen3EncoderModel = result.data;
    },
    qwenImageComponentSourceSelected: (state, action: PayloadAction<ParameterModel | null>) => {
      const result = zParamsState.shape.qwenImageComponentSource.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.qwenImageComponentSource = result.data;
    },
    qwenImageVaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.qwenImageVaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.qwenImageVaeModel = result.data;
    },
    qwenImageQwenVLEncoderModelSelected: (
      state,
      action: PayloadAction<{ key: string; name: string; base: string } | null>
    ) => {
      const result = zParamsState.shape.qwenImageQwenVLEncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.qwenImageQwenVLEncoderModel = result.data;
    },
    qwenImageQuantizationChanged: (state, action: PayloadAction<'none' | 'int8' | 'nf4'>) => {
      state.qwenImageQuantization = action.payload;
    },
    qwenImageShiftChanged: (state, action: PayloadAction<number | null>) => {
      state.qwenImageShift = action.payload;
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
    setColorCompensation: (state, action: PayloadAction<boolean>) => {
      state.colorCompensation = action.payload;
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
      const gridSize = getGridSize(state.model?.base as BaseModelType | undefined);
      state.dimensions.width = Math.max(roundDownToMultiple(width, gridSize), 64);
      state.dimensions.height = Math.max(roundDownToMultiple(height, gridSize), 64);
      state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
      state.dimensions.aspectRatio.id = 'Free';
      state.dimensions.aspectRatio.isLocked = true;
    },
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      const gridSize = getGridSize(state.model?.base as BaseModelType | undefined);
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
      const gridSize = getGridSize(state.model?.base as BaseModelType | undefined);
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
    aspectRatioIdChanged: (
      state,
      action: PayloadAction<{ id: AspectRatioID; fixedSize?: { width: number; height: number } }>
    ) => {
      const { id, fixedSize } = action.payload;
      state.dimensions.aspectRatio.id = id;
      if (id === 'Free') {
        state.dimensions.aspectRatio.isLocked = false;
      } else {
        state.dimensions.aspectRatio.isLocked = true;
        if (fixedSize) {
          state.dimensions.aspectRatio.value = fixedSize.width / fixedSize.height;
          state.dimensions.width = fixedSize.width;
          state.dimensions.height = fixedSize.height;
        } else {
          state.dimensions.aspectRatio.value = ASPECT_RATIO_MAP[id].ratio;
          const { width, height } = calculateNewSize(
            state.dimensions.aspectRatio.value,
            state.dimensions.width * state.dimensions.height,
            state.model?.base as BaseModelType | undefined
          );
          state.dimensions.width = width;
          state.dimensions.height = height;
        }
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
          state.model?.base as BaseModelType | undefined
        );
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.id = ASPECT_RATIO_MAP[state.dimensions.aspectRatio.id].inverseID;
      }
    },
    sizeOptimized: (state) => {
      const optimalDimension = getOptimalDimension(state.model?.base as BaseModelType | undefined);
      if (state.dimensions.aspectRatio.isLocked) {
        const { width, height } = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.model?.base as BaseModelType | undefined
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
      const optimalDimension = getOptimalDimension(state.model?.base as BaseModelType | undefined);

      if (
        !getIsSizeOptimal(
          state.dimensions.width,
          state.dimensions.height,
          state.model?.base as BaseModelType | undefined
        )
      ) {
        const bboxDims = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.model?.base as BaseModelType | undefined
        );
        state.dimensions.width = bboxDims.width;
        state.dimensions.height = bboxDims.height;
      }
    },
    imageSizeChanged: (state, action: PayloadAction<string | null>) => {
      state.imageSize = action.payload;
    },
    openaiQualityChanged: (state, action: PayloadAction<'auto' | 'high' | 'medium' | 'low'>) => {
      state.openaiQuality = action.payload;
    },
    openaiBackgroundChanged: (state, action: PayloadAction<'auto' | 'transparent' | 'opaque'>) => {
      state.openaiBackground = action.payload;
    },
    openaiInputFidelityChanged: (state, action: PayloadAction<'low' | 'high' | null>) => {
      state.openaiInputFidelity = action.payload;
    },
    geminiTemperatureChanged: (state, action: PayloadAction<number | null>) => {
      state.geminiTemperature = action.payload;
    },
    geminiThinkingLevelChanged: (state, action: PayloadAction<'minimal' | 'high' | null>) => {
      state.geminiThinkingLevel = action.payload;
    },
    seedreamWatermarkChanged: (state, action: PayloadAction<boolean>) => {
      state.seedreamWatermark = action.payload;
    },
    seedreamOptimizePromptChanged: (state, action: PayloadAction<boolean>) => {
      state.seedreamOptimizePrompt = action.payload;
    },
    resolutionPresetSelected: (
      state,
      action: PayloadAction<{ imageSize: string; aspectRatio: string; width: number; height: number }>
    ) => {
      const { imageSize, aspectRatio, width, height } = action.payload;
      state.imageSize = imageSize;
      state.dimensions.width = width;
      state.dimensions.height = height;
      state.dimensions.aspectRatio.id = aspectRatio as AspectRatioID;
      state.dimensions.aspectRatio.value = width / height;
      state.dimensions.aspectRatio.isLocked = true;
    },
    paramsReset: (state) => resetState(state),
    paramsRecalled: (_state, action: PayloadAction<ParamsState>) => {
      return action.payload;
    },
  },
  extraReducers(builder) {
    // Reset params state on logout to prevent user data leakage when switching users
    builder.addCase(logout, () => {
      return getInitialParamsState();
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
  if (model.base === 'external') {
    return undefined;
  }
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
  newState.zImageVaeModel = oldState.zImageVaeModel;
  newState.zImageQwen3EncoderModel = oldState.zImageQwen3EncoderModel;
  newState.zImageQwen3SourceModel = oldState.zImageQwen3SourceModel;
  newState.animaVaeModel = oldState.animaVaeModel;
  newState.animaQwen3EncoderModel = oldState.animaQwen3EncoderModel;
  newState.animaT5EncoderModel = oldState.animaT5EncoderModel;
  newState.kleinVaeModel = oldState.kleinVaeModel;
  newState.kleinQwen3EncoderModel = oldState.kleinQwen3EncoderModel;
  newState.qwenImageComponentSource = oldState.qwenImageComponentSource;
  newState.qwenImageVaeModel = oldState.qwenImageVaeModel;
  newState.qwenImageQwenVLEncoderModel = oldState.qwenImageQwenVLEncoderModel;
  newState.qwenImageQuantization = oldState.qwenImageQuantization;
  newState.qwenImageShift = oldState.qwenImageShift;
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
  setFluxScheduler,
  setFluxDypePreset,
  setFluxDypeScale,
  setFluxDypeExponent,
  setZImageScheduler,
  setZImageShift,
  setZImageSeedVarianceEnabled,
  setZImageSeedVarianceStrength,
  setZImageSeedVarianceRandomizePercent,
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
  zImageVaeModelSelected,
  zImageQwen3EncoderModelSelected,
  zImageQwen3SourceModelSelected,
  kleinVaeModelSelected,
  kleinQwen3EncoderModelSelected,
  qwenImageComponentSourceSelected,
  qwenImageVaeModelSelected,
  qwenImageQwenVLEncoderModelSelected,
  qwenImageQuantizationChanged,
  qwenImageShiftChanged,
  setClipSkip,
  shouldUseCpuNoiseChanged,
  setColorCompensation,
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
  modelChanged,

  // Dimensions
  sizeRecalled,
  widthChanged,
  heightChanged,
  aspectRatioLockToggled,
  aspectRatioIdChanged,
  dimensionsSwapped,
  sizeOptimized,
  syncedToOptimalDimension,

  resolutionPresetSelected,
  imageSizeChanged,
  paramsReset,
  openaiQualityChanged,
  openaiBackgroundChanged,
  openaiInputFidelityChanged,
  geminiTemperatureChanged,
  geminiThinkingLevelChanged,
  seedreamWatermarkChanged,
  seedreamOptimizePromptChanged,
  paramsRecalled,
  animaVaeModelSelected,
  animaQwen3EncoderModelSelected,
  animaT5EncoderModelSelected,
  setAnimaScheduler,
} = slice.actions;

export const paramsSliceConfig: SliceConfig<typeof slice> = {
  slice,
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
        // v2 -> v3, add standalone Qwen Image VAE and Qwen VL encoder fields
        state._version = 3;
        state.qwenImageVaeModel = null;
        state.qwenImageQwenVLEncoderModel = null;
      }

      return zParamsState.parse(state);
    },
  },
};

export const selectParamsSlice = (state: RootState) => state.params;
const createParamsSelector = <T>(selector: Selector<ParamsState, T>) => createSelector(selectParamsSlice, selector);

export const selectBase = createParamsSelector((params) => params.model?.base);
export const selectIsSDXL = createParamsSelector((params) => params.model?.base === 'sdxl');
export const selectIsFLUX = createParamsSelector((params) => params.model?.base === 'flux');
export const selectIsSD3 = createParamsSelector((params) => params.model?.base === 'sd-3');
export const selectIsCogView4 = createParamsSelector((params) => params.model?.base === 'cogview4');
export const selectIsZImage = createParamsSelector((params) => params.model?.base === 'z-image');
export const selectIsAnima = createParamsSelector((params) => params.model?.base === 'anima');
export const selectIsFlux2 = createParamsSelector((params) => params.model?.base === 'flux2');
export const selectIsExternal = createParamsSelector((params) => params.model?.base === 'external');
export const selectIsQwenImage = createParamsSelector((params) => params.model?.base === 'qwen-image');
export const selectIsFluxKontext = createParamsSelector((params) => {
  if (params.model?.base === 'flux' && params.model?.name.toLowerCase().includes('kontext')) {
    return true;
  }
  return false;
});

export const selectModel = createParamsSelector((params) => params.model);
export const selectModelKey = createParamsSelector((params) => params.model?.key);
export const selectVAE = createParamsSelector((params) => params.vae);
export const selectFLUXVAE = createParamsSelector((params) => params.fluxVAE);
export const selectVAEKey = createParamsSelector((params) => params.vae?.key);
export const selectT5EncoderModel = createParamsSelector((params) => params.t5EncoderModel);
export const selectCLIPEmbedModel = createParamsSelector((params) => params.clipEmbedModel);
export const selectCLIPLEmbedModel = createParamsSelector((params) => params.clipLEmbedModel);

export const selectCLIPGEmbedModel = createParamsSelector((params) => params.clipGEmbedModel);
export const selectZImageVaeModel = createParamsSelector((params) => params.zImageVaeModel);
export const selectZImageQwen3EncoderModel = createParamsSelector((params) => params.zImageQwen3EncoderModel);
export const selectZImageQwen3SourceModel = createParamsSelector((params) => params.zImageQwen3SourceModel);
export const selectAnimaVaeModel = createParamsSelector((params) => params.animaVaeModel);
export const selectAnimaQwen3EncoderModel = createParamsSelector((params) => params.animaQwen3EncoderModel);
export const selectAnimaT5EncoderModel = createParamsSelector((params) => params.animaT5EncoderModel);
export const selectAnimaScheduler = createParamsSelector((params) => params.animaScheduler);
export const selectKleinVaeModel = createParamsSelector((params) => params.kleinVaeModel);
export const selectKleinQwen3EncoderModel = createParamsSelector((params) => params.kleinQwen3EncoderModel);
export const selectQwenImageComponentSource = createParamsSelector((params) => params.qwenImageComponentSource);
export const selectQwenImageVaeModel = createParamsSelector((params) => params.qwenImageVaeModel);
export const selectQwenImageQwenVLEncoderModel = createParamsSelector((params) => params.qwenImageQwenVLEncoderModel);
export const selectQwenImageQuantization = createParamsSelector((params) => params.qwenImageQuantization);
export const selectQwenImageShift = createParamsSelector((params) => params.qwenImageShift);

export const selectCFGScale = createParamsSelector((params) => params.cfgScale);
export const selectGuidance = createParamsSelector((params) => params.guidance);
export const selectSteps = createParamsSelector((params) => params.steps);
export const selectCFGRescaleMultiplier = createParamsSelector((params) => params.cfgRescaleMultiplier);
export const selectCLIPSkip = createParamsSelector((params) => params.clipSkip);
export const selectHasModelCLIPSkip = createParamsSelector((params) => hasModelClipSkip(params.model));
export const selectCanvasCoherenceEdgeSize = createParamsSelector((params) => params.canvasCoherenceEdgeSize);
export const selectCanvasCoherenceMinDenoise = createParamsSelector((params) => params.canvasCoherenceMinDenoise);
export const selectCanvasCoherenceMode = createParamsSelector((params) => params.canvasCoherenceMode);
export const selectMaskBlur = createParamsSelector((params) => params.maskBlur);
export const selectInfillMethod = createParamsSelector((params) => params.infillMethod);
export const selectInfillTileSize = createParamsSelector((params) => params.infillTileSize);
export const selectInfillPatchmatchDownscaleSize = createParamsSelector(
  (params) => params.infillPatchmatchDownscaleSize
);
export const selectInfillColorValue = createParamsSelector((params) => params.infillColorValue);
export const selectImg2imgStrength = createParamsSelector((params) => params.img2imgStrength);
export const selectOptimizedDenoisingEnabled = createParamsSelector((params) => params.optimizedDenoisingEnabled);
export const selectPositivePrompt = createParamsSelector((params) => params.positivePrompt);
export const selectNegativePrompt = createParamsSelector((params) => params.negativePrompt);
export const selectNegativePromptWithFallback = createParamsSelector((params) => params.negativePrompt ?? '');
export const selectModelConfig = createSelector(
  selectModelConfigsQuery,
  selectParamsSlice,
  (modelConfigs, { model }) => {
    if (!modelConfigs.data) {
      return null;
    }
    if (!model) {
      return null;
    }
    return (
      (modelConfigsAdapterSelectors.selectById(modelConfigs.data, model.key) as
        | AnyModelConfigWithExternal
        | undefined) ?? null
    );
  }
);
export const selectHasNegativePrompt = createParamsSelector((params) => params.negativePrompt !== null);
export const selectModelSupportsNegativePrompt = createSelector(selectModel, (model) => {
  if (!model) {
    return false;
  }
  if (model.base === 'external') {
    return false;
  }
  return SUPPORTS_NEGATIVE_PROMPT_BASE_MODELS.includes(model.base);
});
export const selectModelSupportsRefImages = createSelector(selectModel, selectModelConfig, (model, modelConfig) => {
  if (!model) {
    return false;
  }
  if (modelConfig && isExternalApiModelConfig(modelConfig)) {
    return hasExternalPanelControl(modelConfig, 'prompts', 'reference_images');
  }
  if (model.base === 'external') {
    return false;
  }
  return SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base);
});
export const selectModelSupportsOptimizedDenoising = createSelector(
  selectModel,
  (model) => !!model && model.base !== 'external' && SUPPORTS_OPTIMIZED_DENOISING_BASE_MODELS.includes(model.base)
);
export const selectModelSupportsGuidance = createSelector(selectModel, (model) => {
  if (!model) {
    return false;
  }
  if (model.base === 'external') {
    return false;
  }
  return true;
});
export const selectModelSupportsSeed = createSelector(selectModel, selectModelConfig, (model, modelConfig) => {
  if (!model) {
    return false;
  }
  if (modelConfig && isExternalApiModelConfig(modelConfig)) {
    return hasExternalPanelControl(modelConfig, 'image', 'seed');
  }
  return true;
});
export const selectModelSupportsSteps = createSelector(selectModel, (model) => {
  if (!model) {
    return false;
  }
  if (model.base === 'external') {
    return false;
  }
  return true;
});
export const selectModelSupportsDimensions = createSelector(selectModel, selectModelConfig, (model, modelConfig) => {
  if (!model) {
    return false;
  }
  if (modelConfig && isExternalApiModelConfig(modelConfig)) {
    return hasExternalPanelControl(modelConfig, 'image', 'dimensions');
  }
  return true;
});
export const selectSeedControl = createSelector(selectModelConfig, (modelConfig) => {
  if (modelConfig && isExternalApiModelConfig(modelConfig)) {
    return getExternalPanelControl(modelConfig, 'image', 'seed');
  }
  return null;
});
export const selectScheduler = createParamsSelector((params) => params.scheduler);
export const selectFluxScheduler = createParamsSelector((params) => params.fluxScheduler);
export const selectFluxDypePreset = createParamsSelector((params) => params.fluxDypePreset);
export const selectFluxDypeScale = createParamsSelector((params) => params.fluxDypeScale);
export const selectFluxDypeExponent = createParamsSelector((params) => params.fluxDypeExponent);
export const selectZImageScheduler = createParamsSelector((params) => params.zImageScheduler);
export const selectZImageShift = createParamsSelector((params) => params.zImageShift);
export const selectZImageSeedVarianceEnabled = createParamsSelector((params) => params.zImageSeedVarianceEnabled);
export const selectZImageSeedVarianceStrength = createParamsSelector((params) => params.zImageSeedVarianceStrength);
export const selectZImageSeedVarianceRandomizePercent = createParamsSelector(
  (params) => params.zImageSeedVarianceRandomizePercent
);
export const selectSeamlessXAxis = createParamsSelector((params) => params.seamlessXAxis);
export const selectSeamlessYAxis = createParamsSelector((params) => params.seamlessYAxis);
export const selectSeed = createParamsSelector((params) => params.seed);
export const selectShouldRandomizeSeed = createParamsSelector((params) => params.shouldRandomizeSeed);
export const selectVAEPrecision = createParamsSelector((params) => params.vaePrecision);
export const selectIterations = createParamsSelector((params) => params.iterations);
export const selectShouldUseCPUNoise = createParamsSelector((params) => params.shouldUseCpuNoise);
export const selectColorCompensation = createParamsSelector((params) => params.colorCompensation);

export const selectUpscaleScheduler = createParamsSelector((params) => params.upscaleScheduler);
export const selectUpscaleCfgScale = createParamsSelector((params) => params.upscaleCfgScale);

export const selectPositivePromptHistory = createParamsSelector((params) => params.positivePromptHistory);
export const selectRefinerCFGScale = createParamsSelector((params) => params.refinerCFGScale);
export const selectRefinerModel = createParamsSelector((params) => params.refinerModel);
export const selectIsRefinerModelSelected = createParamsSelector((params) => Boolean(params.refinerModel));
export const selectRefinerPositiveAestheticScore = createParamsSelector(
  (params) => params.refinerPositiveAestheticScore
);
export const selectRefinerNegativeAestheticScore = createParamsSelector(
  (params) => params.refinerNegativeAestheticScore
);
export const selectRefinerScheduler = createParamsSelector((params) => params.refinerScheduler);
export const selectRefinerStart = createParamsSelector((params) => params.refinerStart);
export const selectRefinerSteps = createParamsSelector((params) => params.refinerSteps);

export const selectWidth = createParamsSelector((params) => params.dimensions.width);
export const selectHeight = createParamsSelector((params) => params.dimensions.height);
export const selectAspectRatioID = createParamsSelector((params) => params.dimensions.aspectRatio.id);
export const selectAspectRatioValue = createParamsSelector((params) => params.dimensions.aspectRatio.value);
export const selectAspectRatioIsLocked = createParamsSelector((params) => params.dimensions.aspectRatio.isLocked);
export const selectAllowedAspectRatioIDs = createSelector(selectModelConfig, (modelConfig) => {
  if (!modelConfig || !isExternalApiModelConfig(modelConfig)) {
    return null;
  }
  const allowed = modelConfig.capabilities.allowed_aspect_ratios;
  return allowed?.length ? allowed : null;
});
export const selectAspectRatioSizes = createSelector(selectModelConfig, (modelConfig) => {
  if (!modelConfig || !isExternalApiModelConfig(modelConfig)) {
    return null;
  }
  return modelConfig.capabilities.aspect_ratio_sizes ?? null;
});
export const selectResolutionPresets = createSelector(selectModelConfig, (modelConfig) => {
  if (!modelConfig || !isExternalApiModelConfig(modelConfig)) {
    return null;
  }
  return modelConfig.capabilities.resolution_presets ?? null;
});
export const selectHasFixedDimensionSizes = createSelector(
  selectAspectRatioSizes,
  selectResolutionPresets,
  (sizes, presets) => sizes !== null || (presets !== null && presets.length > 0)
);
export const selectImageSize = createParamsSelector((params) => params.imageSize);
export const selectOpenaiQuality = createParamsSelector((params) => params.openaiQuality);
export const selectOpenaiBackground = createParamsSelector((params) => params.openaiBackground);
export const selectOpenaiInputFidelity = createParamsSelector((params) => params.openaiInputFidelity);
export const selectGeminiTemperature = createParamsSelector((params) => params.geminiTemperature);
export const selectGeminiThinkingLevel = createParamsSelector((params) => params.geminiThinkingLevel);
export const selectSeedreamWatermark = createParamsSelector((params) => params.seedreamWatermark);
export const selectSeedreamOptimizePrompt = createParamsSelector((params) => params.seedreamOptimizePrompt);
export const selectExternalProviderId = createSelector(selectModelConfig, (modelConfig) => {
  if (modelConfig && isExternalApiModelConfig(modelConfig)) {
    return modelConfig.provider_id;
  }
  return null;
});

export const selectMainModelConfig = createSelector(selectModelConfig, (modelConfig) => {
  if (!modelConfig) {
    return null;
  }
  if (isExternalApiModelConfig(modelConfig)) {
    return modelConfig;
  }
  if (!isNonRefinerMainModelConfig(modelConfig)) {
    return null;
  }
  return modelConfig;
});
