import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import { isPlainObject } from 'es-toolkit';
import { clamp } from 'es-toolkit/compat';
import { logout } from 'features/auth/store/authSlice';
import type {
  AspectRatioID,
  InfillMethod,
  ParamsState,
  PidMode,
  PromptHistoryItem,
  RgbaColor,
} from 'features/controlLayers/store/types';
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
import type { BaseModelType, ModelIdentifierField } from 'features/nodes/types/common';
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
  ParameterIdeogram4SamplerPreset,
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
import {
  getGridSize,
  getIsSizeOptimal,
  getOptimalDimension,
  getPidScale,
} from 'features/parameters/util/optimalDimension';
import { getPidDecoderBaseForMainBase } from 'features/parameters/util/pid';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfigWithExternal } from 'services/api/types';
import { isExternalApiModelConfig, isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

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
    setErnieImageScheduler: (state, action: PayloadAction<'euler' | 'heun' | 'lcm'>) => {
      state.ernieImageScheduler = action.payload;
    },
    setErnieImageUsePromptEnhancer: (state, action: PayloadAction<boolean>) => {
      state.ernieImageUsePromptEnhancer = action.payload;
    },
    setZImageShift: (state, action: PayloadAction<number | null>) => {
      state.zImageShift = action.payload;
    },
    setIdeogram4SamplerPreset: (state, action: PayloadAction<ParameterIdeogram4SamplerPreset>) => {
      state.ideogram4SamplerPreset = action.payload;
    },
    setIdeogram4Steps: (state, action: PayloadAction<number | null>) => {
      // Normalize through the schema so a stale/out-of-range value (e.g. 1, below the backend's min of 2)
      // becomes null (= use preset) rather than being dispatched straight into the graph.
      state.ideogram4Steps = zParamsState.shape.ideogram4Steps.parse(action.payload);
    },
    setIdeogram4GuidanceScale: (state, action: PayloadAction<number | null>) => {
      state.ideogram4GuidanceScale = action.payload;
    },
    setIdeogram4Mu: (state, action: PayloadAction<number | null>) => {
      state.ideogram4Mu = action.payload;
    },
    setIdeogram4ColorPalette: (state, action: PayloadAction<string[]>) => {
      state.ideogram4ColorPalette = action.payload;
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
    setKrea2SeedVarianceEnabled: (state, action: PayloadAction<boolean>) => {
      state.krea2SeedVarianceEnabled = action.payload;
    },
    setKrea2SeedVarianceStrength: (state, action: PayloadAction<number>) => {
      state.krea2SeedVarianceStrength = action.payload;
    },
    setKrea2SeedVarianceRandomizePercent: (state, action: PayloadAction<number>) => {
      state.krea2SeedVarianceRandomizePercent = action.payload;
    },
    setKrea2RebalanceEnabled: (state, action: PayloadAction<boolean>) => {
      state.krea2RebalanceEnabled = action.payload;
    },
    setKrea2RebalanceMultiplier: (state, action: PayloadAction<number>) => {
      state.krea2RebalanceMultiplier = action.payload;
    },
    setKrea2RebalanceWeights: (state, action: PayloadAction<string>) => {
      state.krea2RebalanceWeights = action.payload;
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
    setHiDiffusionEnabled: (state, action: PayloadAction<boolean>) => {
      state.hiDiffusionEnabled = action.payload;
    },
    setHiDiffusionRauNetEnabled: (state, action: PayloadAction<boolean>) => {
      state.hiDiffusionRauNetEnabled = action.payload;
    },
    setHiDiffusionWindowAttnEnabled: (state, action: PayloadAction<boolean>) => {
      state.hiDiffusionWindowAttnEnabled = action.payload;
    },
    setHiDiffusionT1Ratio: (state, action: PayloadAction<number>) => {
      state.hiDiffusionT1Ratio = action.payload;
    },
    setHiDiffusionT2Ratio: (state, action: PayloadAction<number>) => {
      state.hiDiffusionT2Ratio = action.payload;
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

      // PiD decoders are trained per backbone, so a decoder selected for the old base is invalid for a
      // different one. Clear it unless the new base maps to the same decoder base (e.g. Z-Image reuses the
      // FLUX decoder), so enqueue requires a matching decoder instead of failing during PiD execution.
      const pidDecoderBase = getPidDecoderBaseForMainBase(model.base);
      if (state.pidDecoderModel && state.pidDecoderModel.base !== pidDecoderBase) {
        state.pidDecoderModel = null;
      }
      // If the new base cannot do PiD at all, turn PiD off. A stuck `native` mode is hidden by the UI
      // (PidSettings is gated on the base) but keeps warping dimensions via getPidScale (4x grid / 2048
      // optimum) with no way to disable it. Re-fit dimensions to the plain scale on the new base.
      if (pidDecoderBase === null && state.pidMode !== 'off') {
        const prevPidScale = getPidScale(state.pidMode);
        state.pidMode = 'off';
        const nextPidScale = getPidScale('off');
        if (prevPidScale !== nextPidScale) {
          const optimalDimension = getOptimalDimension(model.base, nextPidScale);
          const { width, height } = calculateNewSize(
            state.dimensions.aspectRatio.value,
            optimalDimension * optimalDimension,
            model.base,
            nextPidScale
          );
          state.dimensions.width = width;
          state.dimensions.height = height;
        }
      }
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
    krea2VaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.krea2VaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.krea2VaeModel = result.data;
    },
    krea2Qwen3VlEncoderModelSelected: (
      state,
      action: PayloadAction<{ key: string; name: string; base: string } | null>
    ) => {
      const result = zParamsState.shape.krea2Qwen3VlEncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.krea2Qwen3VlEncoderModel = result.data;
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
    animaLLLiteModelSelected: (state, action: PayloadAction<ModelIdentifierField | null>) => {
      const result = zParamsState.shape.animaLLLiteModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.animaLLLiteModel = result.data;
    },
    animaLLLiteWeightChanged: (state, action: PayloadAction<number>) => {
      const result = zParamsState.shape.animaLLLiteWeight.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.animaLLLiteWeight = result.data;
    },
    setAnimaScheduler: (
      state,
      action: PayloadAction<'euler' | 'heun' | 'dpmpp_2m' | 'dpmpp_2m_sde' | 'er_sde' | 'lcm'>
    ) => {
      state.animaScheduler = action.payload;
    },
    flux2VaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.flux2VaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.flux2VaeModel = result.data;
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
    flux2DevMistralEncoderModelSelected: (
      state,
      action: PayloadAction<{ key: string; name: string; base: string } | null>
    ) => {
      const result = zParamsState.shape.flux2DevMistralEncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.flux2DevMistralEncoderModel = result.data;
    },
    pidModeChanged: (state, action: PayloadAction<PidMode>) => {
      const prevPidScale = getPidScale(state.pidMode);
      const nextPidScale = getPidScale(action.payload);
      state.pidMode = action.payload;
      // Entering/leaving native mode reinterprets the dimensions (4x target <-> generation resolution), so
      // re-fit them to the new mode's optimal target on the new grid, preserving aspect ratio.
      if (prevPidScale !== nextPidScale) {
        const base = state.model?.base as BaseModelType | undefined;
        const optimalDimension = getOptimalDimension(base, nextPidScale);
        const { width, height } = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          base,
          nextPidScale
        );
        state.dimensions.width = width;
        state.dimensions.height = height;
      }
    },
    pidDecoderModelSelected: (state, action: PayloadAction<{ key: string; name: string; base: string } | null>) => {
      const result = zParamsState.shape.pidDecoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.pidDecoderModel = result.data;
    },
    gemma2EncoderModelSelected: (state, action: PayloadAction<{ key: string; name: string; base: string } | null>) => {
      const result = zParamsState.shape.gemma2EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.gemma2EncoderModel = result.data;
    },
    pidStepsChanged: (state, action: PayloadAction<number>) => {
      const result = zParamsState.shape.pidSteps.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.pidSteps = result.data;
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
    wanTransformerLowNoiseSelected: (state, action: PayloadAction<ParameterModel | null>) => {
      const result = zParamsState.shape.wanTransformerLowNoise.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.wanTransformerLowNoise = result.data;
    },
    wanComponentSourceSelected: (state, action: PayloadAction<ParameterModel | null>) => {
      const result = zParamsState.shape.wanComponentSource.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.wanComponentSource = result.data;
    },
    wanVaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      const result = zParamsState.shape.wanVaeModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.wanVaeModel = result.data;
    },
    wanT5EncoderModelSelected: (state, action: PayloadAction<{ key: string; name: string; base: string } | null>) => {
      const result = zParamsState.shape.wanT5EncoderModel.safeParse(action.payload);
      if (!result.success) {
        return;
      }
      state.wanT5EncoderModel = result.data;
    },
    wanGuidanceScaleLowNoiseChanged: (state, action: PayloadAction<number | null>) => {
      state.wanGuidanceScaleLowNoise = action.payload;
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
    positivePromptAddedToHistory: (state, action: PayloadAction<PromptHistoryItem>) => {
      const prompt: PromptHistoryItem = {
        positivePrompt: action.payload.positivePrompt.trim(),
        negativePrompt: action.payload.negativePrompt?.trim() || null,
      };
      if (prompt.positivePrompt.length === 0 && !prompt.negativePrompt) {
        return;
      }

      state.positivePromptHistory = [
        prompt,
        ...state.positivePromptHistory.filter(
          (p) =>
            p.positivePrompt !== prompt.positivePrompt || (p.negativePrompt ?? null) !== (prompt.negativePrompt ?? null)
        ),
      ];

      if (state.positivePromptHistory.length > MAX_POSITIVE_PROMPT_HISTORY) {
        state.positivePromptHistory = state.positivePromptHistory.slice(0, MAX_POSITIVE_PROMPT_HISTORY);
      }
    },
    promptRemovedFromHistory: (state, action: PayloadAction<PromptHistoryItem>) => {
      state.positivePromptHistory = state.positivePromptHistory.filter(
        (p) =>
          p.positivePrompt !== action.payload.positivePrompt ||
          (p.negativePrompt ?? null) !== (action.payload.negativePrompt ?? null)
      );
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
      const gridSize = getGridSize(state.model?.base as BaseModelType | undefined, getPidScale(state.pidMode));
      state.dimensions.width = Math.max(roundDownToMultiple(width, gridSize), 64);
      state.dimensions.height = Math.max(roundDownToMultiple(height, gridSize), 64);
      state.dimensions.aspectRatio.value = state.dimensions.width / state.dimensions.height;
      state.dimensions.aspectRatio.id = 'Free';
      state.dimensions.aspectRatio.isLocked = true;
    },
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      const gridSize = getGridSize(state.model?.base as BaseModelType | undefined, getPidScale(state.pidMode));
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
      const gridSize = getGridSize(state.model?.base as BaseModelType | undefined, getPidScale(state.pidMode));
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
            state.model?.base as BaseModelType | undefined,
            getPidScale(state.pidMode)
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
          state.model?.base as BaseModelType | undefined,
          getPidScale(state.pidMode)
        );
        state.dimensions.width = width;
        state.dimensions.height = height;
        state.dimensions.aspectRatio.id = ASPECT_RATIO_MAP[state.dimensions.aspectRatio.id].inverseID;
      }
    },
    sizeOptimized: (state) => {
      const pidScale = getPidScale(state.pidMode);
      const optimalDimension = getOptimalDimension(state.model?.base as BaseModelType | undefined, pidScale);
      if (state.dimensions.aspectRatio.isLocked) {
        const { width, height } = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.model?.base as BaseModelType | undefined,
          pidScale
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
      const pidScale = getPidScale(state.pidMode);
      const optimalDimension = getOptimalDimension(state.model?.base as BaseModelType | undefined, pidScale);

      if (
        !getIsSizeOptimal(
          state.dimensions.width,
          state.dimensions.height,
          state.model?.base as BaseModelType | undefined,
          pidScale
        )
      ) {
        const bboxDims = calculateNewSize(
          state.dimensions.aspectRatio.value,
          optimalDimension * optimalDimension,
          state.model?.base as BaseModelType | undefined,
          pidScale
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
  newState.krea2VaeModel = oldState.krea2VaeModel;
  newState.krea2Qwen3VlEncoderModel = oldState.krea2Qwen3VlEncoderModel;
  newState.animaVaeModel = oldState.animaVaeModel;
  newState.animaQwen3EncoderModel = oldState.animaQwen3EncoderModel;
  newState.animaLLLiteModel = oldState.animaLLLiteModel;
  newState.flux2VaeModel = oldState.flux2VaeModel;
  newState.kleinQwen3EncoderModel = oldState.kleinQwen3EncoderModel;
  newState.flux2DevMistralEncoderModel = oldState.flux2DevMistralEncoderModel;
  newState.pidMode = oldState.pidMode;
  newState.pidDecoderModel = oldState.pidDecoderModel;
  newState.gemma2EncoderModel = oldState.gemma2EncoderModel;
  newState.pidSteps = oldState.pidSteps;
  newState.qwenImageComponentSource = oldState.qwenImageComponentSource;
  newState.qwenImageVaeModel = oldState.qwenImageVaeModel;
  newState.qwenImageQwenVLEncoderModel = oldState.qwenImageQwenVLEncoderModel;
  newState.qwenImageQuantization = oldState.qwenImageQuantization;
  newState.qwenImageShift = oldState.qwenImageShift;
  newState.wanTransformerLowNoise = oldState.wanTransformerLowNoise;
  newState.wanComponentSource = oldState.wanComponentSource;
  newState.wanVaeModel = oldState.wanVaeModel;
  newState.wanT5EncoderModel = oldState.wanT5EncoderModel;
  newState.wanGuidanceScaleLowNoise = oldState.wanGuidanceScaleLowNoise;
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
  setErnieImageScheduler,
  setErnieImageUsePromptEnhancer,
  setZImageShift,
  setIdeogram4SamplerPreset,
  setIdeogram4Steps,
  setIdeogram4GuidanceScale,
  setIdeogram4Mu,
  setIdeogram4ColorPalette,
  setZImageSeedVarianceEnabled,
  setZImageSeedVarianceStrength,
  setZImageSeedVarianceRandomizePercent,
  setKrea2SeedVarianceEnabled,
  setKrea2SeedVarianceStrength,
  setKrea2SeedVarianceRandomizePercent,
  setKrea2RebalanceEnabled,
  setKrea2RebalanceMultiplier,
  setKrea2RebalanceWeights,
  setUpscaleScheduler,
  setUpscaleCfgScale,
  setSeed,
  setImg2imgStrength,
  setOptimizedDenoisingEnabled,
  setHiDiffusionEnabled,
  setHiDiffusionRauNetEnabled,
  setHiDiffusionWindowAttnEnabled,
  setHiDiffusionT1Ratio,
  setHiDiffusionT2Ratio,
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
  flux2VaeModelSelected,
  krea2VaeModelSelected,
  krea2Qwen3VlEncoderModelSelected,
  kleinQwen3EncoderModelSelected,
  flux2DevMistralEncoderModelSelected,
  pidModeChanged,
  pidDecoderModelSelected,
  gemma2EncoderModelSelected,
  pidStepsChanged,
  qwenImageComponentSourceSelected,
  qwenImageVaeModelSelected,
  qwenImageQwenVLEncoderModelSelected,
  qwenImageQuantizationChanged,
  qwenImageShiftChanged,
  wanTransformerLowNoiseSelected,
  wanComponentSourceSelected,
  wanVaeModelSelected,
  wanT5EncoderModelSelected,
  wanGuidanceScaleLowNoiseChanged,
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
  animaLLLiteModelSelected,
  animaLLLiteWeightChanged,
  setAnimaScheduler,
} = slice.actions;

/**
 * Last-resort repair for the persisted params slice, applied after the version steps have run.
 *
 * The `zParamsState.parse()` at the end of `migrate()` is all-or-nothing: a single missing required
 * key throws, and the caller in `store.ts` catches it and falls back to the initial state — silently
 * wiping every generation param the user had (prompts, prompt history, model selection, dimensions).
 * That has happened whenever a key was added to the schema with neither a `.default()` nor a seed in
 * the migration chain, which is what the Wan and post-v3 seeds above exist to undo.
 *
 * Repairing the offending key turns that into "one field sits at its default" instead of "the user
 * lost everything". Two kinds of damage are repaired, both per key:
 *   - `backfilled`: the key is *absent* and the schema cannot fill it itself. Anything with
 *     `.default()` / `.catch()` / `.optional()` is left to zod, so the schema's own default stays
 *     authoritative.
 *   - `reset`: the key is present but its value does not satisfy that field's schema. Whatever the
 *     cause — a tightened field schema, a hand-edited blob, a half-applied migration — resetting the
 *     one field is strictly better than the alternative, which is `store.ts` discarding all of them.
 *     Note the granularity is one top-level key, so this is not always cheap: one malformed entry in
 *     `positivePromptHistory` costs the whole history, and a `model` whose `base` has since been
 *     removed from `zBaseModelType` (the external-API bases dropped in v6.9.0rc1, say) clears the
 *     user's model selection. Both are still a single field rather than every field.
 *
 * This is a safety net, not a substitute for a migration step — it fills fields with *today's*
 * initial value, which is only the right answer for genuinely new fields. `paramsSlice.test.ts`
 * asserts nothing needs repairing for real persisted blobs, so a forgotten seed still fails CI.
 *
 * Exported for that test.
 */
export const repairParamsState = (state: Record<string, unknown>): { backfilled: string[]; reset: string[] } => {
  const initial = getInitialParamsState() as unknown as Record<string, unknown>;
  const backfilled: string[] = [];
  const reset: string[] = [];

  for (const [key, fieldSchema] of Object.entries(zParamsState.shape)) {
    // Never touch `_version`: it is the input to the version steps, so repairing it would stamp a
    // blob as current having run no step at all. A `_version` from the future (a downgrade) is
    // deliberately still fatal — that slice really was written by a newer schema.
    if (key === '_version') {
      continue;
    }

    if (state[key] === undefined) {
      // `undefined` is the only value that counts as missing: persisted JSON can't hold it, and
      // every nullable field in the schema uses `null` for "unset".
      if (!fieldSchema.safeParse(undefined).success) {
        state[key] = initial[key];
        backfilled.push(key);
      }
      continue;
    }

    if (!fieldSchema.safeParse(state[key]).success) {
      state[key] = initial[key];
      reset.push(key);
    }
  }

  return { backfilled, reset };
};

/**
 * Bring a persisted params blob up to the current `_version` in place.
 *
 * Every key added to `zParamsState` without a zod default must be seeded by the step for the version
 * that predates it, or upgrading users lose the whole slice — see `repairParamsState`. Note that
 * the *current* version has no step by definition, so a key added after the last bump needs a zod
 * default (as the ERNIE-Image and PiD fields have) rather than a seed.
 *
 * Exported so the tests can assert the version steps alone are complete, without the safety net
 * hiding a missing seed.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const applyParamsVersionMigrations = (state: any): void => {
  // Value check rather than `!('_version' in state)`. The two are equivalent on real input, since
  // the only production caller feeds this `JSON.parse` output, which cannot produce an explicit
  // `undefined` — but a value check is what the rest of this file uses, and with a presence check a
  // blob carrying `_version: undefined` would match no branch at all, reach the parse and take the
  // whole slice down.
  if (state._version === undefined) {
    // v0 -> v1, add _version and remove x/y from dimensions, lifting width/height to top level.
    // `dimensions.rect` is guarded: a truncated or hand-edited blob that lacks it would otherwise
    // throw a TypeError out of migrate() and cost the user the whole slice. What that leaves behind
    // — a `dimensions` object with no width/height — fails its own field schema, so
    // repairParamsState() swaps in the initial dimensions instead of letting the parse wipe
    // everything else along with it.
    state._version = 1;
    if (state.dimensions === undefined) {
      // The oldest v0 builds (v6.0.0a1 - v6.0.0rc3) had no `dimensions` key at all; it arrives in
      // v6.0.0rc4. Seeding it here rather than leaving it to the repair pass is what keeps the
      // version steps self-sufficient for the v0 tier.
      state.dimensions = getInitialParamsState().dimensions;
    } else if (state.dimensions && state.dimensions.rect) {
      state.dimensions.width = state.dimensions.rect.width;
      state.dimensions.height = state.dimensions.rect.height;
    }
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

    // Everything below was added to the schema while releases were still persisting v2 blobs
    // (v6.7.0 - v6.12.0), but without a version bump. None has a zod default, which makes them
    // required, so their absence fails the parse() at the end of migrate() and silently wipes the
    // whole slice on upgrade. Seed only when missing so that v2 blobs written by dev builds after
    // each field landed keep the values they already hold.
    //
    // The oldest v2 releases (v6.7.0 - v6.9.0) are missing these six as well as everything below.
    state.fluxScheduler = state.fluxScheduler ?? 'euler';
    state.zImageScheduler = state.zImageScheduler ?? 'euler';
    state.colorCompensation = state.colorCompensation ?? false;
    state.zImageVaeModel = state.zImageVaeModel ?? null;
    state.zImageQwen3EncoderModel = state.zImageQwen3EncoderModel ?? null;
    state.zImageQwen3SourceModel = state.zImageQwen3SourceModel ?? null;
    // Added by v6.10.0 - v6.12.0 and later.
    state.fluxDypePreset = state.fluxDypePreset ?? 'off';
    state.fluxDypeScale = state.fluxDypeScale ?? 2.0;
    state.fluxDypeExponent = state.fluxDypeExponent ?? 2.0;
    state.zImageShift = state.zImageShift ?? null;
    state.zImageSeedVarianceEnabled = state.zImageSeedVarianceEnabled ?? false;
    state.zImageSeedVarianceStrength = state.zImageSeedVarianceStrength ?? 0.1;
    state.zImageSeedVarianceRandomizePercent = state.zImageSeedVarianceRandomizePercent ?? 50;
    state.animaVaeModel = state.animaVaeModel ?? null;
    state.animaQwen3EncoderModel = state.animaQwen3EncoderModel ?? null;
    state.animaScheduler = state.animaScheduler ?? 'euler';
    // No `kleinVaeModel` seed: the v4 -> v5 step below folds that slot into `flux2VaeModel` and
    // deletes it, so it is no longer part of the schema and needs nothing here.
    state.kleinQwen3EncoderModel = state.kleinQwen3EncoderModel ?? null;
    state.qwenImageComponentSource = state.qwenImageComponentSource ?? null;
    state.qwenImageQuantization = state.qwenImageQuantization ?? 'none';
    state.qwenImageShift = state.qwenImageShift ?? null;
  }

  if (state._version === 3) {
    // v3 -> v4, add Krea-2 standalone component and conditioning enhancer fields, and the
    // PiD (Pixel Diffusion Decoder) fields. Also seed the Wan component fields — they were
    // added to the schema without a version bump while releases were still writing v3 blobs,
    // and they're nullable with no default, so a genuine released-build (v6.13.x) v3 blob
    // without them fails zParamsState.parse() below, which wipes the whole slice on upgrade.
    // Seed only when missing: dev-build v3 blobs written after the Wan merge already carry
    // (possibly non-null) values.
    state._version = 4;
    state.krea2VaeModel = null;
    state.krea2Qwen3VlEncoderModel = null;
    state.krea2SeedVarianceEnabled = false;
    state.krea2SeedVarianceStrength = 0.1;
    state.krea2SeedVarianceRandomizePercent = 50;
    state.krea2RebalanceEnabled = false;
    state.krea2RebalanceMultiplier = 4;
    state.krea2RebalanceWeights = '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0';
    state.pidMode = 'off';
    state.pidDecoderModel = null;
    state.gemma2EncoderModel = null;
    state.pidSteps = 4;
    state.wanTransformerLowNoise = state.wanTransformerLowNoise ?? null;
    state.wanComponentSource = state.wanComponentSource ?? null;
    state.wanVaeModel = state.wanVaeModel ?? null;
    state.wanT5EncoderModel = state.wanT5EncoderModel ?? null;
    state.wanGuidanceScaleLowNoise = state.wanGuidanceScaleLowNoise ?? null;
  }

  if (state._version === 4) {
    // v4 -> v5, merge the separate Klein / [dev] FLUX.2 VAE slots into one shared
    // flux2VaeModel (both drew from the same FLUX.2 VAE pool — keep whichever was set) and
    // seed the new standalone [dev] Mistral encoder slot. Both parents of the FLUX.2 [dev]
    // merge shipped incompatible schemas under _version 4 (main added the PiD fields; the
    // [dev] branch added the flux2 fields), so a v4 blob may be missing either side's keys —
    // every seed here is conditional, and the PiD keys are re-seeded for blobs written by
    // pre-merge [dev] builds. All are nullable-with-no-default, so any missing key would
    // fail zParamsState.parse() and wipe the whole slice.
    state._version = 5;
    state.flux2VaeModel = state.flux2VaeModel ?? state.kleinVaeModel ?? state.flux2DevVaeModel ?? null;
    state.flux2DevMistralEncoderModel = state.flux2DevMistralEncoderModel ?? null;
    delete state.kleinVaeModel;
    delete state.flux2DevVaeModel;
    state.pidMode = state.pidMode ?? 'off';
    state.pidDecoderModel = state.pidDecoderModel ?? null;
    state.gemma2EncoderModel = state.gemma2EncoderModel ?? null;
    state.pidSteps = state.pidSteps ?? 4;
  }

  // The HiDiffusion fields were added to the schema without a version bump, so they can be missing
  // from a blob of any version — seeded outside the version steps for that reason. They have no zod
  // default, so an absent key would fail the parse() at the end of migrate() and wipe the slice.
  if (!('hiDiffusionEnabled' in state)) {
    state.hiDiffusionEnabled = false;
  }
  if (!('hiDiffusionRauNetEnabled' in state)) {
    state.hiDiffusionRauNetEnabled = true;
  }
  if (!('hiDiffusionWindowAttnEnabled' in state)) {
    state.hiDiffusionWindowAttnEnabled = true;
  }
  if (!('hiDiffusionT1Ratio' in state)) {
    state.hiDiffusionT1Ratio = 0.4;
  }
  if (!('hiDiffusionT2Ratio' in state)) {
    state.hiDiffusionT2Ratio = 0.0;
  }
};

export const paramsSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zParamsState,
  getInitialState: getInitialParamsState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));

      applyParamsVersionMigrations(state);

      const { backfilled, reset } = repairParamsState(state);
      if (backfilled.length > 0) {
        log.warn(
          { backfilled },
          `Backfilled ${backfilled.length} params key(s) missing from the persisted state: ${backfilled.join(', ')}. ` +
            `These need a zod default or a seed in the migration chain.`
        );
      }
      if (reset.length > 0) {
        log.warn(
          { reset },
          `Reset ${reset.length} params key(s) whose persisted value no longer satisfies the schema: ${reset.join(', ')}.`
        );
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
export const selectIsIdeogram4 = createParamsSelector((params) => params.model?.base === 'ideogram-4');
export const selectIsAnima = createParamsSelector((params) => params.model?.base === 'anima');
export const selectIsFlux2 = createParamsSelector((params) => params.model?.base === 'flux2');
export const selectIsErnieImage = createParamsSelector((params) => params.model?.base === 'ernie-image');
export const selectIsExternal = createParamsSelector((params) => params.model?.base === 'external');
export const selectIsQwenImage = createParamsSelector((params) => params.model?.base === 'qwen-image');
export const selectIsKrea2 = createParamsSelector((params) => params.model?.base === 'krea-2');
export const selectIsWan = createParamsSelector((params) => params.model?.base === 'wan');
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
export const selectKrea2VaeModel = createParamsSelector((params) => params.krea2VaeModel);
export const selectKrea2Qwen3VlEncoderModel = createParamsSelector((params) => params.krea2Qwen3VlEncoderModel);
export const selectAnimaVaeModel = createParamsSelector((params) => params.animaVaeModel);
export const selectAnimaQwen3EncoderModel = createParamsSelector((params) => params.animaQwen3EncoderModel);
export const selectAnimaScheduler = createParamsSelector((params) => params.animaScheduler);
export const selectAnimaLLLiteModel = createParamsSelector((params) => params.animaLLLiteModel);
export const selectAnimaLLLiteWeight = createParamsSelector((params) => params.animaLLLiteWeight);
export const selectFlux2VaeModel = createParamsSelector((params) => params.flux2VaeModel);
export const selectKleinQwen3EncoderModel = createParamsSelector((params) => params.kleinQwen3EncoderModel);
export const selectFlux2DevMistralEncoderModel = createParamsSelector((params) => params.flux2DevMistralEncoderModel);
export const selectPidMode = createParamsSelector((params) => params.pidMode);
export const selectPidDecoderModel = createParamsSelector((params) => params.pidDecoderModel);
export const selectPidSteps = createParamsSelector((params) => params.pidSteps);
export const selectGemma2EncoderModel = createParamsSelector((params) => params.gemma2EncoderModel);
export const selectQwenImageComponentSource = createParamsSelector((params) => params.qwenImageComponentSource);
export const selectQwenImageVaeModel = createParamsSelector((params) => params.qwenImageVaeModel);
export const selectQwenImageQwenVLEncoderModel = createParamsSelector((params) => params.qwenImageQwenVLEncoderModel);
export const selectQwenImageQuantization = createParamsSelector((params) => params.qwenImageQuantization);
export const selectQwenImageShift = createParamsSelector((params) => params.qwenImageShift);
export const selectWanTransformerLowNoise = createParamsSelector((params) => params.wanTransformerLowNoise);
export const selectWanComponentSource = createParamsSelector((params) => params.wanComponentSource);
export const selectWanVaeModel = createParamsSelector((params) => params.wanVaeModel);
export const selectWanT5EncoderModel = createParamsSelector((params) => params.wanT5EncoderModel);
export const selectWanGuidanceScaleLowNoise = createParamsSelector((params) => params.wanGuidanceScaleLowNoise);

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
export const selectHiDiffusionEnabled = createParamsSelector((params) => params.hiDiffusionEnabled);
export const selectHiDiffusionRauNetEnabled = createParamsSelector((params) => params.hiDiffusionRauNetEnabled);
export const selectHiDiffusionWindowAttnEnabled = createParamsSelector((params) => params.hiDiffusionWindowAttnEnabled);
export const selectHiDiffusionT1Ratio = createParamsSelector((params) => params.hiDiffusionT1Ratio);
export const selectHiDiffusionT2Ratio = createParamsSelector((params) => params.hiDiffusionT2Ratio);
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
  if (!SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)) {
    return false;
  }
  // Wan: only the I2V variant of A14B consumes a reference image. T2V and
  // TI2V-5B ignore ref images, so hide the panel for those.
  if (model.base === 'wan') {
    const variant = modelConfig && 'variant' in modelConfig ? modelConfig.variant : null;
    return variant === 'i2v_a14b';
  }
  return true;
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
  if (model.base === 'ideogram-4') {
    // Ideogram 4 bundles step count into its sampler preset, so there is no standalone steps control.
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
export const selectErnieImageScheduler = createParamsSelector((params) => params.ernieImageScheduler);
export const selectErnieImageUsePromptEnhancer = createParamsSelector((params) => params.ernieImageUsePromptEnhancer);
export const selectZImageShift = createParamsSelector((params) => params.zImageShift);
export const selectIdeogram4SamplerPreset = createParamsSelector((params) => params.ideogram4SamplerPreset);
export const selectIdeogram4Steps = createParamsSelector((params) => params.ideogram4Steps);
export const selectIdeogram4GuidanceScale = createParamsSelector((params) => params.ideogram4GuidanceScale);
export const selectIdeogram4Mu = createParamsSelector((params) => params.ideogram4Mu);
export const selectIdeogram4ColorPalette = createParamsSelector((params) => params.ideogram4ColorPalette);
export const selectZImageSeedVarianceEnabled = createParamsSelector((params) => params.zImageSeedVarianceEnabled);
export const selectZImageSeedVarianceStrength = createParamsSelector((params) => params.zImageSeedVarianceStrength);
export const selectZImageSeedVarianceRandomizePercent = createParamsSelector(
  (params) => params.zImageSeedVarianceRandomizePercent
);
export const selectKrea2SeedVarianceEnabled = createParamsSelector((params) => params.krea2SeedVarianceEnabled);
export const selectKrea2SeedVarianceStrength = createParamsSelector((params) => params.krea2SeedVarianceStrength);
export const selectKrea2SeedVarianceRandomizePercent = createParamsSelector(
  (params) => params.krea2SeedVarianceRandomizePercent
);
export const selectKrea2RebalanceEnabled = createParamsSelector((params) => params.krea2RebalanceEnabled);
export const selectKrea2RebalanceMultiplier = createParamsSelector((params) => params.krea2RebalanceMultiplier);
export const selectKrea2RebalanceWeights = createParamsSelector((params) => params.krea2RebalanceWeights);

// The Krea-2 Conditioning Rebalance node taps exactly 12 encoder layers, so its per-layer weights string
// must be exactly 12 finite comma-separated numbers. Mirrors Krea2ConditioningRebalanceInvocation._parse_weights
// so an invalid string is blocked before it can queue a generation the backend will reject.
export const KREA2_REBALANCE_WEIGHT_COUNT = 12;

// Plain decimal / scientific-notation float, matching what Python's float() accepts. Crucially this rejects
// the hex/binary/octal literals (0x10, 0b10, 0o10) that JS Number() would happily parse but the backend
// float() rejects — which would otherwise let a graph queue that is guaranteed to fail at generation.
const KREA2_DECIMAL_NUMBER_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

export const parseKrea2RebalanceWeights = (weights: string): number[] | null => {
  const parts = weights
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s !== '');
  if (parts.length !== KREA2_REBALANCE_WEIGHT_COUNT) {
    return null;
  }
  if (!parts.every((p) => KREA2_DECIMAL_NUMBER_RE.test(p))) {
    return null;
  }
  const nums = parts.map(Number);
  if (nums.some((n) => !Number.isFinite(n))) {
    return null;
  }
  return nums;
};

export const isValidKrea2RebalanceWeights = (weights: string): boolean => parseKrea2RebalanceWeights(weights) !== null;

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

export const selectIsFlux2Dev = createSelector(selectMainModelConfig, (modelConfig) => {
  if (!modelConfig || modelConfig.base !== 'flux2') {
    return false;
  }
  return 'variant' in modelConfig && modelConfig.variant === 'dev';
});
