import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { type RgbaColor, zRgbaColor } from 'features/controlLayers/store/types';
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
  ParameterPrecision,
  ParameterScheduler,
  ParameterSDXLRefinerModel,
  ParameterT5EncoderModel,
  ParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import {
  zParameterCanvasCoherenceMode,
  zParameterCFGRescaleMultiplier,
  zParameterCFGScale,
  zParameterCLIPEmbedModel,
  zParameterCLIPGEmbedModel,
  zParameterCLIPLEmbedModel,
  zParameterControlLoRAModel,
  zParameterGuidance,
  zParameterMaskBlurMethod,
  zParameterModel,
  zParameterNegativePrompt,
  zParameterNegativeStylePromptSDXL,
  zParameterPositivePrompt,
  zParameterPositiveStylePromptSDXL,
  zParameterPrecision,
  zParameterScheduler,
  zParameterSDXLRefinerModel,
  zParameterSeed,
  zParameterSteps,
  zParameterStrength,
  zParameterT5EncoderModel,
  zParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { clamp } from 'lodash-es';
import { z } from 'zod';

import { newSessionRequested } from './actions';

const zParamsState = z.object({
  maskBlur: z.number().default(16),
  maskBlurMethod: zParameterMaskBlurMethod.default('box'),
  canvasCoherenceMode: zParameterCanvasCoherenceMode.default('Gaussian Blur'),
  canvasCoherenceMinDenoise: zParameterStrength.default(0),
  canvasCoherenceEdgeSize: z.number().default(16),
  infillMethod: z.string().default('lama'),
  infillTileSize: z.number().default(32),
  infillPatchmatchDownscaleSize: z.number().default(1),
  infillColorValue: zRgbaColor.default({ r: 0, g: 0, b: 0, a: 1 }),
  cfgScale: zParameterCFGScale.default(7.5),
  cfgRescaleMultiplier: zParameterCFGRescaleMultiplier.default(0),
  guidance: zParameterGuidance.default(4),
  img2imgStrength: zParameterStrength.default(0.75),
  optimizedDenoisingEnabled: z.boolean().default(true),
  iterations: z.number().default(1),
  scheduler: zParameterScheduler.default('dpmpp_3m_k'),
  upscaleScheduler: zParameterScheduler.default('kdpm_2'),
  upscaleCfgScale: zParameterCFGScale.default(2),
  seed: zParameterSeed.default(0),
  shouldRandomizeSeed: z.boolean().default(true),
  steps: zParameterSteps.default(30),
  model: zParameterModel.nullable().default(null),
  vae: zParameterVAEModel.nullable().default(null),
  vaePrecision: zParameterPrecision.default('fp32'),
  fluxVAE: zParameterVAEModel.nullable().default(null),
  seamlessXAxis: z.boolean().default(false),
  seamlessYAxis: z.boolean().default(false),
  clipSkip: z.number().default(0),
  shouldUseCpuNoise: z.boolean().default(true),
  positivePrompt: zParameterPositivePrompt.default(''),
  negativePrompt: zParameterNegativePrompt.default(''),
  positivePrompt2: zParameterPositiveStylePromptSDXL.default(''),
  negativePrompt2: zParameterNegativeStylePromptSDXL.default(''),
  shouldConcatPrompts: z.boolean().default(true),
  refinerModel: zParameterSDXLRefinerModel.nullable().default(null),
  refinerSteps: z.number().default(20),
  refinerCFGScale: z.number().default(7.5),
  refinerScheduler: zParameterScheduler.default('euler'),
  refinerPositiveAestheticScore: z.number().default(6),
  refinerNegativeAestheticScore: z.number().default(2.5),
  refinerStart: z.number().default(0.8),
  t5EncoderModel: zParameterT5EncoderModel.nullable().default(null),
  clipEmbedModel: zParameterCLIPEmbedModel.nullable().default(null),
  clipLEmbedModel: zParameterCLIPLEmbedModel.nullable().default(null),
  clipGEmbedModel: zParameterCLIPGEmbedModel.nullable().default(null),
  controlLora: zParameterControlLoRAModel.nullable().default(null),
});

export type ParamsState = z.infer<typeof zParamsState>;

const INITIAL_STATE = zParamsState.parse({});
const getInitialState = () => deepClone(INITIAL_STATE);

export const paramsSlice = createSlice({
  name: 'params',
  initialState: getInitialState(),
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
    modelChanged: (
      state,
      action: PayloadAction<{ model: ParameterModel | null; previousModel?: ParameterModel | null }>
    ) => {
      const { model, previousModel } = action.payload;
      state.model = model;

      // If the model base changes (e.g. SD1.5 -> SDXL), we need to change a few things
      if (model === null || previousModel?.base === model.base) {
        return;
      }

      // Clamp CLIP skip layer count to the bounds of the new model
      if (model.base === 'sdxl') {
        // We don't support user-defined CLIP skip for SDXL because it doesn't do anything useful
        state.clipSkip = 0;
      } else {
        const { maxClip } = CLIP_SKIP_MAP[model.base];
        state.clipSkip = clamp(state.clipSkip, 0, maxClip);
      }
    },
    vaeSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      // null is a valid VAE!
      state.vae = action.payload;
    },
    fluxVAESelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      state.fluxVAE = action.payload;
    },
    t5EncoderModelSelected: (state, action: PayloadAction<ParameterT5EncoderModel | null>) => {
      state.t5EncoderModel = action.payload;
    },
    controlLoRAModelSelected: (state, action: PayloadAction<ParameterControlLoRAModel | null>) => {
      state.controlLora = action.payload;
    },
    clipEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPEmbedModel | null>) => {
      state.clipEmbedModel = action.payload;
    },
    clipLEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPLEmbedModel | null>) => {
      state.clipLEmbedModel = action.payload;
    },
    clipGEmbedModelSelected: (state, action: PayloadAction<ParameterCLIPGEmbedModel | null>) => {
      state.clipGEmbedModel = action.payload;
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
    positivePromptChanged: (state, action: PayloadAction<string>) => {
      state.positivePrompt = action.payload;
    },
    negativePromptChanged: (state, action: PayloadAction<string>) => {
      state.negativePrompt = action.payload;
    },
    positivePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.positivePrompt2 = action.payload;
    },
    negativePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.negativePrompt2 = action.payload;
    },
    shouldConcatPromptsChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldConcatPrompts = action.payload;
    },
    refinerModelChanged: (state, action: PayloadAction<ParameterSDXLRefinerModel | null>) => {
      state.refinerModel = action.payload;
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
    paramsReset: (state) => resetState(state),
  },
  extraReducers(builder) {
    builder.addMatcher(newSessionRequested, (state) => resetState(state));
  },
});

const resetState = (state: ParamsState): ParamsState => {
  // When a new session is requested, we need to keep the current model selections, plus dependent state
  // like VAE precision. Everything else gets reset to default.
  const newState = getInitialState();
  newState.model = state.model;
  newState.vae = state.vae;
  newState.fluxVAE = state.fluxVAE;
  newState.vaePrecision = state.vaePrecision;
  newState.t5EncoderModel = state.t5EncoderModel;
  newState.clipEmbedModel = state.clipEmbedModel;
  newState.refinerModel = state.refinerModel;
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
  negativePromptChanged,
  positivePrompt2Changed,
  negativePrompt2Changed,
  shouldConcatPromptsChanged,
  refinerModelChanged,
  setRefinerSteps,
  setRefinerCFGScale,
  setRefinerScheduler,
  setRefinerPositiveAestheticScore,
  setRefinerNegativeAestheticScore,
  setRefinerStart,
  modelChanged,
  paramsReset,
} = paramsSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const paramsPersistConfig: PersistConfig<ParamsState> = {
  name: paramsSlice.name,
  initialState: getInitialState(),
  migrate,
  persistDenylist: [],
};

export const selectParamsSlice = (state: RootState) => state.params;
export const createParamsSelector = <T>(selector: Selector<ParamsState, T>) =>
  createSelector(selectParamsSlice, selector);

export const selectBase = createParamsSelector((params) => params.model?.base);
export const selectIsSDXL = createParamsSelector((params) => params.model?.base === 'sdxl');
export const selectIsFLUX = createParamsSelector((params) => params.model?.base === 'flux');
export const selectIsSD3 = createParamsSelector((params) => params.model?.base === 'sd-3');
export const selectIsCogView4 = createParamsSelector((params) => params.model?.base === 'cogview4');
export const selectIsImagen3 = createParamsSelector((params) => params.model?.base === 'imagen3');
export const selectIsImagen4 = createParamsSelector((params) => params.model?.base === 'imagen4');
export const selectIsChatGTP4o = createParamsSelector((params) => params.model?.base === 'chatgpt-4o');

export const selectModel = createParamsSelector((params) => params.model);
export const selectModelKey = createParamsSelector((params) => params.model?.key);
export const selectVAE = createParamsSelector((params) => params.vae);
export const selectFLUXVAE = createParamsSelector((params) => params.fluxVAE);
export const selectVAEKey = createParamsSelector((params) => params.vae?.key);
export const selectT5EncoderModel = createParamsSelector((params) => params.t5EncoderModel);
export const selectCLIPEmbedModel = createParamsSelector((params) => params.clipEmbedModel);
export const selectCLIPLEmbedModel = createParamsSelector((params) => params.clipLEmbedModel);

export const selectCLIPGEmbedModel = createParamsSelector((params) => params.clipGEmbedModel);

export const selectCFGScale = createParamsSelector((params) => params.cfgScale);
export const selectGuidance = createParamsSelector((params) => params.guidance);
export const selectSteps = createParamsSelector((params) => params.steps);
export const selectCFGRescaleMultiplier = createParamsSelector((params) => params.cfgRescaleMultiplier);
export const selectCLIPSKip = createParamsSelector((params) => params.clipSkip);
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
export const selectPositivePrompt2 = createParamsSelector((params) => params.positivePrompt2);
export const selectNegativePrompt2 = createParamsSelector((params) => params.negativePrompt2);
export const selectShouldConcatPrompts = createParamsSelector((params) => params.shouldConcatPrompts);
export const selectScheduler = createParamsSelector((params) => params.scheduler);
export const selectSeamlessXAxis = createParamsSelector((params) => params.seamlessXAxis);
export const selectSeamlessYAxis = createParamsSelector((params) => params.seamlessYAxis);
export const selectSeed = createParamsSelector((params) => params.seed);
export const selectShouldRandomizeSeed = createParamsSelector((params) => params.shouldRandomizeSeed);
export const selectVAEPrecision = createParamsSelector((params) => params.vaePrecision);
export const selectIterations = createParamsSelector((params) => params.iterations);
export const selectShouldUseCPUNoise = createParamsSelector((params) => params.shouldUseCpuNoise);

export const selectUpscaleScheduler = createParamsSelector((params) => params.upscaleScheduler);
export const selectUpscaleCfgScale = createParamsSelector((params) => params.upscaleCfgScale);

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
