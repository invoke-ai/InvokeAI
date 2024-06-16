import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import type {
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterModel,
  ParameterPrecision,
  ParameterScheduler,
  ParameterSDXLRefinerModel,
  ParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { clamp } from 'lodash-es';

export const paramsReducers = {
  setIterations: (state, action: PayloadAction<number>) => {
    state.params.iterations = action.payload;
  },
  setSteps: (state, action: PayloadAction<number>) => {
    state.params.steps = action.payload;
  },
  setCfgScale: (state, action: PayloadAction<ParameterCFGScale>) => {
    state.params.cfgScale = action.payload;
  },
  setCfgRescaleMultiplier: (state, action: PayloadAction<ParameterCFGRescaleMultiplier>) => {
    state.params.cfgRescaleMultiplier = action.payload;
  },
  setScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
    state.params.scheduler = action.payload;
  },
  setSeed: (state, action: PayloadAction<number>) => {
    state.params.seed = action.payload;
    state.params.shouldRandomizeSeed = false;
  },
  setImg2imgStrength: (state, action: PayloadAction<number>) => {
    state.params.img2imgStrength = action.payload;
  },
  setSeamlessXAxis: (state, action: PayloadAction<boolean>) => {
    state.params.seamlessXAxis = action.payload;
  },
  setSeamlessYAxis: (state, action: PayloadAction<boolean>) => {
    state.params.seamlessYAxis = action.payload;
  },
  setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
    state.params.shouldRandomizeSeed = action.payload;
  },
  modelChanged: (
    state,
    action: PayloadAction<{ model: ParameterModel | null; previousModel?: ParameterModel | null }>
  ) => {
    const { model, previousModel } = action.payload;
    state.params.model = model;

    // If the model base changes (e.g. SD1.5 -> SDXL), we need to change a few things
    if (model === null || previousModel?.base === model.base) {
      return;
    }

    // Update the bbox size to match the new model's optimal size
    // TODO(psyche): Should we change the document size too?
    const optimalDimension = getOptimalDimension(model);
    if (!getIsSizeOptimal(state.document.width, state.document.height, optimalDimension)) {
      const bboxDims = calculateNewSize(state.document.aspectRatio.value, optimalDimension * optimalDimension);
      state.bbox.width = bboxDims.width;
      state.bbox.height = bboxDims.height;

      if (state.bbox.scaleMethod === 'auto') {
        const scaledBboxDims = getScaledBoundingBoxDimensions(bboxDims, optimalDimension);
        state.bbox.scaledWidth = scaledBboxDims.width;
        state.bbox.scaledHeight = scaledBboxDims.height;
      }
    }

    // Clamp CLIP skip layer count to the bounds of the new model
    if (model.base === 'sdxl') {
      // We don't support user-defined CLIP skip for SDXL because it doesn't do anything useful
      state.params.clipSkip = 0;
    } else {
      const { maxClip } = CLIP_SKIP_MAP[model.base];
      state.params.clipSkip = clamp(state.params.clipSkip, 0, maxClip);
    }
  },
  vaeSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
    // null is a valid VAE!
    state.params.vae = action.payload;
  },
  vaePrecisionChanged: (state, action: PayloadAction<ParameterPrecision>) => {
    state.params.vaePrecision = action.payload;
  },
  setClipSkip: (state, action: PayloadAction<number>) => {
    state.params.clipSkip = action.payload;
  },
  shouldUseCpuNoiseChanged: (state, action: PayloadAction<boolean>) => {
    state.params.shouldUseCpuNoise = action.payload;
  },
  positivePromptChanged: (state, action: PayloadAction<string>) => {
    state.params.positivePrompt = action.payload;
  },
  negativePromptChanged: (state, action: PayloadAction<string>) => {
    state.params.negativePrompt = action.payload;
  },
  positivePrompt2Changed: (state, action: PayloadAction<string>) => {
    state.params.positivePrompt2 = action.payload;
  },
  negativePrompt2Changed: (state, action: PayloadAction<string>) => {
    state.params.negativePrompt2 = action.payload;
  },
  shouldConcatPromptsChanged: (state, action: PayloadAction<boolean>) => {
    state.params.shouldConcatPrompts = action.payload;
  },
  refinerModelChanged: (state, action: PayloadAction<ParameterSDXLRefinerModel | null>) => {
    state.params.refinerModel = action.payload;
  },
  setRefinerSteps: (state, action: PayloadAction<number>) => {
    state.params.refinerSteps = action.payload;
  },
  setRefinerCFGScale: (state, action: PayloadAction<number>) => {
    state.params.refinerCFGScale = action.payload;
  },
  setRefinerScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
    state.params.refinerScheduler = action.payload;
  },
  setRefinerPositiveAestheticScore: (state, action: PayloadAction<number>) => {
    state.params.refinerPositiveAestheticScore = action.payload;
  },
  setRefinerNegativeAestheticScore: (state, action: PayloadAction<number>) => {
    state.params.refinerNegativeAestheticScore = action.payload;
  },
  setRefinerStart: (state, action: PayloadAction<number>) => {
    state.params.refinerStart = action.payload;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
