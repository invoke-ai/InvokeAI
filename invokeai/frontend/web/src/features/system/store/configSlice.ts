import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { AppConfig, NumericalParameterConfig, PartialAppConfig } from 'app/types/invokeai';
import { merge } from 'lodash-es';

const baseDimensionConfig: NumericalParameterConfig = {
  initial: 512, // determined by model selection, unused in practice
  sliderMin: 64,
  sliderMax: 1536,
  numberInputMin: 64,
  numberInputMax: 4096,
  fineStep: 8,
  coarseStep: 64,
};

const initialConfigState: AppConfig = {
  isLocal: true,
  shouldUpdateImagesOnConnect: false,
  shouldFetchMetadataFromApi: false,
  allowPrivateBoards: false,
  allowPrivateStylePresets: false,
  disabledTabs: [],
  disabledFeatures: ['lightbox', 'faceRestore', 'batches'],
  disabledSDFeatures: ['variation', 'symmetry', 'hires', 'perlinNoise', 'noiseThreshold'],
  nodesAllowlist: undefined,
  nodesDenylist: undefined,
  sd: {
    disabledControlNetModels: [],
    disabledControlNetProcessors: [],
    iterations: {
      initial: 1,
      sliderMin: 1,
      sliderMax: 1000,
      numberInputMin: 1,
      numberInputMax: 10000,
      fineStep: 1,
      coarseStep: 1,
    },
    width: { ...baseDimensionConfig },
    height: { ...baseDimensionConfig },
    boundingBoxWidth: { ...baseDimensionConfig },
    boundingBoxHeight: { ...baseDimensionConfig },
    scaledBoundingBoxWidth: { ...baseDimensionConfig },
    scaledBoundingBoxHeight: { ...baseDimensionConfig },
    scheduler: 'dpmpp_3m_k',
    vaePrecision: 'fp32',
    steps: {
      initial: 30,
      sliderMin: 1,
      sliderMax: 100,
      numberInputMin: 1,
      numberInputMax: 500,
      fineStep: 1,
      coarseStep: 1,
    },
    guidance: {
      initial: 7,
      sliderMin: 1,
      sliderMax: 20,
      numberInputMin: 1,
      numberInputMax: 200,
      fineStep: 0.1,
      coarseStep: 0.5,
    },
    img2imgStrength: {
      initial: 0.7,
      sliderMin: 0,
      sliderMax: 1,
      numberInputMin: 0,
      numberInputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    canvasCoherenceStrength: {
      initial: 0.3,
      sliderMin: 0,
      sliderMax: 1,
      numberInputMin: 0,
      numberInputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    hrfStrength: {
      initial: 0.45,
      sliderMin: 0,
      sliderMax: 1,
      numberInputMin: 0,
      numberInputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    canvasCoherenceEdgeSize: {
      initial: 16,
      sliderMin: 0,
      sliderMax: 128,
      numberInputMin: 0,
      numberInputMax: 1024,
      fineStep: 8,
      coarseStep: 16,
    },
    cfgRescaleMultiplier: {
      initial: 0,
      sliderMin: 0,
      sliderMax: 0.99,
      numberInputMin: 0,
      numberInputMax: 0.99,
      fineStep: 0.05,
      coarseStep: 0.1,
    },
    clipSkip: {
      initial: 0,
      sliderMin: 0,
      sliderMax: 12, // determined by model selection, unused in practice
      numberInputMin: 0,
      numberInputMax: 12, // determined by model selection, unused in practice
      fineStep: 1,
      coarseStep: 1,
    },
    infillPatchmatchDownscaleSize: {
      initial: 1,
      sliderMin: 1,
      sliderMax: 10,
      numberInputMin: 1,
      numberInputMax: 10,
      fineStep: 1,
      coarseStep: 1,
    },
    infillTileSize: {
      initial: 32,
      sliderMin: 16,
      sliderMax: 64,
      numberInputMin: 16,
      numberInputMax: 256,
      fineStep: 1,
      coarseStep: 1,
    },
    maskBlur: {
      initial: 16,
      sliderMin: 0,
      sliderMax: 128,
      numberInputMin: 0,
      numberInputMax: 512,
      fineStep: 1,
      coarseStep: 1,
    },
    ca: {
      weight: {
        initial: 1,
        sliderMin: 0,
        sliderMax: 2,
        numberInputMin: -1,
        numberInputMax: 2,
        fineStep: 0.01,
        coarseStep: 0.05,
      },
    },
    dynamicPrompts: {
      maxPrompts: {
        initial: 100,
        sliderMin: 1,
        sliderMax: 1000,
        numberInputMin: 1,
        numberInputMax: 10000,
        fineStep: 1,
        coarseStep: 10,
      },
    },
  },
  flux: {
    guidance: {
      initial: 4,
      sliderMin: 2,
      sliderMax: 6,
      numberInputMin: 1,
      numberInputMax: 20,
      fineStep: 0.1,
      coarseStep: 0.5,
    },
  },
};

export const configSlice = createSlice({
  name: 'config',
  initialState: initialConfigState,
  reducers: {
    configChanged: (state, action: PayloadAction<PartialAppConfig>) => {
      merge(state, action.payload);
    },
  },
});

export const { configChanged } = configSlice.actions;

export const selectConfigSlice = (state: RootState) => state.config;
const createConfigSelector = <T>(selector: Selector<AppConfig, T>) => createSelector(selectConfigSlice, selector);

export const selectWidthConfig = createConfigSelector((config) => config.sd.width);
export const selectHeightConfig = createConfigSelector((config) => config.sd.height);
export const selectStepsConfig = createConfigSelector((config) => config.sd.steps);
export const selectCFGScaleConfig = createConfigSelector((config) => config.sd.guidance);
export const selectGuidanceConfig = createConfigSelector((config) => config.flux.guidance);
export const selectCLIPSkipConfig = createConfigSelector((config) => config.sd.clipSkip);
export const selectCFGRescaleMultiplierConfig = createConfigSelector((config) => config.sd.cfgRescaleMultiplier);
export const selectCanvasCoherenceEdgeSizeConfig = createConfigSelector((config) => config.sd.canvasCoherenceEdgeSize);
export const selectMaskBlurConfig = createConfigSelector((config) => config.sd.maskBlur);
export const selectInfillPatchmatchDownscaleSizeConfig = createConfigSelector(
  (config) => config.sd.infillPatchmatchDownscaleSize
);
export const selectInfillTileSizeConfig = createConfigSelector((config) => config.sd.infillTileSize);
export const selectImg2imgStrengthConfig = createConfigSelector((config) => config.sd.img2imgStrength);
export const selectMaxPromptsConfig = createConfigSelector((config) => config.sd.dynamicPrompts.maxPrompts);
export const selectIterationsConfig = createConfigSelector((config) => config.sd.iterations);

export const selectMaxUpscaleDimension = createConfigSelector((config) => config.maxUpscaleDimension);
export const selectAllowPrivateStylePresets = createConfigSelector((config) => config.allowPrivateStylePresets);
export const selectWorkflowFetchDebounce = createConfigSelector((config) => config.workflowFetchDebounce ?? 300);
export const selectMetadataFetchDebounce = createConfigSelector((config) => config.metadataFetchDebounce ?? 300);

export const selectIsModelsTabDisabled = createConfigSelector((config) => config.disabledTabs.includes('models'));
export const selectMaxImageUploadCount = createConfigSelector((config) => config.maxImageUploadCount);

export const selectIsLocal = createSelector(selectConfigSlice, (config) => config.isLocal);
