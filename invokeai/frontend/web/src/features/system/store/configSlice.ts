import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
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
  shouldUpdateImagesOnConnect: false,
  shouldFetchMetadataFromApi: false,
  disabledTabs: [],
  disabledFeatures: ['lightbox', 'faceRestore', 'batches'],
  disabledSDFeatures: ['variation', 'symmetry', 'hires', 'perlinNoise', 'noiseThreshold'],
  nodesAllowlist: undefined,
  nodesDenylist: undefined,
  canRestoreDeletedImagesFromBin: true,
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
