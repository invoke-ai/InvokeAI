import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { AppConfig, PartialAppConfig } from 'app/types/invokeai';
import { merge } from 'lodash-es';

export const initialConfigState: AppConfig = {
  shouldUpdateImagesOnConnect: false,
  disabledTabs: [],
  disabledFeatures: ['lightbox', 'faceRestore', 'batches'],
  disabledSDFeatures: [
    'variation',
    'symmetry',
    'hires',
    'perlinNoise',
    'noiseThreshold',
  ],
  canRestoreDeletedImagesFromBin: true,
  sd: {
    disabledControlNetModels: [],
    disabledControlNetProcessors: [],
    iterations: {
      initial: 1,
      min: 1,
      sliderMax: 20,
      inputMax: 9999,
      fineStep: 1,
      coarseStep: 1,
    },
    width: {
      initial: 512,
      min: 64,
      sliderMax: 1536,
      inputMax: 4096,
      fineStep: 8,
      coarseStep: 64,
    },
    height: {
      initial: 512,
      min: 64,
      sliderMax: 1536,
      inputMax: 4096,
      fineStep: 8,
      coarseStep: 64,
    },
    steps: {
      initial: 30,
      min: 1,
      sliderMax: 100,
      inputMax: 500,
      fineStep: 1,
      coarseStep: 1,
    },
    guidance: {
      initial: 7,
      min: 1,
      sliderMax: 20,
      inputMax: 200,
      fineStep: 0.1,
      coarseStep: 0.5,
    },
    img2imgStrength: {
      initial: 0.7,
      min: 0,
      sliderMax: 1,
      inputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    dynamicPrompts: {
      maxPrompts: {
        initial: 100,
        min: 1,
        sliderMax: 1000,
        inputMax: 10000,
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

export default configSlice.reducer;
