import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { SDMetadata } from '../gallery/gallerySlice';
import randomInt from './util/randomInt';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';

const calculateRealSteps = (
  steps: number,
  strength: number,
  hasInitImage: boolean
): number => {
  return hasInitImage ? Math.floor(strength * steps) : steps;
};

export type UpscalingLevel = 0 | 2 | 3 | 4;

export interface SDState {
  prompt: string;
  iterations: number;
  steps: number;
  realSteps: number;
  cfgScale: number;
  height: number;
  width: number;
  sampler: string;
  seed: number;
  img2imgStrength: number;
  gfpganStrength: number;
  upscalingLevel: UpscalingLevel;
  upscalingStrength: number;
  initialImagePath: string;
  maskPath: string;
  seamless: boolean;
  shouldFitToWidthHeight: boolean;
  shouldGenerateVariations: boolean;
  variantAmount: number;
  seedWeights: string;
  shouldRunESRGAN: boolean;
  shouldRunGFPGAN: boolean;
  shouldRandomizeSeed: boolean;
}

const initialSDState: SDState = {
  prompt: '',
  iterations: 1,
  steps: 50,
  realSteps: 50,
  cfgScale: 7.5,
  height: 512,
  width: 512,
  sampler: 'k_lms',
  seed: -1,
  seamless: false,
  img2imgStrength: 0.75,
  initialImagePath: '',
  maskPath: '',
  shouldFitToWidthHeight: true,
  shouldGenerateVariations: false,
  variantAmount: 0.1,
  seedWeights: '',
  shouldRunESRGAN: true,
  upscalingLevel: 4,
  upscalingStrength: 0.75,
  shouldRunGFPGAN: true,
  gfpganStrength: 0.8,
  shouldRandomizeSeed: true,
};

const initialState: SDState = initialSDState;

export const sdSlice = createSlice({
  name: 'sd',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string>) => {
      state.prompt = action.payload;
    },
    setIterations: (state, action: PayloadAction<number>) => {
      state.iterations = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      const { img2imgStrength, initialImagePath } = state;
      state.steps = action.payload;
      state.realSteps = calculateRealSteps(
        action.payload,
        img2imgStrength,
        Boolean(initialImagePath)
      );
    },
    setCfgScale: (state, action: PayloadAction<number>) => {
      state.cfgScale = action.payload;
    },
    setHeight: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    setWidth: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    setSampler: (state, action: PayloadAction<string>) => {
      state.sampler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
      const { steps, initialImagePath } = state;
      state.img2imgStrength = action.payload;
      state.realSteps = calculateRealSteps(
        steps,
        action.payload,
        Boolean(initialImagePath)
      );
    },
    setGfpganStrength: (state, action: PayloadAction<number>) => {
      state.gfpganStrength = action.payload;
    },
    setUpscalingLevel: (state, action: PayloadAction<UpscalingLevel>) => {
      state.upscalingLevel = action.payload;
    },
    setUpscalingStrength: (state, action: PayloadAction<number>) => {
      state.upscalingStrength = action.payload;
    },
    setInitialImagePath: (state, action: PayloadAction<string>) => {
      state.initialImagePath = action.payload;
      const { steps, img2imgStrength } = state;
      state.realSteps = calculateRealSteps(
        steps,
        img2imgStrength,
        Boolean(action.payload)
      );
    },
    setMaskPath: (state, action: PayloadAction<string>) => {
      state.maskPath = action.payload;
    },
    setSeamless: (state, action: PayloadAction<boolean>) => {
      state.seamless = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    resetInitialImagePath: (state) => {
      state.initialImagePath = '';
      state.maskPath = '';
    },
    resetMaskPath: (state) => {
      state.maskPath = '';
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    randomizeSeed: (state) => {
      state.seed = randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX);
    },
    setParameter: (
      state,
      action: PayloadAction<{ key: string; value: string | number | boolean }>
    ) => {
      const { key, value } = action.payload;
      const temp = { ...state, [key]: value };
      return temp;
    },
    setShouldGenerateVariations: (state, action: PayloadAction<boolean>) => {
      state.shouldGenerateVariations = action.payload;
    },
    setVariantAmount: (state, action: PayloadAction<number>) => {
      state.variantAmount = action.payload;
    },
    setSeedWeights: (state, action: PayloadAction<string>) => {
      state.seedWeights = action.payload;
    },
    setAllParameters: (state, action: PayloadAction<SDMetadata>) => {
      return { ...state, ...action.payload };
    },
    resetSDState: (state) => {
      return {
        ...state,
        ...initialSDState,
      };
    },
    setShouldRunGFPGAN: (state, action: PayloadAction<boolean>) => {
      state.shouldRunGFPGAN = action.payload;
    },
    setShouldRunESRGAN: (state, action: PayloadAction<boolean>) => {
      state.shouldRunESRGAN = action.payload;
    },
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
  },
});

export const {
  setPrompt,
  setIterations,
  setSteps,
  setCfgScale,
  setHeight,
  setWidth,
  setSampler,
  setSeed,
  setSeamless,
  setImg2imgStrength,
  setGfpganStrength,
  setUpscalingLevel,
  setUpscalingStrength,
  setInitialImagePath,
  setMaskPath,
  resetInitialImagePath,
  resetSeed,
  randomizeSeed,
  resetSDState,
  setShouldFitToWidthHeight,
  setParameter,
  setShouldGenerateVariations,
  setSeedWeights,
  setVariantAmount,
  setAllParameters,
  setShouldRunGFPGAN,
  setShouldRunESRGAN,
  setShouldRandomizeSeed,
} = sdSlice.actions;

export default sdSlice.reducer;
