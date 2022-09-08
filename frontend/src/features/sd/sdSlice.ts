import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { SDMetadata } from '../gallery/gallerySlice';

const calculateRealSteps = (
  steps: number,
  strength: number,
  hasInitImage: boolean
): number => {
  return hasInitImage ? Math.floor(strength * steps) : steps;
};

export interface SDState {
  prompt: string;
  imagesToGenerate: number;
  steps: number;
  realSteps: number;
  cfgScale: number;
  height: number;
  width: number;
  sampler: string;
  seed: number;
  img2imgStrength: number;
  gfpganStrength: number;
  upscalingLevel: number;
  upscalingStrength: number;
  initialImagePath: string;
  maskPath: string;
  seamless: boolean;
  shouldFitToWidthHeight: boolean;
}

const initialSDState = {
  prompt: 'Cyborg pickle shooting lasers',
  imagesToGenerate: 1,
  steps: 50,
  realSteps: 50,
  cfgScale: 7.5,
  height: 512,
  width: 512,
  sampler: 'k_lms',
  seed: -1,
  img2imgStrength: 0.75,
  gfpganStrength: 0.8,
  upscalingLevel: 0,
  upscalingStrength: 0.75,
  initialImagePath: '',
  maskPath: '',
  seamless: false,
  shouldFitToWidthHeight: true,
};

const initialState: SDState = initialSDState;

export const sdSlice = createSlice({
  name: 'sd',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string>) => {
      state.prompt = action.payload;
    },
    setImagesToGenerate: (state, action: PayloadAction<number>) => {
      state.imagesToGenerate = action.payload;
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
    setUpscalingLevel: (state, action: PayloadAction<number>) => {
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
      state.seed = Math.round(Math.random() * 1000000);
    },
    setParameter: (
      state,
      action: PayloadAction<{ key: string; value: string | number | boolean }>
    ) => {
      const { key, value } = action.payload;
      const temp = { ...state, [key]: value };
      return temp;
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
  },
});

export const {
  setPrompt,
  setImagesToGenerate,
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
  setAllParameters,
} = sdSlice.actions;

export default sdSlice.reducer;
