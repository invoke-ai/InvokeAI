import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { FACETOOL_TYPES } from 'app/constants';

export type UpscalingLevel = 2 | 4;

export type FacetoolType = (typeof FACETOOL_TYPES)[number];

export interface PostprocessingState {
  codeformerFidelity: number;
  facetoolStrength: number;
  facetoolType: FacetoolType;
  hiresFix: boolean;
  hiresStrength: number;
  shouldLoopback: boolean;
  shouldRunESRGAN: boolean;
  shouldRunFacetool: boolean;
  upscalingLevel: UpscalingLevel;
  upscalingDenoising: number;
  upscalingStrength: number;
}

export const initialPostprocessingState: PostprocessingState = {
  codeformerFidelity: 0.75,
  facetoolStrength: 0.75,
  facetoolType: 'gfpgan',
  hiresFix: false,
  hiresStrength: 0.75,
  shouldLoopback: false,
  shouldRunESRGAN: false,
  shouldRunFacetool: false,
  upscalingLevel: 4,
  upscalingDenoising: 0.75,
  upscalingStrength: 0.75,
};

export const postprocessingSlice = createSlice({
  name: 'postprocessing',
  initialState: initialPostprocessingState,
  reducers: {
    setFacetoolStrength: (state, action: PayloadAction<number>) => {
      state.facetoolStrength = action.payload;
    },
    setCodeformerFidelity: (state, action: PayloadAction<number>) => {
      state.codeformerFidelity = action.payload;
    },
    setUpscalingLevel: (state, action: PayloadAction<UpscalingLevel>) => {
      state.upscalingLevel = action.payload;
    },
    setUpscalingDenoising: (state, action: PayloadAction<number>) => {
      state.upscalingDenoising = action.payload;
    },
    setUpscalingStrength: (state, action: PayloadAction<number>) => {
      state.upscalingStrength = action.payload;
    },
    setHiresFix: (state, action: PayloadAction<boolean>) => {
      state.hiresFix = action.payload;
    },
    setHiresStrength: (state, action: PayloadAction<number>) => {
      state.hiresStrength = action.payload;
    },
    resetPostprocessingState: (state) => {
      return {
        ...state,
        ...initialPostprocessingState,
      };
    },
    setShouldRunFacetool: (state, action: PayloadAction<boolean>) => {
      state.shouldRunFacetool = action.payload;
    },
    setFacetoolType: (state, action: PayloadAction<FacetoolType>) => {
      state.facetoolType = action.payload;
    },
    setShouldRunESRGAN: (state, action: PayloadAction<boolean>) => {
      state.shouldRunESRGAN = action.payload;
    },
    setShouldLoopback: (state, action: PayloadAction<boolean>) => {
      state.shouldLoopback = action.payload;
    },
  },
});

export const {
  resetPostprocessingState,
  setCodeformerFidelity,
  setFacetoolStrength,
  setFacetoolType,
  setHiresFix,
  setHiresStrength,
  setShouldLoopback,
  setShouldRunESRGAN,
  setShouldRunFacetool,
  setUpscalingLevel,
  setUpscalingDenoising,
  setUpscalingStrength,
} = postprocessingSlice.actions;

export default postprocessingSlice.reducer;
