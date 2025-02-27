import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { ParameterSpandrelImageToImageModel } from 'features/parameters/types/parameterSchemas';
import type { ControlNetModelConfig, ImageDTO } from 'services/api/types';

export interface UpscaleState {
  _version: 1;
  upscaleModel: ParameterSpandrelImageToImageModel | null;
  upscaleInitialImage: ImageDTO | null;
  structure: number;
  creativity: number;
  tileControlnetModel: ControlNetModelConfig | null;
  scale: number;
  postProcessingModel: ParameterSpandrelImageToImageModel | null;
}

const initialUpscaleState: UpscaleState = {
  _version: 1,
  upscaleModel: null,
  upscaleInitialImage: null,
  structure: 0,
  creativity: 0,
  tileControlnetModel: null,
  scale: 4,
  postProcessingModel: null,
};

export const upscaleSlice = createSlice({
  name: 'upscale',
  initialState: initialUpscaleState,
  reducers: {
    upscaleModelChanged: (state, action: PayloadAction<ParameterSpandrelImageToImageModel | null>) => {
      state.upscaleModel = action.payload;
    },
    upscaleInitialImageChanged: (state, action: PayloadAction<ImageDTO | null>) => {
      state.upscaleInitialImage = action.payload;
    },
    structureChanged: (state, action: PayloadAction<number>) => {
      state.structure = action.payload;
    },
    creativityChanged: (state, action: PayloadAction<number>) => {
      state.creativity = action.payload;
    },
    tileControlnetModelChanged: (state, action: PayloadAction<ControlNetModelConfig | null>) => {
      state.tileControlnetModel = action.payload;
    },
    scaleChanged: (state, action: PayloadAction<number>) => {
      state.scale = action.payload;
    },
    postProcessingModelChanged: (state, action: PayloadAction<ParameterSpandrelImageToImageModel | null>) => {
      state.postProcessingModel = action.payload;
    },
  },
});

export const {
  upscaleModelChanged,
  upscaleInitialImageChanged,
  structureChanged,
  creativityChanged,
  tileControlnetModelChanged,
  scaleChanged,
  postProcessingModelChanged,
} = upscaleSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateUpscaleState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const upscalePersistConfig: PersistConfig<UpscaleState> = {
  name: upscaleSlice.name,
  initialState: initialUpscaleState,
  migrate: migrateUpscaleState,
  persistDenylist: [],
};

export const selectUpscaleSlice = (state: RootState) => state.upscale;
const createUpscaleSelector = <T>(selector: Selector<UpscaleState, T>) => createSelector(selectUpscaleSlice, selector);
export const selectPostProcessingModel = createUpscaleSelector((upscale) => upscale.postProcessingModel);
export const selectCreativity = createUpscaleSelector((upscale) => upscale.creativity);
export const selectUpscaleModel = createUpscaleSelector((upscale) => upscale.upscaleModel);
export const selectTileControlNetModel = createUpscaleSelector((upscale) => upscale.tileControlnetModel);
export const selectStructure = createUpscaleSelector((upscale) => upscale.structure);
export const selectUpscaleInitialImage = createUpscaleSelector((upscale) => upscale.upscaleInitialImage);
export const selectUpscaleScale = createUpscaleSelector((upscale) => upscale.scale);
