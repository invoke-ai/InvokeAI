import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { ParameterSpandrelImageToImageModel } from '../types/parameterSchemas';
import { ImageDTO } from '../../../services/api/types';


interface UpscaleState {
  _version: 1;
  upscaleModel: ParameterSpandrelImageToImageModel | null;
  upscaleInitialImage: ImageDTO | null;
  sharpness: number;
  structure: number;
  creativity: number;
}

const initialUpscaleState: UpscaleState = {
  _version: 1,
  upscaleModel: null,
  upscaleInitialImage: null,
  sharpness: 0,
  structure: 0,
  creativity: 0

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
  },
});

export const { upscaleModelChanged, upscaleInitialImageChanged } = upscaleSlice.actions;

export const selectUpscalelice = (state: RootState) => state.upscale;

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
