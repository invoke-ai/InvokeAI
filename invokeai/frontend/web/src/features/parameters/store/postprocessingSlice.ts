import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { z } from 'zod';

const zParamESRGANModelName = z.enum([
  'RealESRGAN_x4plus.pth',
  'RealESRGAN_x4plus_anime_6B.pth',
  'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
  'RealESRGAN_x2plus.pth',
]);
export type ParamESRGANModelName = z.infer<typeof zParamESRGANModelName>;
export const isParamESRGANModelName = (v: unknown): v is ParamESRGANModelName =>
  zParamESRGANModelName.safeParse(v).success;

export interface PostprocessingState {
  _version: 1;
  esrganModelName: ParamESRGANModelName;
}

const initialPostprocessingState: PostprocessingState = {
  _version: 1,
  esrganModelName: 'RealESRGAN_x4plus.pth',
};

export const postprocessingSlice = createSlice({
  name: 'postprocessing',
  initialState: initialPostprocessingState,
  reducers: {
    esrganModelNameChanged: (state, action: PayloadAction<ParamESRGANModelName>) => {
      state.esrganModelName = action.payload;
    },
  },
});

export const { esrganModelNameChanged } = postprocessingSlice.actions;

export const selectPostprocessingSlice = (state: RootState) => state.postprocessing;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migratePostprocessingState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const postprocessingPersistConfig: PersistConfig<PostprocessingState> = {
  name: postprocessingSlice.name,
  initialState: initialPostprocessingState,
  migrate: migratePostprocessingState,
  persistDenylist: [],
};
