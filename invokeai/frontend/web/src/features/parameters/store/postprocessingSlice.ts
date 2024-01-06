import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { z } from 'zod';

export const zParamESRGANModelName = z.enum([
  'RealESRGAN_x4plus.pth',
  'RealESRGAN_x4plus_anime_6B.pth',
  'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
  'RealESRGAN_x2plus.pth',
]);
export type ParamESRGANModelName = z.infer<typeof zParamESRGANModelName>;
export const isParamESRGANModelName = (v: unknown): v is ParamESRGANModelName =>
  zParamESRGANModelName.safeParse(v).success;

export interface PostprocessingState {
  esrganModelName: ParamESRGANModelName;
}

export const initialPostprocessingState: PostprocessingState = {
  esrganModelName: 'RealESRGAN_x4plus.pth',
};

export const postprocessingSlice = createSlice({
  name: 'postprocessing',
  initialState: initialPostprocessingState,
  reducers: {
    esrganModelNameChanged: (
      state,
      action: PayloadAction<ParamESRGANModelName>
    ) => {
      state.esrganModelName = action.payload;
    },
  },
});

export const { esrganModelNameChanged } = postprocessingSlice.actions;

export default postprocessingSlice.reducer;

export const selectPostprocessingSlice = (state: RootState) =>
  state.postprocessing;
