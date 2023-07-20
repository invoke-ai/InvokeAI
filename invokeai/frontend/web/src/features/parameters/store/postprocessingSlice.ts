import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { ESRGANInvocation } from 'services/api/types';

export type ESRGANModelName = NonNullable<ESRGANInvocation['model_name']>;

export interface PostprocessingState {
  esrganModelName: ESRGANModelName;
}

export const initialPostprocessingState: PostprocessingState = {
  esrganModelName: 'RealESRGAN_x4plus.pth',
};

export const postprocessingSlice = createSlice({
  name: 'postprocessing',
  initialState: initialPostprocessingState,
  reducers: {
    esrganModelNameChanged: (state, action: PayloadAction<ESRGANModelName>) => {
      state.esrganModelName = action.payload;
    },
  },
});

export const { esrganModelNameChanged } = postprocessingSlice.actions;

export default postprocessingSlice.reducer;
