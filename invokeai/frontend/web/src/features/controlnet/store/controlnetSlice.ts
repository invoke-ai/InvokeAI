import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { ControlNetConfig, ControlNetUIState } from './controlnetTypes';

const controlnetInitialState: ControlNetUIState = {
  controlnet1: {
    controlnetEnabled: false,
    controlnetImage: null,
    controlnetProcessor: 'none',
    controlnetModel: null,
    controlnetWeight: 1,
    controlnetStart: 0,
    controlnetEnd: 1,
  },
  controlnet2: {
    controlnetEnabled: false,
    controlnetImage: null,
    controlnetProcessor: 'none',
    controlnetModel: null,
    controlnetWeight: 1,
    controlnetStart: 0,
    controlnetEnd: 1,
  },
  controlnet3: {
    controlnetEnabled: false,
    controlnetImage: null,
    controlnetProcessor: 'none',
    controlnetModel: null,
    controlnetWeight: 1,
    controlnetStart: 0,
    controlnetEnd: 1,
  },
};

export const controlnetSlice = createSlice({
  name: 'controlnet',
  initialState: controlnetInitialState,
  reducers: {
    setControlNet1: (state, action: PayloadAction<ControlNetConfig>) => {
      state.controlnet1 = action.payload;
    },
    setControlNet2: (state, action: PayloadAction<ControlNetConfig>) => {
      state.controlnet2 = action.payload;
    },
    setControlNet3: (state, action: PayloadAction<ControlNetConfig>) => {
      state.controlnet3 = action.payload;
    },
  },
});

export const { setControlNet1, setControlNet2, setControlNet3 } =
  controlnetSlice.actions;

export default controlnetSlice.reducer;
