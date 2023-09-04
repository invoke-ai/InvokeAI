import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';

type HotkeysState = {
  shift: boolean;
  ctrl: boolean;
  meta: boolean;
};

export const initialHotkeysState: HotkeysState = {
  shift: false,
  ctrl: false,
  meta: false,
};

export const hotkeysSlice = createSlice({
  name: 'hotkeys',
  initialState: initialHotkeysState,
  reducers: {
    shiftKeyPressed: (state, action: PayloadAction<boolean>) => {
      state.shift = action.payload;
    },
    ctrlKeyPressed: (state, action: PayloadAction<boolean>) => {
      state.ctrl = action.payload;
    },
    metaKeyPressed: (state, action: PayloadAction<boolean>) => {
      state.meta = action.payload;
    },
  },
});

export const { shiftKeyPressed, ctrlKeyPressed, metaKeyPressed } =
  hotkeysSlice.actions;

export default hotkeysSlice.reducer;
