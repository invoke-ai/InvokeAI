import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';

type HotkeysState = {
  shift: boolean;
};

const initialHotkeysState: HotkeysState = {
  shift: false,
};

const initialState: HotkeysState = initialHotkeysState;

export const hotkeysSlice = createSlice({
  name: 'hotkeys',
  initialState,
  reducers: {
    shiftKeyPressed: (state, action: PayloadAction<boolean>) => {
      state.shift = action.payload;
    },
  },
});

export const { shiftKeyPressed } = hotkeysSlice.actions;

export default hotkeysSlice.reducer;
