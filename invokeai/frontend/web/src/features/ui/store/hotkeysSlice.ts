import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store';

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

export const hotkeysSelector = (state: RootState) => state.hotkeys;
