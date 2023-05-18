import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';

type HotkeysState = {
  shift: boolean;
};

export const initialHotkeysState: HotkeysState = {
  shift: false,
};

export const hotkeysSlice = createSlice({
  name: 'hotkeys',
  initialState: initialHotkeysState,
  reducers: {
    shiftKeyPressed: (state, action: PayloadAction<boolean>) => {
      state.shift = action.payload;
    },
  },
});

export const { shiftKeyPressed } = hotkeysSlice.actions;

export default hotkeysSlice.reducer;

export const hotkeysSelector = (state: RootState) => state.hotkeys;
