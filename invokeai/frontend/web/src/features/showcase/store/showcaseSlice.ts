import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';

export interface ShowcaseState {
  showSeamless: boolean;
}

export const initialShowcaseState: ShowcaseState = {
  showSeamless: false,
};

const initialState: ShowcaseState = initialShowcaseState;

export const showcaseSlice = createSlice({
  name: 'showcase',
  initialState,
  reducers: {
    setShowSeamless: (state, action: PayloadAction<boolean>) => {
      state.showSeamless = action.payload;
    },
  },
});

export const { setShowSeamless } = showcaseSlice.actions;

export const selectShowcaseSlice = (state: RootState) => state.showcase;
