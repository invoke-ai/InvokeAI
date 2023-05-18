import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';

export interface LightboxState {
  isLightboxOpen: boolean;
}

export const initialLightboxState: LightboxState = {
  isLightboxOpen: false,
};

const initialState: LightboxState = initialLightboxState;

export const lightboxSlice = createSlice({
  name: 'lightbox',
  initialState,
  reducers: {
    setIsLightboxOpen: (state, action: PayloadAction<boolean>) => {
      state.isLightboxOpen = action.payload;
    },
  },
});

export const { setIsLightboxOpen } = lightboxSlice.actions;

export default lightboxSlice.reducer;
