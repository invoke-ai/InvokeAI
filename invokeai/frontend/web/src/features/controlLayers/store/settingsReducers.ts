import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State } from 'features/controlLayers/store/types';

export const settingsReducers = {
  maskOpacityChanged: (state, action: PayloadAction<number>) => {
    state.settings.maskOpacity = action.payload;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
