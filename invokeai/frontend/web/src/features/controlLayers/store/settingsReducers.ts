import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State } from 'features/controlLayers/store/types';

export const settingsReducers = {
  clipToBboxChanged: (state, action: PayloadAction<boolean>) => {
    state.settings.clipToBbox = action.payload;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
