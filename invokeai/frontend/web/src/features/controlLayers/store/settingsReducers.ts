import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasBackgroundStyle, CanvasV2State } from 'features/controlLayers/store/types';

export const settingsReducers = {
  clipToBboxChanged: (state, action: PayloadAction<boolean>) => {
    state.settings.clipToBbox = action.payload;
  },
  canvasBackgroundStyleChanged: (state, action: PayloadAction<CanvasBackgroundStyle>) => {
    state.settings.canvasBackgroundStyle = action.payload;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
