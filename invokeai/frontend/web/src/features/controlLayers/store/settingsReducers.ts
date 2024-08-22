import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State } from 'features/controlLayers/store/types';

export const settingsReducers = {
  clipToBboxChanged: (state, action: PayloadAction<boolean>) => {
    state.settings.clipToBbox = action.payload;
  },
  settingsDynamicGridToggled: (state) => {
    state.settings.dynamicGrid = !state.settings.dynamicGrid;
  },
  settingsAutoSaveToggled: (state) => {
    state.settings.autoSave = !state.settings.autoSave;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
