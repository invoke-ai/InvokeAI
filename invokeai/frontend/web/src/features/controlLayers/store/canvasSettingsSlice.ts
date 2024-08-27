import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';

export type CanvasSettingsState = {
  imageSmoothing: boolean;
  showHUD: boolean;
  autoSave: boolean;
  preserveMaskedArea: boolean;
  cropToBboxOnSave: boolean;
  clipToBbox: boolean;
  dynamicGrid: boolean;
};

const initialState: CanvasSettingsState = {
  // TODO(psyche): These are copied from old canvas state, need to be implemented
  autoSave: false,
  imageSmoothing: true,
  preserveMaskedArea: false,
  showHUD: true,
  clipToBbox: false,
  cropToBboxOnSave: false,
  dynamicGrid: false,
};

export const canvasSettingsSlice = createSlice({
  name: 'canvasSettings',
  initialState,
  reducers: {
    clipToBboxChanged: (state, action: PayloadAction<boolean>) => {
      state.clipToBbox = action.payload;
    },
    settingsDynamicGridToggled: (state) => {
      state.dynamicGrid = !state.dynamicGrid;
    },
    settingsAutoSaveToggled: (state) => {
      state.autoSave = !state.autoSave;
    },
  },
});

export const { clipToBboxChanged, settingsAutoSaveToggled, settingsDynamicGridToggled } = canvasSettingsSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasSettingsPersistConfig: PersistConfig<CanvasSettingsState> = {
  name: canvasSettingsSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
export const selectCanvasSettingsSlice = (s: RootState) => s.canvasSettings;
