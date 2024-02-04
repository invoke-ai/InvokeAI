import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';

type ModelManagerState = {
  _version: 1;
  searchFolder: string | null;
  advancedAddScanModel: string | null;
};

export const initialModelManagerState: ModelManagerState = {
  _version: 1,
  searchFolder: null,
  advancedAddScanModel: null,
};

export const modelManagerSlice = createSlice({
  name: 'modelmanager',
  initialState: initialModelManagerState,
  reducers: {
    setSearchFolder: (state, action: PayloadAction<string | null>) => {
      state.searchFolder = action.payload;
    },
    setAdvancedAddScanModel: (state, action: PayloadAction<string | null>) => {
      state.advancedAddScanModel = action.payload;
    },
  },
});

export const { setSearchFolder, setAdvancedAddScanModel } = modelManagerSlice.actions;

export const selectModelManagerSlice = (state: RootState) => state.modelmanager;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateModelManagerState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const modelManagerPersistConfig: PersistConfig<ModelManagerState> = {
  name: modelManagerSlice.name,
  initialState: initialModelManagerState,
  migrate: migrateModelManagerState,
  persistDenylist: [],
};
