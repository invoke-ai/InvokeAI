import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig } from 'app/store/store';

type ModelManagerState = {
  _version: 1;
  selectedModelKey: string | null;
  selectedModelMode: 'edit' | 'view';
  searchTerm: string;
  filteredModelType: string | null;
  scanPath: string | undefined;
};

const initialModelManagerState: ModelManagerState = {
  _version: 1,
  selectedModelKey: null,
  selectedModelMode: 'view',
  filteredModelType: null,
  searchTerm: '',
  scanPath: undefined,
};

export const modelManagerV2Slice = createSlice({
  name: 'modelmanagerV2',
  initialState: initialModelManagerState,
  reducers: {
    setSelectedModelKey: (state, action: PayloadAction<string | null>) => {
      state.selectedModelMode = 'view';
      state.selectedModelKey = action.payload;
    },
    setSelectedModelMode: (state, action: PayloadAction<'view' | 'edit'>) => {
      state.selectedModelMode = action.payload;
    },
    setSearchTerm: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },

    setFilteredModelType: (state, action: PayloadAction<string | null>) => {
      state.filteredModelType = action.payload;
    },
    setScanPath: (state, action: PayloadAction<string | undefined>) => {
      state.scanPath = action.payload;
    },
  },
});

export const { setSelectedModelKey, setSearchTerm, setFilteredModelType, setSelectedModelMode, setScanPath } =
  modelManagerV2Slice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateModelManagerState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const modelManagerV2PersistConfig: PersistConfig<ModelManagerState> = {
  name: modelManagerV2Slice.name,
  initialState: initialModelManagerState,
  migrate: migrateModelManagerState,
  persistDenylist: ['selectedModelKey', 'selectedModelMode', 'filteredModelType', 'searchTerm'],
};
