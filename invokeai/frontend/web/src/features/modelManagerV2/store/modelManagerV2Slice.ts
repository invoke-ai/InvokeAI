import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { ModelType } from 'services/api/types';

export type FilterableModelType = Exclude<ModelType, 'onnx'> | 'refiner';

type ModelManagerState = {
  _version: 1;
  selectedModelKey: string | null;
  selectedModelMode: 'edit' | 'view';
  searchTerm: string;
  filteredModelType: FilterableModelType | null;
  scanPath: string | undefined;
  shouldInstallInPlace: boolean;
};

const initialModelManagerState: ModelManagerState = {
  _version: 1,
  selectedModelKey: null,
  selectedModelMode: 'view',
  filteredModelType: null,
  searchTerm: '',
  scanPath: undefined,
  shouldInstallInPlace: true,
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
    setFilteredModelType: (state, action: PayloadAction<FilterableModelType | null>) => {
      state.filteredModelType = action.payload;
    },
    setScanPath: (state, action: PayloadAction<string | undefined>) => {
      state.scanPath = action.payload;
    },
    shouldInstallInPlaceChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldInstallInPlace = action.payload;
    },
  },
});

export const {
  setSelectedModelKey,
  setSearchTerm,
  setFilteredModelType,
  setSelectedModelMode,
  setScanPath,
  shouldInstallInPlaceChanged,
} = modelManagerV2Slice.actions;

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

export const selectModelManagerV2Slice = (state: RootState) => state.modelmanagerV2;

export const createModelManagerSelector = <T>(selector: (state: ModelManagerState) => T) =>
  createSelector(selectModelManagerV2Slice, selector);

export const selectSelectedModelKey = createModelManagerSelector((modelManager) => modelManager.selectedModelKey);
export const selectSelectedModelMode = createModelManagerSelector((modelManager) => modelManager.selectedModelMode);
export const selectSearchTerm = createModelManagerSelector((mm) => mm.searchTerm);
export const selectFilteredModelType = createModelManagerSelector((mm) => mm.filteredModelType);
export const selectShouldInstallInPlace = createModelManagerSelector((mm) => mm.shouldInstallInPlace);
