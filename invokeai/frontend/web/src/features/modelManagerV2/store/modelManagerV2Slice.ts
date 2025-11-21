import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { zModelType } from 'features/nodes/types/common';
import { assert } from 'tsafe';
import z from 'zod';

const zFilterableModelType = zModelType.exclude(['onnx']).or(z.literal('refiner'));
export type FilterableModelType = z.infer<typeof zFilterableModelType>;

const zModelManagerState = z.object({
  _version: z.literal(1),
  selectedModelKey: z.string().nullable(),
  selectedModelMode: z.enum(['edit', 'view']),
  searchTerm: z.string(),
  filteredModelType: zFilterableModelType.nullable(),
  scanPath: z.string().optional(),
  shouldInstallInPlace: z.boolean(),
  selectedModelKeys: z.array(z.string()),
});

type ModelManagerState = z.infer<typeof zModelManagerState>;

const getInitialState = (): ModelManagerState => ({
  _version: 1,
  selectedModelKey: null,
  selectedModelMode: 'view',
  filteredModelType: null,
  searchTerm: '',
  scanPath: undefined,
  shouldInstallInPlace: true,
  selectedModelKeys: [],
});

const slice = createSlice({
  name: 'modelmanagerV2',
  initialState: getInitialState(),
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
    modelSelectionChanged: (state, action: PayloadAction<string[]>) => {
      state.selectedModelKeys = action.payload;
    },
    toggleModelSelection: (state, action: PayloadAction<string>) => {
      const index = state.selectedModelKeys.indexOf(action.payload);
      if (index > -1) {
        state.selectedModelKeys.splice(index, 1);
      } else {
        state.selectedModelKeys.push(action.payload);
      }
    },
    clearModelSelection: (state) => {
      state.selectedModelKeys = [];
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
  modelSelectionChanged,
  toggleModelSelection,
  clearModelSelection,
} = slice.actions;

export const modelManagerSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zModelManagerState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zModelManagerState.parse(state);
    },
    persistDenylist: ['selectedModelKey', 'selectedModelMode', 'filteredModelType', 'searchTerm', 'selectedModelKeys'],
  },
};

export const selectModelManagerV2Slice = (state: RootState) => state.modelmanagerV2;

export const createModelManagerSelector = <T>(selector: (state: ModelManagerState) => T) =>
  createSelector(selectModelManagerV2Slice, selector);

export const selectSelectedModelKey = createModelManagerSelector((modelManager) => modelManager.selectedModelKey);
export const selectSelectedModelMode = createModelManagerSelector((modelManager) => modelManager.selectedModelMode);
export const selectSearchTerm = createModelManagerSelector((mm) => mm.searchTerm);
export const selectFilteredModelType = createModelManagerSelector((mm) => mm.filteredModelType);
export const selectShouldInstallInPlace = createModelManagerSelector((mm) => mm.shouldInstallInPlace);
export const selectSelectedModelKeys = createModelManagerSelector((mm) => mm.selectedModelKeys);
