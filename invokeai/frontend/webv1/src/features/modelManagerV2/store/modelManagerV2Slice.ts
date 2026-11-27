import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { zModelType } from 'features/nodes/types/common';
import { assert } from 'tsafe';
import z from 'zod';

const zModelCategoryType = zModelType
  .exclude(['onnx', 'prompt_enhancer'])
  .or(z.literal('refiner'))
  .or(z.literal('external_image_generator'));
export type ModelCategoryType = z.infer<typeof zModelCategoryType>;

const zFilterableModelType = zModelCategoryType.or(z.literal('missing'));
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
  orderBy: z
    .enum(['default', 'name', 'type', 'base', 'size', 'created_at', 'updated_at', 'path', 'format'])
    .default('name'),
  sortDirection: z.enum(['asc', 'desc']).default('asc'),
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
  orderBy: 'name',
  sortDirection: 'asc',
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
    setOrderBy: (state, action: PayloadAction<ModelManagerState['orderBy']>) => {
      state.orderBy = action.payload;
    },
    setSortDirection: (state, action: PayloadAction<ModelManagerState['sortDirection']>) => {
      state.sortDirection = action.payload;
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
  setOrderBy,
  setSortDirection,
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
export const selectOrderBy = createModelManagerSelector((mm) => mm.orderBy);
export const selectSortDirection = createModelManagerSelector((mm) => mm.sortDirection);
