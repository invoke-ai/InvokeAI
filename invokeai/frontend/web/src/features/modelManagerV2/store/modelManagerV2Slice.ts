import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';


type ModelManagerState = {
    _version: 1;
    selectedModelKey: string | null;
    searchTerm: string;
    filteredModelType: string | null;
};

export const initialModelManagerState: ModelManagerState = {
    _version: 1,
    selectedModelKey: null,
    filteredModelType: null,
    searchTerm: ""
};

export const modelManagerV2Slice = createSlice({
    name: 'modelmanagerV2',
    initialState: initialModelManagerState,
    reducers: {
        setSelectedModelKey: (state, action: PayloadAction<string | null>) => {
            state.selectedModelKey = action.payload;
        },
        setSearchTerm: (state, action: PayloadAction<string>) => {
            state.searchTerm = action.payload;
        },

        setFilteredModelType: (state, action: PayloadAction<string | null>) => {
            state.filteredModelType = action.payload;
        },
    },
});

export const { setSelectedModelKey, setSearchTerm, setFilteredModelType } = modelManagerV2Slice.actions;

export const selectModelManagerSlice = (state: RootState) => state.modelmanager;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateModelManagerState = (state: any): any => {
    if (!('_version' in state)) {
        state._version = 1;
    }
    return state;
};

export const modelManagerPersistConfig: PersistConfig<ModelManagerState> = {
    name: modelManagerV2Slice.name,
    initialState: initialModelManagerState,
    migrate: migrateModelManagerState,
    persistDenylist: [],
};
