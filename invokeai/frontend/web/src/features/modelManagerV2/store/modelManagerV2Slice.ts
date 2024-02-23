import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';


type ModelManagerState = {
    _version: 1;
    selectedModelKey: string | null;
    selectedModelMode: "edit" | "view",
    searchTerm: string;
    filteredModelType: string | null;
};

export const initialModelManagerState: ModelManagerState = {
    _version: 1,
    selectedModelKey: null,
    selectedModelMode: "view",
    filteredModelType: null,
    searchTerm: ""
};

export const modelManagerV2Slice = createSlice({
    name: 'modelmanagerV2',
    initialState: initialModelManagerState,
    reducers: {
        setSelectedModelKey: (state, action: PayloadAction<string | null>) => {
            state.selectedModelMode = "view"
            state.selectedModelKey = action.payload;
        },
        setSelectedModelMode: (state, action: PayloadAction<"view" | "edit">) => {
            state.selectedModelMode = action.payload;
        },
        setSearchTerm: (state, action: PayloadAction<string>) => {
            state.searchTerm = action.payload;
        },

        setFilteredModelType: (state, action: PayloadAction<string | null>) => {
            state.filteredModelType = action.payload;
        },
    },
});

export const { setSelectedModelKey, setSearchTerm, setFilteredModelType, setSelectedModelMode } = modelManagerV2Slice.actions;

export const selectModelManagerSlice = (state: RootState) => state.modelmanager;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateModelManagerState = (state: any): any => {
    if (!('_version' in state)) {
        state._version = 1;
    }
    return state;
};

export const modelManagerV2PersistConfig: PersistConfig<ModelManagerState> = {
    name: modelManagerV2Slice.name,
    initialState: initialModelManagerState,
    migrate: migrateModelManagerState,
    persistDenylist: [],
};
