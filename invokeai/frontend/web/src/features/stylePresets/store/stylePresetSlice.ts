import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { StylePresetRecordDTO } from 'services/api/endpoints/stylePresets';

import type { StylePresetState } from './types';


export const initialState: StylePresetState = {
    isMenuOpen: false,
    activeStylePreset: null,
    calculatedPosPrompt: undefined,
    calculatedNegPrompt: undefined
};


export const stylePresetSlice = createSlice({
    name: 'stylePreset',
    initialState: initialState,
    reducers: {
        isMenuOpenChanged: (state, action: PayloadAction<boolean>) => {
            state.isMenuOpen = action.payload;
        },
        activeStylePresetChanged: (state, action: PayloadAction<StylePresetRecordDTO | null>) => {
            state.activeStylePreset = action.payload;
        },
        calculatedPosPromptChanged: (state, action: PayloadAction<string | undefined>) => {
            state.calculatedPosPrompt = action.payload;
        },
        calculatedNegPromptChanged: (state, action: PayloadAction<string | undefined>) => {
            state.calculatedNegPrompt = action.payload;
        },
    },
});

export const { isMenuOpenChanged, activeStylePresetChanged, calculatedPosPromptChanged, calculatedNegPromptChanged } = stylePresetSlice.actions;

export const selectStylePresetSlice = (state: RootState) => state.stylePreset;
