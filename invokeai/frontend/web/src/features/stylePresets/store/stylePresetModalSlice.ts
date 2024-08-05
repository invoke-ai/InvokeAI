import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { StylePresetRecordDTO } from 'services/api/endpoints/stylePresets';

import type { StylePresetModalState } from './types';


export const initialState: StylePresetModalState = {
    isModalOpen: false,
    updatingStylePreset: null,
};


export const stylePresetModalSlice = createSlice({
    name: 'stylePresetModal',
    initialState: initialState,
    reducers: {
        isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
            state.isModalOpen = action.payload;
        },
        updatingStylePresetChanged: (state, action: PayloadAction<StylePresetRecordDTO | null>) => {
            state.updatingStylePreset = action.payload;
        },
    },
});

export const { isModalOpenChanged, updatingStylePresetChanged } = stylePresetModalSlice.actions;

export const selectStylePresetModalSlice = (state: RootState) => state.stylePresetModal;
