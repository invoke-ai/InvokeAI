import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

import type { StylePresetModalState, StylePresetPrefillOptions } from './types';
import { ImageDTO } from '../../../services/api/types';


export const initialState: StylePresetModalState = {
    isModalOpen: false,
    updatingStylePreset: null,
    createPresetFromImage: null
};


export const stylePresetModalSlice = createSlice({
    name: 'stylePresetModal',
    initialState: initialState,
    reducers: {
        isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
            state.isModalOpen = action.payload;
        },
        updatingStylePresetChanged: (state, action: PayloadAction<StylePresetRecordWithImage | null>) => {
            state.updatingStylePreset = action.payload;
        },
        createPresetFromImageChanged: (state, action: PayloadAction<ImageDTO | null>) => {
            state.createPresetFromImage = action.payload;
        },
    },
});

export const { isModalOpenChanged, updatingStylePresetChanged, createPresetFromImageChanged } = stylePresetModalSlice.actions;

export const selectStylePresetModalSlice = (state: RootState) => state.stylePresetModal;
