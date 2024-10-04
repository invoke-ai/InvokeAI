import { createSelector } from '@reduxjs/toolkit';
import { selectConfigSlice } from 'features/system/store/configSlice';

export const selectAllowPrivateBoards = createSelector(selectConfigSlice, (config) => config.allowPrivateBoards);
