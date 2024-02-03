import { createSelector } from '@reduxjs/toolkit';
import { selectSystemSlice } from 'features/system/store/systemSlice';

export const languageSelector = createSelector(selectSystemSlice, (system) => system.language);
