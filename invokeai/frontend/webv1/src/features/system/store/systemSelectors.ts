import { createSelector } from '@reduxjs/toolkit';
import { selectSystemSlice } from 'features/system/store/systemSlice';

export const selectLanguage = createSelector(selectSystemSlice, (system) => system.language);
