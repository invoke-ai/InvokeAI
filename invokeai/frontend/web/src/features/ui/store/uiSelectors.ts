import { createSelector } from '@reduxjs/toolkit';
import { selectUiSlice } from 'features/ui/store/uiSlice';

export const selectActiveTab = createSelector(selectUiSlice, (ui) => ui.activeTab);
