import { createSelector } from '@reduxjs/toolkit';
import { selectUiSlice } from 'features/ui/store/uiSlice';

export const selectShouldShowItemDetails = createSelector(selectUiSlice, (ui) => ui.shouldShowItemDetails);
export const selectShouldShowProgressInViewer = createSelector(selectUiSlice, (ui) => ui.shouldShowProgressInViewer);
export const selectPickerCompactViewStates = createSelector(selectUiSlice, (ui) => ui.pickerCompactViewStates);
