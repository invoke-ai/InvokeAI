import { createSelector } from '@reduxjs/toolkit';
import { selectUiSlice } from 'features/ui/store/uiSlice';

export const selectActiveTab = createSelector(selectUiSlice, (ui) => ui.activeTab);
export const selectShouldShowImageDetails = createSelector(selectUiSlice, (ui) => ui.shouldShowImageDetails);
export const selectShouldShowProgressInViewer = createSelector(selectUiSlice, (ui) => ui.shouldShowProgressInViewer);
export const selectActiveTabCanvasRightPanel = createSelector(selectUiSlice, (ui) => ui.activeTabCanvasRightPanel);
