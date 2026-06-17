import { createSelector } from '@reduxjs/toolkit';
import { selectUiSlice } from 'features/ui/store/uiSlice';

export const selectActiveTab = createSelector(selectUiSlice, (ui) => ui.activeTab);
export const selectShouldShowItemDetails = createSelector(selectUiSlice, (ui) => ui.shouldShowItemDetails);
export const selectShouldShowProgressInViewer = createSelector(selectUiSlice, (ui) => ui.shouldShowProgressInViewer);
export const selectShouldUsePagedGalleryView = createSelector(selectUiSlice, (ui) => ui.shouldUsePagedGalleryView);
// TODO: One day maybe this'll be by board or something?
export const selectStageViewerMode = createSelector(selectUiSlice, (ui) => ui.stageViewerMode);
export const selectPickerCompactViewStates = createSelector(selectUiSlice, (ui) => ui.pickerCompactViewStates);
