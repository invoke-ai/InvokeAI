import { createSelector } from '@reduxjs/toolkit';
import { selectUiSlice } from 'features/ui/store/uiSlice';

export const selectActiveTab = createSelector(selectUiSlice, (ui) => ui.activeTab);
export const selectShouldShowImageDetails = createSelector(selectUiSlice, (ui) => ui.shouldShowImageDetails);
export const selectShouldShowProgressInViewer = createSelector(selectUiSlice, (ui) => ui.shouldShowProgressInViewer);
export const selectActiveTabCanvasRightPanel = createSelector(selectUiSlice, (ui) => ui.activeTabCanvasRightPanel);
export const selectActiveTabCanvasMainPanel = createSelector(selectUiSlice, (ui) => ui.activeTabCanvasMainPanel);
export const selectActiveTabGenerateMainPanel = createSelector(selectUiSlice, (ui) => ui.activeTabGenerateMainPanel);
export const selectActiveTabUpscalingMainPanel = createSelector(selectUiSlice, (ui) => ui.activeTabUpscalingMainPanel);
export const selectActiveTabWorkflowsMainPanel = createSelector(selectUiSlice, (ui) => ui.activeTabWorkflowsMainPanel);
