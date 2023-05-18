import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { systemSelector } from 'features/system/store/systemSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { CanvasImage, CanvasState, isCanvasBaseImage } from './canvasTypes';

export const canvasSelector = (state: RootState): CanvasState => state.canvas;

export const isStagingSelector = createSelector(
  [canvasSelector, activeTabNameSelector, systemSelector],
  (canvas, activeTabName, system) =>
    canvas.layerState.stagingArea.images.length > 0 ||
    (activeTabName === 'unifiedCanvas' && system.isProcessing)
);

export const initialCanvasImageSelector = (
  state: RootState
): CanvasImage | undefined =>
  state.canvas.layerState.objects.find(isCanvasBaseImage);
