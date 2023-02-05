import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
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
