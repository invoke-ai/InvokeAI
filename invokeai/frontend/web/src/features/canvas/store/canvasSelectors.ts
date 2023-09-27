import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { CanvasImage, CanvasState, isCanvasBaseImage } from './canvasTypes';

export const canvasSelector = (state: RootState): CanvasState => state.canvas;

export const isStagingSelector = createSelector(
  [stateSelector],
  ({ canvas }) => canvas.batchIds.length > 0
);

export const initialCanvasImageSelector = (
  state: RootState
): CanvasImage | undefined =>
  state.canvas.layerState.objects.find(isCanvasBaseImage);
