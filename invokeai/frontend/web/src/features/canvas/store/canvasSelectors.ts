import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';

import { selectCanvasSlice } from './canvasSlice';
import { isCanvasBaseImage } from './canvasTypes';

export const isStagingSelector = createSelector(
  selectCanvasSlice,
  (canvas) => canvas.batchIds.length > 0 || canvas.layerState.stagingArea.images.length > 0
);

export const initialCanvasImageSelector = createMemoizedSelector(selectCanvasSlice, (canvas) =>
  canvas.layerState.objects.find(isCanvasBaseImage)
);
