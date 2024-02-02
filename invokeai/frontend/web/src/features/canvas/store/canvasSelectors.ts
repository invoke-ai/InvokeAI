import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectProgressSlice } from 'features/progress/store/progressSlice';

import { selectCanvasSlice } from './canvasSlice';
import { isCanvasBaseImage } from './canvasTypes';

export const isStagingSelector = createSelector(
  selectProgressSlice,
  selectCanvasSlice,
  (progress, canvas) => progress.canvasBatchIds.length > 0 || canvas.layerState.stagingArea.images.length > 0
);

export const initialCanvasImageSelector = createMemoizedSelector(selectCanvasSlice, (canvas) =>
  canvas.layerState.objects.find(isCanvasBaseImage)
);
