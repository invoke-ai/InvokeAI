import { createSelector } from '@reduxjs/toolkit';

import { selectCanvasSlice } from './canvasSlice';

export const isStagingSelector = createSelector(
  selectCanvasSlice,
  (canvas) => canvas.batchIds.length > 0 || canvas.layerState.stagingArea.images.length > 0
);
