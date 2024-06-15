import { createSelector } from '@reduxjs/toolkit';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';

export const selectEntityCount = createSelector(selectCanvasSlice, (canvasV2) => {
  return (
    canvasV2.regions.length + canvasV2.controlAdapters.length + canvasV2.ipAdapters.length + canvasV2.layers.length
  );
});

export const selectOptimalDimension = createSelector(selectCanvasSlice, (canvasV2) => {
  return getOptimalDimension(canvasV2.params.model);
});
