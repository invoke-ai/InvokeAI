import { createSelector } from '@reduxjs/toolkit';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';

export const selectEntityCount = createSelector(selectCanvasV2Slice, (canvasV2) => {
  return (
    canvasV2.regions.length + canvasV2.controlAdapters.length + canvasV2.ipAdapters.length + canvasV2.layers.length
  );
});

export const selectOptimalDimension = createSelector(selectCanvasV2Slice, (canvasV2) => {
  return getOptimalDimension(canvasV2.params.model);
});
