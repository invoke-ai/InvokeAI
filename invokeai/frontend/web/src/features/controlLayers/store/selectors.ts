import { createSelector } from '@reduxjs/toolkit';
import { selectControlAdaptersV2Slice } from 'features/controlLayers/store/controlAdaptersSlice';
import { selectIPAdaptersSlice } from 'features/controlLayers/store/ipAdaptersSlice';
import { selectLayersSlice } from 'features/controlLayers/store/layersSlice';
import { selectRegionalGuidanceSlice } from 'features/controlLayers/store/regionalGuidanceSlice';

export const selectEntityCount = createSelector(
  selectRegionalGuidanceSlice,
  selectControlAdaptersV2Slice,
  selectIPAdaptersSlice,
  selectLayersSlice,
  (rgState, caState, ipaState, layersState) => {
    return (
      rgState.regions.length + caState.controlAdapters.length + ipaState.ipAdapters.length + layersState.layers.length
    );
  }
);
