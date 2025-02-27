import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { canvasReset } from 'features/controlLayers/store/actions';
import type { StagingAreaImage } from 'features/controlLayers/store/types';
import { selectCanvasQueueCounts } from 'services/api/endpoints/queue';

import { newSessionRequested } from './actions';

type CanvasStagingAreaState = {
  stagedImages: StagingAreaImage[];
  selectedStagedImageIndex: number;
};

const initialState: CanvasStagingAreaState = {
  stagedImages: [],
  selectedStagedImageIndex: 0,
};

export const canvasStagingAreaSlice = createSlice({
  name: 'canvasStagingArea',
  initialState,
  reducers: {
    stagingAreaImageStaged: (state, action: PayloadAction<{ stagingAreaImage: StagingAreaImage }>) => {
      const { stagingAreaImage } = action.payload;
      state.stagedImages.push(stagingAreaImage);
      state.selectedStagedImageIndex = state.stagedImages.length - 1;
    },
    stagingAreaNextStagedImageSelected: (state) => {
      state.selectedStagedImageIndex = (state.selectedStagedImageIndex + 1) % state.stagedImages.length;
    },
    stagingAreaPrevStagedImageSelected: (state) => {
      state.selectedStagedImageIndex =
        (state.selectedStagedImageIndex - 1 + state.stagedImages.length) % state.stagedImages.length;
    },
    stagingAreaStagedImageDiscarded: (state, action: PayloadAction<{ index: number }>) => {
      const { index } = action.payload;
      state.stagedImages.splice(index, 1);
      state.selectedStagedImageIndex = Math.min(state.selectedStagedImageIndex, state.stagedImages.length - 1);
    },
    stagingAreaReset: (state) => {
      state.stagedImages = [];
      state.selectedStagedImageIndex = 0;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, () => deepClone(initialState));
    builder.addMatcher(newSessionRequested, () => deepClone(initialState));
  },
});

export const {
  stagingAreaImageStaged,
  stagingAreaStagedImageDiscarded,
  stagingAreaReset,
  stagingAreaNextStagedImageSelected,
  stagingAreaPrevStagedImageSelected,
} = canvasStagingAreaSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasStagingAreaPersistConfig: PersistConfig<CanvasStagingAreaState> = {
  name: canvasStagingAreaSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};

export const selectCanvasStagingAreaSlice = (s: RootState) => s.canvasStagingArea;

/**
 * Selects if we should be staging images. This is true if:
 * - There are staged images.
 * - There are any in-progress or pending canvas queue items.
 */
export const selectIsStaging = createSelector(
  selectCanvasQueueCounts,
  selectCanvasStagingAreaSlice,
  ({ data }, staging) => {
    if (staging.stagedImages.length > 0) {
      return true;
    }
    if (!data) {
      return false;
    }
    return data.in_progress > 0 || data.pending > 0;
  }
);
export const selectStagedImageIndex = createSelector(
  selectCanvasStagingAreaSlice,
  (stagingArea) => stagingArea.selectedStagedImageIndex
);
export const selectSelectedImage = createSelector(
  [selectCanvasStagingAreaSlice, selectStagedImageIndex],
  (stagingArea, index) => stagingArea.stagedImages[index] ?? null
);
export const selectImageCount = createSelector(
  selectCanvasStagingAreaSlice,
  (stagingArea) => stagingArea.stagedImages.length
);
