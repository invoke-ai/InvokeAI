import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  canvasReset,
  newAdvancedCanvasSessionRequested,
  newSimpleCanvasSessionRequested,
} from 'features/controlLayers/store/actions';
import type { StagingAreaImage } from 'features/controlLayers/store/types';
import { selectCanvasQueueCounts } from 'services/api/endpoints/queue';

type CanvasStagingAreaState = {
  sessionType: 'simple' | 'advanced' | null;
  images: StagingAreaImage[];
  selectedImageIndex: number;
};

const initialState: CanvasStagingAreaState = {
  sessionType: null,
  images: [],
  selectedImageIndex: 0,
};

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState,
  reducers: {
    stagingAreaImageStaged: (state, action: PayloadAction<{ stagingAreaImage: StagingAreaImage }>) => {
      const { stagingAreaImage } = action.payload;
      state.images.push(stagingAreaImage);
      state.selectedImageIndex = state.images.length - 1;
    },
    stagingAreaImageSelected: (state, action: PayloadAction<{ index: number }>) => {
      const { index } = action.payload;
      state.selectedImageIndex = index;
    },
    stagingAreaNextStagedImageSelected: (state) => {
      state.selectedImageIndex = (state.selectedImageIndex + 1) % state.images.length;
    },
    stagingAreaPrevStagedImageSelected: (state) => {
      state.selectedImageIndex = (state.selectedImageIndex - 1 + state.images.length) % state.images.length;
    },
    stagingAreaStagedImageDiscarded: (state, action: PayloadAction<{ index: number }>) => {
      const { index } = action.payload;
      state.images.splice(index, 1);
      state.selectedImageIndex = Math.min(state.selectedImageIndex, state.images.length - 1);
    },
    stagingAreaReset: (state) => {
      state.images = [];
      state.selectedImageIndex = 0;
    },
    canvasSessionStarted: (state, action: PayloadAction<{ sessionType: CanvasStagingAreaState['sessionType'] }>) => {
      const { sessionType } = action.payload;
      state.sessionType = sessionType;
      state.images = [];
      state.selectedImageIndex = 0;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, () => deepClone(initialState));
    builder.addCase(newSimpleCanvasSessionRequested, () => {
      const state = deepClone(initialState);
      state.sessionType === 'simple';
      return state;
    });
    builder.addCase(newAdvancedCanvasSessionRequested, () => {
      const state = deepClone(initialState);
      state.sessionType === 'advanced';
      return state;
    });
  },
});

export const {
  stagingAreaImageStaged,
  stagingAreaStagedImageDiscarded,
  stagingAreaReset,
  stagingAreaImageSelected,
  stagingAreaNextStagedImageSelected,
  stagingAreaPrevStagedImageSelected,
  canvasSessionStarted,
} = canvasSessionSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasStagingAreaPersistConfig: PersistConfig<CanvasStagingAreaState> = {
  name: canvasSessionSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};

export const selectCanvasStagingAreaSlice = (s: RootState) => s[canvasSessionSlice.name];

/**
 * Selects if we should be staging images. This is true if:
 * - There are staged images.
 * - There are any in-progress or pending canvas queue items.
 */
export const selectIsStaging = createSelector(
  selectCanvasQueueCounts,
  selectCanvasStagingAreaSlice,
  ({ data }, staging) => {
    if (staging.images.length > 0) {
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
  (stagingArea) => stagingArea.selectedImageIndex
);
export const selectSelectedImage = createSelector(
  [selectCanvasStagingAreaSlice, selectStagedImageIndex],
  (stagingArea, index) => stagingArea.images[index] ?? null
);
export const selectStagedImages = createSelector(selectCanvasStagingAreaSlice, (stagingArea) => stagingArea.images);
export const selectImageCount = createSelector(
  selectCanvasStagingAreaSlice,
  (stagingArea) => stagingArea.images.length
);
export const selectCanvasSessionType = createSelector(
  selectCanvasStagingAreaSlice,
  (canvasSession) => canvasSession.sessionType
);
