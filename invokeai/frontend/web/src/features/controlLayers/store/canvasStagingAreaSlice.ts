import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { canvasReset } from 'features/controlLayers/store/actions';
import type { StagingAreaImage, StagingAreaProgressImage } from 'features/controlLayers/store/types';
import { selectCanvasQueueCounts } from 'services/api/endpoints/queue';

type CanvasStagingAreaState = {
  sessionType: 'simple' | 'advanced' | null;
  images: (StagingAreaImage | StagingAreaProgressImage)[];
  selectedImageIndex: number;
};

const INITIAL_STATE: CanvasStagingAreaState = {
  sessionType: null,
  images: [],
  selectedImageIndex: 0,
};

const getInitialState = (): CanvasStagingAreaState => deepClone(INITIAL_STATE);

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState: getInitialState(),
  reducers: {
    stagingAreaImageStaged: (state, action: PayloadAction<{ stagingAreaImage: StagingAreaImage }>) => {
      const { stagingAreaImage } = action.payload;
      let didReplace = false;
      const newImages = [];
      for (const i of state.images) {
        if (i.sessionId === stagingAreaImage.sessionId) {
          newImages.push(stagingAreaImage);
          didReplace = true;
        } else {
          newImages.push(i);
        }
      }
      if (!didReplace) {
        newImages.push(stagingAreaImage);
      }
      state.images = newImages;
    },
    stagingAreaGenerationStarted: (state, action: PayloadAction<{ sessionId: string }>) => {
      const { sessionId } = action.payload;
      state.images.push({ type: 'progress', sessionId });
    },
    stagingAreaGenerationFinished: (state, action: PayloadAction<{ sessionId: string }>) => {
      const { sessionId } = action.payload;
      state.images = state.images.filter((data) => data.sessionId !== sessionId);
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
    canvasSessionStarted: (_, action: PayloadAction<{ sessionType: CanvasStagingAreaState['sessionType'] }>) => {
      const { sessionType } = action.payload;
      const state = getInitialState();
      state.sessionType = sessionType;
      return state;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, () => getInitialState());
  },
});

export const {
  stagingAreaImageStaged,
  stagingAreaGenerationStarted,
  stagingAreaGenerationFinished,
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
  initialState: getInitialState(),
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
