import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { canvasReset } from 'features/controlLayers/store/canvasSlice';
import type { StagingAreaImage } from 'features/controlLayers/store/types';

type CanvasStagingAreaState = {
  isStaging: boolean;
  stagedImages: StagingAreaImage[];
  selectedStagedImageIndex: number;
};

const initialState: CanvasStagingAreaState = {
  isStaging: false,
  stagedImages: [],
  selectedStagedImageIndex: 0,
};

export const canvasStagingAreaSlice = createSlice({
  name: 'canvasStagingArea',
  initialState,
  reducers: {
    stagingAreaStartedStaging: (state) => {
      state.isStaging = true;
      state.selectedStagedImageIndex = 0;
    },
    stagingAreaImageStaged: (state, action: PayloadAction<{ stagingAreaImage: StagingAreaImage }>) => {
      const { stagingAreaImage } = action.payload;
      state.isStaging = true;
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
      if (state.stagedImages.length === 0) {
        state.isStaging = false;
      }
    },
    stagingAreaReset: (state) => {
      state.isStaging = false;
      state.stagedImages = [];
      state.selectedStagedImageIndex = 0;
    },
    stagingAreaImageAccepted: (_state, _action: PayloadAction<{ index: number }>) => {
      // no-op, handled in a listener
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, () => deepClone(initialState));
  },
});

export const {
  stagingAreaStartedStaging,
  stagingAreaImageStaged,
  stagingAreaStagedImageDiscarded,
  stagingAreaReset,
  stagingAreaNextStagedImageSelected,
  stagingAreaPrevStagedImageSelected,
  stagingAreaImageAccepted,
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

export const selectIsStaging = createSelector(selectCanvasStagingAreaSlice, (stagingaArea) => stagingaArea.isStaging);
