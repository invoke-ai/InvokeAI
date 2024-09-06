import { createAction, createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { canvasReset, canvasSlice } from 'features/controlLayers/store/canvasSlice';
import type { StagingAreaImage } from 'features/controlLayers/store/types';

type CanvasSessionState = {
  isStaging: boolean;
  stagedImages: StagingAreaImage[];
  selectedStagedImageIndex: number;
};

const initialState: CanvasSessionState = {
  isStaging: false,
  stagedImages: [],
  selectedStagedImageIndex: 0,
};

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState,
  reducers: {
    sessionStartedStaging: (state) => {
      state.isStaging = true;
      state.selectedStagedImageIndex = 0;
    },
    sessionImageStaged: (state, action: PayloadAction<{ stagingAreaImage: StagingAreaImage }>) => {
      const { stagingAreaImage } = action.payload;
      state.isStaging = true;
      state.stagedImages.push(stagingAreaImage);
      state.selectedStagedImageIndex = state.stagedImages.length - 1;
    },
    sessionNextStagedImageSelected: (state) => {
      state.selectedStagedImageIndex = (state.selectedStagedImageIndex + 1) % state.stagedImages.length;
    },
    sessionPrevStagedImageSelected: (state) => {
      state.selectedStagedImageIndex =
        (state.selectedStagedImageIndex - 1 + state.stagedImages.length) % state.stagedImages.length;
    },
    sessionStagedImageDiscarded: (state, action: PayloadAction<{ index: number }>) => {
      const { index } = action.payload;
      state.stagedImages.splice(index, 1);
      state.selectedStagedImageIndex = Math.min(state.selectedStagedImageIndex, state.stagedImages.length - 1);
      if (state.stagedImages.length === 0) {
        state.isStaging = false;
      }
    },
    sessionStagingAreaReset: (state) => {
      state.isStaging = false;
      state.stagedImages = [];
      state.selectedStagedImageIndex = 0;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, () => deepClone(initialState));
  },
});

export const {
  sessionStartedStaging,
  sessionImageStaged,
  sessionStagedImageDiscarded,
  sessionStagingAreaReset,
  sessionNextStagedImageSelected,
  sessionPrevStagedImageSelected,
} = canvasSessionSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasSessionPersistConfig: PersistConfig<CanvasSessionState> = {
  name: canvasSessionSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
export const sessionStagingAreaImageAccepted = createAction<{ index: number }>(
  `${canvasSlice.name}/sessionStagingAreaImageAccepted`
);

export const selectCanvasSessionSlice = (s: RootState) => s.canvasSession;

export const selectIsStaging = createSelector(selectCanvasSessionSlice, (canvasSession) => canvasSession.isStaging);
