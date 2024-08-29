import { createAction, createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { canvasSlice } from 'features/controlLayers/store/canvasSlice';
import type { StagingAreaImage } from 'features/controlLayers/store/types';

export type CanvasSessionState = {
  sendToCanvas: boolean;
  isStaging: boolean;
  stagedImages: StagingAreaImage[];
  selectedStagedImageIndex: number;
};

const initialState: CanvasSessionState = {
  sendToCanvas: false,
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
    sessionSendToCanvasChanged: (state, action: PayloadAction<boolean>) => {
      state.sendToCanvas = action.payload;
    },
  },
});

export const {
  sessionStartedStaging,
  sessionImageStaged,
  sessionStagedImageDiscarded,
  sessionStagingAreaReset,
  sessionNextStagedImageSelected,
  sessionPrevStagedImageSelected,
  sessionSendToCanvasChanged,
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
export const selectIsComposing = createSelector(
  selectCanvasSessionSlice,
  (canvasSession) => canvasSession.sendToCanvas
);
