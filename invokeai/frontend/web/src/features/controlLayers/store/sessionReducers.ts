import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State, SessionMode, StagingAreaImage } from 'features/controlLayers/store/types';

export const sessionReducers = {
  sessionStartedStaging: (state) => {
    state.session.isStaging = true;
    state.session.selectedStagedImageIndex = 0;
  },
  sessionImageStaged: (state, action: PayloadAction<{ stagingAreaImage: StagingAreaImage }>) => {
    const { stagingAreaImage } = action.payload;
    state.session.stagedImages.push(stagingAreaImage);
    state.session.selectedStagedImageIndex = state.session.stagedImages.length - 1;
  },
  sessionNextStagedImageSelected: (state) => {
    state.session.selectedStagedImageIndex =
      (state.session.selectedStagedImageIndex + 1) % state.session.stagedImages.length;
  },
  sessionPrevStagedImageSelected: (state) => {
    state.session.selectedStagedImageIndex =
      (state.session.selectedStagedImageIndex - 1 + state.session.stagedImages.length) %
      state.session.stagedImages.length;
  },
  sessionStagedImageDiscarded: (state, action: PayloadAction<{ index: number }>) => {
    const { index } = action.payload;
    state.session.stagedImages.splice(index, 1);
    state.session.selectedStagedImageIndex = Math.min(
      state.session.selectedStagedImageIndex,
      state.session.stagedImages.length - 1
    );
    if (state.session.stagedImages.length === 0) {
      state.session.isStaging = false;
    }
  },
  sessionStagingAreaReset: (state) => {
    state.session.isStaging = false;
    state.session.stagedImages = [];
    state.session.selectedStagedImageIndex = 0;
  },
  sessionModeChanged: (state, action: PayloadAction<{ mode: SessionMode }>) => {
    const { mode } = action.payload;
    state.session.mode = mode;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
