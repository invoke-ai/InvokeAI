import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State, StagingAreaImage } from 'features/controlLayers/store/types';

export const sessionReducers = {
  sessionStarted: (state) => {
    state.session.isActive = true;
    state.selectedEntityIdentifier = { id: 'inpaint_mask', type: 'inpaint_mask' };
  },
  sessionStartedStaging: (state) => {
    state.session.isStaging = true;
    state.session.selectedStagedImageIndex = 0;
    // When we start staging, the user should not be interacting with the stage except to move it around. Set the tool
    // to view.
    state.tool.selectedBuffer = state.tool.selected;
    state.tool.selected = 'view';
  },
  sessionImageStaged: (state, action: PayloadAction<StagingAreaImage>) => {
    const { imageDTO, rect } = action.payload;
    state.session.stagedImages.push({ imageDTO, rect });
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
    state.session.stagedImages = state.session.stagedImages.splice(index, 1);
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
    // When we finish staging, reset the tool back to the previous selection.
    if (state.tool.selectedBuffer) {
      state.tool.selected = state.tool.selectedBuffer;
      state.tool.selectedBuffer = null;
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
