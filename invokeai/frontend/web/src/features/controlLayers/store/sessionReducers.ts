import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';

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
  sessionImageStaged: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    state.session.stagedImages.push(imageDTO);
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
  sessionStagedImageDiscarded: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    state.session.stagedImages = state.session.stagedImages.filter((image) => image.image_name !== imageDTO.image_name);
    state.session.selectedStagedImageIndex = Math.min(
      state.session.selectedStagedImageIndex,
      state.session.stagedImages.length - 1
    );
    if (state.session.stagedImages.length === 0) {
      state.session.isStaging = false;
    }
  },
  sessionStagedImageAccepted: (state, _: PayloadAction<{ imageDTO: ImageDTO }>) => {
    // When we finish staging, reset the tool back to the previous selection.
    state.session.isStaging = false;
    state.session.stagedImages = [];
    state.session.selectedStagedImageIndex = 0;
    if (state.tool.selectedBuffer) {
      state.tool.selected = state.tool.selectedBuffer;
      state.tool.selectedBuffer = null;
    }
  },
  sessionStagingCanceled: (state) => {
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
