import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';

export const stagingAreaReducers = {
  stagingAreaStartedStaging: (state) => {
    state.stagingArea.isStaging = true;
    state.stagingArea.selectedImageIndex = 0;
    // When we start staging, the user should not be interacting with the stage except to move it around. Set the tool
    // to view.
    state.tool.selectedBuffer = state.tool.selected;
    state.tool.selected = 'view';
  },
  stagingAreaImageAdded: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    state.stagingArea.images.push(imageDTO);
    state.stagingArea.selectedImageIndex = state.stagingArea.images.length - 1;
  },
  stagingAreaNextImageSelected: (state) => {
    state.stagingArea.selectedImageIndex = (state.stagingArea.selectedImageIndex + 1) % state.stagingArea.images.length;
  },
  stagingAreaPreviousImageSelected: (state) => {
    state.stagingArea.selectedImageIndex =
      (state.stagingArea.selectedImageIndex - 1 + state.stagingArea.images.length) % state.stagingArea.images.length;
  },
  stagingAreaImageDiscarded: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    state.stagingArea.images = state.stagingArea.images.filter((image) => image.image_name !== imageDTO.image_name);
    state.stagingArea.selectedImageIndex = Math.min(
      state.stagingArea.selectedImageIndex,
      state.stagingArea.images.length - 1
    );
    if (state.stagingArea.images.length === 0) {
      state.stagingArea.isStaging = false;
    }
  },
  stagingAreaImageAccepted: (state, _: PayloadAction<{ imageDTO: ImageDTO }>) => {
    // When we finish staging, reset the tool back to the previous selection.
    state.stagingArea.isStaging = false;
    state.stagingArea.images = [];
    state.stagingArea.selectedImageIndex = 0;
    if (state.tool.selectedBuffer) {
      state.tool.selected = state.tool.selectedBuffer;
      state.tool.selectedBuffer = null;
    }
  },
  stagingAreaCanceledStaging: (state) => {
    state.stagingArea.isStaging = false;
    state.stagingArea.images = [];
    state.stagingArea.selectedImageIndex = 0;
    // When we finish staging, reset the tool back to the previous selection.
    if (state.tool.selectedBuffer) {
      state.tool.selected = state.tool.selectedBuffer;
      state.tool.selectedBuffer = null;
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
