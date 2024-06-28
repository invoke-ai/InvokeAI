import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State, Rect } from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';

export const stagingAreaReducers = {
  stagingAreaInitialized: (state, action: PayloadAction<{ bbox: Rect; batchIds: string[] }>) => {
    const { bbox, batchIds } = action.payload;
    state.stagingArea = {
      bbox,
      batchIds,
      selectedImageIndex: null,
      images: [],
    };
    // When we start staging, the user should not be interacting with the stage except to move it around. Set the tool
    // to view.
    state.tool.selectedBuffer = state.tool.selected;
    state.tool.selected = 'view';
  },
  stagingAreaImageAdded: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    if (!state.stagingArea) {
      // Should not happen
      return;
    }
    state.stagingArea.images.push(imageDTO);
    if (!state.stagingArea.selectedImageIndex) {
      state.stagingArea.selectedImageIndex = state.stagingArea.images.length - 1;
    }
  },
  stagingAreaNextImageSelected: (state) => {
    if (!state.stagingArea) {
      // Should not happen
      return;
    }
    if (state.stagingArea.selectedImageIndex === null) {
      if (state.stagingArea.images.length > 0) {
        state.stagingArea.selectedImageIndex = 0;
      }
      return;
    }
    state.stagingArea.selectedImageIndex = (state.stagingArea.selectedImageIndex + 1) % state.stagingArea.images.length;
  },
  stagingAreaPreviousImageSelected: (state) => {
    if (!state.stagingArea) {
      // Should not happen
      return;
    }
    if (state.stagingArea.selectedImageIndex === null) {
      if (state.stagingArea.images.length > 0) {
        state.stagingArea.selectedImageIndex = 0;
      }
      return;
    }
    state.stagingArea.selectedImageIndex =
      (state.stagingArea.selectedImageIndex - 1 + state.stagingArea.images.length) % state.stagingArea.images.length;
  },
  stagingAreaBatchIdAdded: (state, action: PayloadAction<{ batchId: string }>) => {
    const { batchId } = action.payload;
    if (!state.stagingArea) {
      // Should not happen
      return;
    }
    state.stagingArea.batchIds.push(batchId);
  },
  stagingAreaImageDiscarded: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    if (!state.stagingArea) {
      // Should not happen
      return;
    }
    state.stagingArea.images = state.stagingArea.images.filter((image) => image.image_name !== imageDTO.image_name);
  },
  stagingAreaImageAccepted: (state, _: PayloadAction<{ imageDTO: ImageDTO }>) => {
    // When we finish staging, reset the tool back to the previous selection.
    if (state.tool.selectedBuffer) {
      state.tool.selected = state.tool.selectedBuffer;
      state.tool.selectedBuffer = null;
    }
  },
  stagingAreaReset: (state) => {
    state.stagingArea = null;
    // When we finish staging, reset the tool back to the previous selection.
    if (state.tool.selectedBuffer) {
      state.tool.selected = state.tool.selectedBuffer;
      state.tool.selectedBuffer = null;
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
