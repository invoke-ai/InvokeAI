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
  stagingAreaImageAccepted: (state, _: PayloadAction<{ imageDTO: ImageDTO }>) => state,
  stagingAreaReset: (state) => {
    state.stagingArea = null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
