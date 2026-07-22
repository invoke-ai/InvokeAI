import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getImageUsage } from 'features/deleteImageModal/store/state';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { imagesApi } from 'services/api/endpoints/images';

export const getDeletedImagesFromDeleteBoardAction = (payload: unknown): string[] => {
  if (!payload || typeof payload !== 'object') {
    return [];
  }
  const response = payload as { deleted_images?: unknown; data?: unknown };
  let deletedImages = response.deleted_images;
  if (deletedImages === undefined && response.data && typeof response.data === 'object') {
    const detail = (response.data as { detail?: unknown }).detail;
    if (detail && typeof detail === 'object') {
      deletedImages = (detail as { deleted_images?: unknown }).deleted_images;
    }
  }
  return Array.isArray(deletedImages) && deletedImages.every((name) => typeof name === 'string') ? deletedImages : [];
};

export const addDeleteBoardAndImagesFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(
      imagesApi.endpoints.deleteBoardAndImages.matchFulfilled,
      imagesApi.endpoints.deleteBoardAndImages.matchRejected
    ),
    effect: (action, { dispatch, getState }) => {
      const deletedImages = getDeletedImagesFromDeleteBoardAction(action.payload);

      // Remove all deleted images from the UI

      let wasNodeEditorReset = false;

      const state = getState();
      const nodes = selectNodesSlice(state);
      const canvas = selectCanvasSlice(state);
      const upscale = selectUpscaleSlice(state);
      const refImages = selectRefImagesSlice(state);

      deletedImages.forEach((image_name) => {
        const imageUsage = getImageUsage(nodes, canvas, upscale, refImages, image_name);

        if (imageUsage.isNodesImage && !wasNodeEditorReset) {
          dispatch(nodeEditorReset());
          wasNodeEditorReset = true;
        }
      });
    },
  });
};
