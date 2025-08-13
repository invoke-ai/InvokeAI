import type { AppStartListening } from 'app/store/store';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getImageUsage } from 'features/deleteImageModal/store/state';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addDeleteBoardAndImagesFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteBoardAndImages.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const { deleted_resources } = action.payload;

      // Remove all deleted images from the UI

      let wasNodeEditorReset = false;

      const state = getState();
      const nodes = selectNodesSlice(state);
      const canvas = selectCanvasSlice(state);
      const upscale = selectUpscaleSlice(state);
      const refImages = selectRefImagesSlice(state);

      deleted_resources.forEach((resource_id) => {
        const imageUsage = getImageUsage(nodes, canvas, upscale, refImages, resource_id);

        if (imageUsage.isNodesImage && !wasNodeEditorReset) {
          dispatch(nodeEditorReset());
          wasNodeEditorReset = true;
        }
      });
    },
  });
};
