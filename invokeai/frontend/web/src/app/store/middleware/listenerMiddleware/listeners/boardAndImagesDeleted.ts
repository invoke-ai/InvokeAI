import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { caAllDeleted, ipaAllDeleted, layerAllDeleted } from 'features/controlLayers/store/canvasV2Slice';
import { getImageUsage } from 'features/deleteImageModal/store/selectors';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addDeleteBoardAndImagesFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteBoardAndImages.matchFulfilled,
    effect: async (action, { dispatch, getState }) => {
      const { deleted_images } = action.payload;

      // Remove all deleted images from the UI

      let wereLayersReset = false;
      let wasNodeEditorReset = false;
      let wereControlAdaptersReset = false;
      let wereIPAdaptersReset = false;

      const { nodes, canvasV2 } = getState();
      deleted_images.forEach((image_name) => {
        const imageUsage = getImageUsage(nodes.present, canvasV2, image_name);

        if (imageUsage.isLayerImage && !wereLayersReset) {
          dispatch(layerAllDeleted());
          wereLayersReset = true;
        }

        if (imageUsage.isNodesImage && !wasNodeEditorReset) {
          dispatch(nodeEditorReset());
          wasNodeEditorReset = true;
        }

        if (imageUsage.isControlAdapterImage && !wereControlAdaptersReset) {
          dispatch(caAllDeleted());
          wereControlAdaptersReset = true;
        }

        if (imageUsage.isIPAdapterImage && !wereIPAdaptersReset) {
          dispatch(ipaAllDeleted());
          wereIPAdaptersReset = true;
        }
      });
    },
  });
};
