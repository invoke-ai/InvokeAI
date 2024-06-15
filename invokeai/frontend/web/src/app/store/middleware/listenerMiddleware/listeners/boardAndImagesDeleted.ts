import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import { allLayersDeleted } from 'features/controlLayers/store/canvasV2Slice';
import { getImageUsage } from 'features/deleteImageModal/store/selectors';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addDeleteBoardAndImagesFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteBoardAndImages.matchFulfilled,
    effect: async (action, { dispatch, getState }) => {
      const { deleted_images } = action.payload;

      // Remove all deleted images from the UI

      let wasCanvasReset = false;
      let wasNodeEditorReset = false;
      let wereControlAdaptersReset = false;
      let wereControlLayersReset = false;

      const { canvas, nodes, controlAdapters, canvasV2 } = getState();
      deleted_images.forEach((image_name) => {
        const imageUsage = getImageUsage(canvas, nodes.present, controlAdapters, canvasV2, image_name);

        if (imageUsage.isCanvasImage && !wasCanvasReset) {
          dispatch(resetCanvas());
          wasCanvasReset = true;
        }

        if (imageUsage.isNodesImage && !wasNodeEditorReset) {
          dispatch(nodeEditorReset());
          wasNodeEditorReset = true;
        }

        if (imageUsage.isControlImage && !wereControlAdaptersReset) {
          dispatch(controlAdaptersReset());
          wereControlAdaptersReset = true;
        }

        if (imageUsage.isControlLayerImage && !wereControlLayersReset) {
          dispatch(allLayersDeleted());
          wereControlLayersReset = true;
        }
      });
    },
  });
};
