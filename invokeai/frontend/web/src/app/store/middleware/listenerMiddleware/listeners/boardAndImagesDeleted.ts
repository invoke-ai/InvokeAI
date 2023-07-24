import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { getImageUsage } from 'features/imageDeletion/store/imageDeletionSelectors';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { startAppListening } from '..';
import { boardsApi } from '../../../../../services/api/endpoints/boards';

export const addDeleteBoardAndImagesFulfilledListener = () => {
  startAppListening({
    matcher: boardsApi.endpoints.deleteBoardAndImages.matchFulfilled,
    effect: async (action, { dispatch, getState }) => {
      const { deleted_images } = action.payload;

      // Remove all deleted images from the UI

      let wasInitialImageReset = false;
      let wasCanvasReset = false;
      let wasNodeEditorReset = false;
      let wasControlNetReset = false;

      const state = getState();
      deleted_images.forEach((image_name) => {
        const imageUsage = getImageUsage(state, image_name);

        if (imageUsage.isInitialImage && !wasInitialImageReset) {
          dispatch(clearInitialImage());
          wasInitialImageReset = true;
        }

        if (imageUsage.isCanvasImage && !wasCanvasReset) {
          dispatch(resetCanvas());
          wasCanvasReset = true;
        }

        if (imageUsage.isNodesImage && !wasNodeEditorReset) {
          dispatch(nodeEditorReset());
          wasNodeEditorReset = true;
        }

        if (imageUsage.isControlNetImage && !wasControlNetReset) {
          dispatch(controlNetReset());
          wasControlNetReset = true;
        }
      });
    },
  });
};
