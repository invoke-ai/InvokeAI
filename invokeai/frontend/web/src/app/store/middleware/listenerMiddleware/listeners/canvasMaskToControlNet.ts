import { logger } from 'app/logging/logger';
import { canvasMaskToControlNet } from 'features/canvas/store/actions';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

export const addCanvasMaskToControlNetListener = () => {
  startAppListening({
    actionCreator: canvasMaskToControlNet,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();

      const canvasBlobsAndImageData = await getCanvasData(
        state.canvas.layerState,
        state.canvas.boundingBoxCoordinates,
        state.canvas.boundingBoxDimensions,
        state.canvas.isMaskEnabled,
        state.canvas.shouldPreserveMaskedArea
      );

      if (!canvasBlobsAndImageData) {
        return;
      }

      const { maskBlob } = canvasBlobsAndImageData;

      if (!maskBlob) {
        log.error('Problem getting mask layer blob');
        dispatch(
          addToast({
            title: 'Problem Importing Mask',
            description: 'Unable to export mask',
            status: 'error',
          })
        );
        return;
      }

      const { autoAddBoardId } = state.gallery;

      const imageDTO = await dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([maskBlob], 'canvasMaskImage.png', {
            type: 'image/png',
          }),
          image_category: 'mask',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          crop_visible: true,
          postUploadAction: {
            type: 'TOAST',
            toastOptions: { title: 'Mask Sent to ControlNet & Assets' },
          },
        })
      ).unwrap();

      const { image_name } = imageDTO;

      dispatch(
        controlNetImageChanged({
          controlNetId: action.payload.controlNet.controlNetId,
          controlImage: image_name,
        })
      );
    },
  });
};
