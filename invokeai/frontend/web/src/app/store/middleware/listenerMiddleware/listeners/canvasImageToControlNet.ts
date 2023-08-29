import { logger } from 'app/logging/logger';
import { canvasImageToControlNet } from 'features/canvas/store/actions';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

export const addCanvasImageToControlNetListener = () => {
  startAppListening({
    actionCreator: canvasImageToControlNet,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();

      const blob = await getBaseLayerBlob(state);

      if (!blob) {
        log.error('Problem getting base layer blob');
        dispatch(
          addToast({
            title: 'Problem Saving Canvas',
            description: 'Unable to export base layer',
            status: 'error',
          })
        );
        return;
      }

      const { autoAddBoardId } = state.gallery;

      const imageDTO = await dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([blob], 'savedCanvas.png', {
            type: 'image/png',
          }),
          image_category: 'mask',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          crop_visible: true,
          postUploadAction: {
            type: 'TOAST',
            toastOptions: { title: 'Canvas Sent to ControlNet & Assets' },
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
