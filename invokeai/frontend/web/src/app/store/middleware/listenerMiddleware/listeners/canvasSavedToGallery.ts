import { logger } from 'app/logging/logger';
import { canvasSavedToGallery } from 'features/canvas/store/actions';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

export const addCanvasSavedToGalleryListener = () => {
  startAppListening({
    actionCreator: canvasSavedToGallery,
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

      dispatch(
        imagesApi.endpoints.uploadImage.initiate({
          file: new File([blob], 'savedCanvas.png', {
            type: 'image/png',
          }),
          image_category: 'general',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          crop_visible: true,
          postUploadAction: {
            type: 'TOAST',
            toastOptions: { title: 'Canvas Saved to Gallery' },
          },
        })
      );
    },
  });
};
