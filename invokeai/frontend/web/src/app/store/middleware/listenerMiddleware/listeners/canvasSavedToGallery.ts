import { canvasSavedToGallery } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { imageUploaded } from 'services/thunks/image';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';

const moduleLog = log.child({ namespace: 'canvasSavedToGalleryListener' });

export const addCanvasSavedToGalleryListener = () => {
  startAppListening({
    actionCreator: canvasSavedToGallery,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();

      const blob = await getBaseLayerBlob(state);

      if (!blob) {
        moduleLog.error('Problem getting base layer blob');
        dispatch(
          addToast({
            title: 'Problem Saving Canvas',
            description: 'Unable to export base layer',
            status: 'error',
          })
        );
        return;
      }

      dispatch(
        imageUploaded({
          imageType: 'results',
          formData: {
            file: new File([blob], 'mergedCanvas.png', { type: 'image/png' }),
          },
        })
      );
    },
  });
};
