import { canvasSavedToGallery } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { imageUploaded } from 'services/thunks/image';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { v4 as uuidv4 } from 'uuid';
import { imageUpserted } from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'canvasSavedToGalleryListener' });

export const addCanvasSavedToGalleryListener = () => {
  startAppListening({
    actionCreator: canvasSavedToGallery,
    effect: async (action, { dispatch, getState, take }) => {
      const state = getState();

      const blob = await getBaseLayerBlob(state, true);

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

      const filename = `mergedCanvas_${uuidv4()}.png`;

      dispatch(
        imageUploaded({
          formData: {
            file: new File([blob], filename, { type: 'image/png' }),
          },
          imageCategory: 'general',
          isIntermediate: false,
        })
      );

      const [{ payload: uploadedImageDTO }] = await take(
        (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
          imageUploaded.fulfilled.match(action) &&
          action.meta.arg.formData.file.name === filename
      );

      dispatch(imageUpserted(uploadedImageDTO));
    },
  });
};
