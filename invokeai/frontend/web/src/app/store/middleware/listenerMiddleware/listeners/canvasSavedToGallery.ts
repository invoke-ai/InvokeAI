import { log } from 'app/logging/useLogger';
import { canvasSavedToGallery } from 'features/canvas/store/actions';
import { getBaseLayerBlob } from 'features/canvas/util/getBaseLayerBlob';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { imageUploaded } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'canvasSavedToGalleryListener' });

export const addCanvasSavedToGalleryListener = () => {
  startAppListening({
    actionCreator: canvasSavedToGallery,
    effect: async (action, { dispatch, getState, take }) => {
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

      const imageUploadedRequest = dispatch(
        imageUploaded({
          file: new File([blob], 'savedCanvas.png', {
            type: 'image/png',
          }),
          image_category: 'general',
          is_intermediate: false,
          postUploadAction: {
            type: 'TOAST_CANVAS_SAVED_TO_GALLERY',
          },
        })
      );

      const [{ payload: uploadedImageDTO }] = await take(
        (
          uploadedImageAction
        ): uploadedImageAction is ReturnType<typeof imageUploaded.fulfilled> =>
          imageUploaded.fulfilled.match(uploadedImageAction) &&
          uploadedImageAction.meta.requestId === imageUploadedRequest.requestId
      );

      imagesApi.util.upsertQueryData(
        'getImageDTO',
        uploadedImageDTO.image_name,
        uploadedImageDTO
      );
    },
  });
};
