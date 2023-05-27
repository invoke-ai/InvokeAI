import { startAppListening } from '..';
import { uploadUpserted } from 'features/gallery/store/uploadsSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imageUploaded } from 'services/thunks/image';
import { addToast } from 'features/system/store/systemSlice';
import { resultUpserted } from 'features/gallery/store/resultsSlice';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUploadedFulfilledListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
      imageUploaded.fulfilled.match(action) &&
      action.payload.is_intermediate === false,
    effect: (action, { dispatch, getState }) => {
      const image = action.payload;

      moduleLog.debug({ arg: '<Blob>', image }, 'Image uploaded');

      const state = getState();

      // Handle uploads
      if (!image.show_in_gallery && image.image_type === 'uploads') {
        dispatch(uploadUpserted(image));

        dispatch(addToast({ title: 'Image Uploaded', status: 'success' }));

        if (state.gallery.shouldAutoSwitchToNewImages) {
          dispatch(imageSelected(image));
        }
      }

      // Handle results
      // TODO: Can this ever happen? I don't think so...
      if (image.show_in_gallery) {
        dispatch(resultUpserted(image));
      }
    },
  });
};

export const addImageUploadedRejectedListener = () => {
  startAppListening({
    actionCreator: imageUploaded.rejected,
    effect: (action, { dispatch }) => {
      dispatch(
        addToast({
          title: 'Image Upload Failed',
          description: action.error.message,
          status: 'error',
        })
      );
    },
  });
};
