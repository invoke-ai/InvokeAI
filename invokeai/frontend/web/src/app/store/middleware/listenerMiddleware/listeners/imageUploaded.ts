import { startAppListening } from '..';
import { uploadUpserted } from 'features/gallery/store/uploadsSlice';
import {
  imageSelected,
  setCurrentCategory,
} from 'features/gallery/store/gallerySlice';
import { imageUploaded } from 'services/thunks/image';
import { addToast } from 'features/system/store/systemSlice';
import { resultUpserted } from 'features/gallery/store/resultsSlice';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUploadedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageUploaded.fulfilled,
    effect: (action, { dispatch, getState }) => {
      const image = action.payload;

      moduleLog.debug({ arg: '<Blob>', image }, 'Image uploaded');

      if (action.payload.is_intermediate) {
        // No further actions needed for intermediate images
        return;
      }

      const state = getState();

      // Handle uploads
      if (image.image_category === 'user' && !image.is_intermediate) {
        dispatch(uploadUpserted(image));
        dispatch(addToast({ title: 'Image Uploaded', status: 'success' }));
      }

      // Handle results
      // TODO: Can this ever happen? I don't think so...
      if (image.image_category !== 'user' && !image.is_intermediate) {
        dispatch(resultUpserted(image));
        dispatch(setCurrentCategory('results'));
      }
    },
  });
};

export const addImageUploadedRejectedListener = () => {
  startAppListening({
    actionCreator: imageUploaded.rejected,
    effect: (action, { dispatch }) => {
      const { formData, ...rest } = action.meta.arg;
      const sanitizedData = { arg: { ...rest, formData: { file: '<Blob>' } } };
      moduleLog.error({ data: sanitizedData }, 'Image upload failed');
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
