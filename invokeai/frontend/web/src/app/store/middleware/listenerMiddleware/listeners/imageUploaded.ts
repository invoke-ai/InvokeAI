import { startAppListening } from '..';
import { imageUploaded } from 'services/thunks/image';
import { addToast } from 'features/system/store/systemSlice';
import { log } from 'app/logging/useLogger';
import { imageUpserted } from 'features/gallery/store/imagesSlice';
import { SAVED_CANVAS_FILENAME } from './canvasSavedToGallery';
import { MERGED_CANVAS_FILENAME } from './canvasMerged';

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

      const originalFileName = action.meta.arg.formData.file.name;

      dispatch(imageUpserted(image));

      if (originalFileName === SAVED_CANVAS_FILENAME) {
        dispatch(
          addToast({ title: 'Canvas Saved to Gallery', status: 'success' })
        );
        return;
      }

      if (originalFileName === MERGED_CANVAS_FILENAME) {
        dispatch(addToast({ title: 'Canvas Merged', status: 'success' }));
        return;
      }

      dispatch(addToast({ title: 'Image Uploaded', status: 'success' }));
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
