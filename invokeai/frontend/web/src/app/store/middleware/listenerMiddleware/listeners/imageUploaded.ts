import { deserializeImageResponse } from 'services/util/deserializeImageResponse';
import { startAppListening } from '..';
import { uploadAdded } from 'features/gallery/store/uploadsSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imageUploaded } from 'services/thunks/image';
import { addToast } from 'features/system/store/systemSlice';
import { initialImageSelected } from 'features/parameters/store/actions';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { resultAdded } from 'features/gallery/store/resultsSlice';

export const addImageUploadedListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
      imageUploaded.fulfilled.match(action) &&
      action.payload.response.image_type !== 'intermediates',
    effect: (action, { dispatch, getState }) => {
      const { response } = action.payload;
      const { imageType } = action.meta.arg;

      const state = getState();
      const image = deserializeImageResponse(response);

      if (imageType === 'uploads') {
        dispatch(uploadAdded(image));

        dispatch(addToast({ title: 'Image Uploaded', status: 'success' }));

        if (state.gallery.shouldAutoSwitchToNewImages) {
          dispatch(imageSelected(image));
        }

        if (action.meta.arg.activeTabName === 'img2img') {
          dispatch(initialImageSelected(image));
        }

        if (action.meta.arg.activeTabName === 'unifiedCanvas') {
          dispatch(setInitialCanvasImage(image));
        }
      }

      if (imageType === 'results') {
        dispatch(resultAdded(image));
      }
    },
  });
};
