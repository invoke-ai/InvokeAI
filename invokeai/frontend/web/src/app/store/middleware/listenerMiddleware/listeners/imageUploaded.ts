import { startAppListening } from '..';
import { uploadAdded } from 'features/gallery/store/uploadsSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imageUploaded } from 'services/thunks/image';
import { addToast } from 'features/system/store/systemSlice';
import { initialImageSelected } from 'features/parameters/store/actions';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { resultAdded } from 'features/gallery/store/resultsSlice';
import { isResultsImageDTO, isUploadsImageDTO } from 'services/types/guards';

export const addImageUploadedListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
      imageUploaded.fulfilled.match(action) &&
      action.payload.response.image_type !== 'intermediates',
    effect: (action, { dispatch, getState }) => {
      const { response: image } = action.payload;

      const state = getState();

      if (isUploadsImageDTO(image)) {
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

      if (isResultsImageDTO(image)) {
        dispatch(resultAdded(image));
      }
    },
  });
};
