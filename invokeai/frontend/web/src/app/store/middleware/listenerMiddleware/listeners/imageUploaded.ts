import { deserializeImageResponse } from 'services/util/deserializeImageResponse';
import { startAppListening } from '..';
import { uploadAdded } from 'features/gallery/store/uploadsSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imageUploaded } from 'services/thunks/image';

export const addImageUploadedListener = () => {
  startAppListening({
    actionCreator: imageUploaded.fulfilled,
    effect: (action, { dispatch, getState }) => {
      const { response } = action.payload;
      const state = getState();
      const image = deserializeImageResponse(response);

      dispatch(uploadAdded(image));

      if (state.gallery.shouldAutoSwitchToNewImages) {
        dispatch(imageSelected(image));
      }
    },
  });
};
