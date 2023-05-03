import { deserializeImageResponse } from 'services/util/deserializeImageResponse';
import { AppListenerEffect } from '..';
import { uploadAdded } from 'features/gallery/store/uploadsSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';

export const imageUploadedListener: AppListenerEffect = (
  action,
  { dispatch, getState }
) => {
  const { response } = action.payload;
  const state = getState();
  const image = deserializeImageResponse(response);

  dispatch(uploadAdded(image));

  if (state.gallery.shouldAutoSwitchToNewImages) {
    dispatch(imageSelected(image));
  }
};
