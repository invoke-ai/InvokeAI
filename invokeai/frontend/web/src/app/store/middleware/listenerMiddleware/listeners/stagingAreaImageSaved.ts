import { stagingAreaImageSaved } from 'features/canvas/store/actions';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { imageUpdated } from 'services/thunks/image';
import { imageUpserted } from 'features/gallery/store/imagesSlice';
import { addToast } from 'features/system/store/systemSlice';

const moduleLog = log.child({ namespace: 'canvas' });

export const addStagingAreaImageSavedListener = () => {
  startAppListening({
    actionCreator: stagingAreaImageSaved,
    effect: async (action, { dispatch, getState, take }) => {
      const { image_name, image_origin } = action.payload;

      dispatch(
        imageUpdated({
          imageName: image_name,
          imageOrigin: image_origin,
          requestBody: {
            is_intermediate: false,
          },
        })
      );

      const [imageUpdatedAction] = await take(
        (action) =>
          (imageUpdated.fulfilled.match(action) ||
            imageUpdated.rejected.match(action)) &&
          action.meta.arg.imageName === image_name
      );

      if (imageUpdated.rejected.match(imageUpdatedAction)) {
        moduleLog.error(
          { data: { arg: imageUpdatedAction.meta.arg } },
          'Image saving failed'
        );
        dispatch(
          addToast({
            title: 'Image Saving Failed',
            description: imageUpdatedAction.error.message,
            status: 'error',
          })
        );
        return;
      }

      if (imageUpdated.fulfilled.match(imageUpdatedAction)) {
        dispatch(imageUpserted(imageUpdatedAction.payload));
        dispatch(addToast({ title: 'Image Saved', status: 'success' }));
      }
    },
  });
};
