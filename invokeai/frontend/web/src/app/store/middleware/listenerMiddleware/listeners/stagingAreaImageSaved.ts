import { log } from 'app/logging/useLogger';
import { stagingAreaImageSaved } from 'features/canvas/store/actions';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { imageUpdated } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'canvas' });

export const addStagingAreaImageSavedListener = () => {
  startAppListening({
    actionCreator: stagingAreaImageSaved,
    effect: async (action, { dispatch, getState, take }) => {
      const { imageName } = action.payload;

      dispatch(
        imageUpdated({
          image_name: imageName,
          is_intermediate: false,
        })
      );

      const [imageUpdatedAction] = await take(
        (action) =>
          (imageUpdated.fulfilled.match(action) ||
            imageUpdated.rejected.match(action)) &&
          action.meta.arg.image_name === imageName
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
        // update cache
        imagesApi.util.updateQueryData('getImageDTO', imageName, (draft) => {
          Object.assign(draft, { is_intermediate: false });
        });
        dispatch(addToast({ title: 'Image Saved', status: 'success' }));
      }
    },
  });
};
