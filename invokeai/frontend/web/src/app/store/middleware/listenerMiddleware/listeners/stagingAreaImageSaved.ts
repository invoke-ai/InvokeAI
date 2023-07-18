import { log } from 'app/logging/useLogger';
import { stagingAreaImageSaved } from 'features/canvas/store/actions';
import { addToast } from 'features/system/store/systemSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'canvas' });

export const addStagingAreaImageSavedListener = () => {
  startAppListening({
    actionCreator: stagingAreaImageSaved,
    effect: async (action, { dispatch, getState, take }) => {
      const { imageDTO } = action.payload;

      dispatch(
        imagesApi.endpoints.updateImage.initiate({
          imageDTO,
          changes: { is_intermediate: false },
        })
      )
        .unwrap()
        .then((image) => {
          dispatch(addToast({ title: 'Image Saved', status: 'success' }));
        })
        .catch((error) => {
          dispatch(
            addToast({
              title: 'Image Saving Failed',
              description: error.message,
              status: 'error',
            })
          );
        });
    },
  });
};
