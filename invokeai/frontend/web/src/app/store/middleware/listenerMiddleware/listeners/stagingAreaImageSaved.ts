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

      try {
        const newImageDTO = await dispatch(
          imagesApi.endpoints.changeImageIsIntermediate.initiate({
            imageDTO,
            is_intermediate: false,
          })
        ).unwrap();

        // we may need to add it to the autoadd board
        const { autoAddBoardId } = getState().gallery;

        if (autoAddBoardId) {
          await dispatch(
            imagesApi.endpoints.addImageToBoard.initiate({
              imageDTO: newImageDTO,
              board_id: autoAddBoardId,
            })
          );
        }
        dispatch(addToast({ title: 'Image Saved', status: 'success' }));
      } catch (error) {
        dispatch(
          addToast({
            title: 'Image Saving Failed',
            description: (error as Error)?.message,
            status: 'error',
          })
        );
      }
    },
  });
};
