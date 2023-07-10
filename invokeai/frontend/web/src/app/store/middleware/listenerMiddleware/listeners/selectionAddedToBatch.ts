import { log } from 'app/logging/useLogger';
import {
  imagesAddedToBatch,
  selectionAddedToBatch,
} from 'features/batch/store/batchSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { receivedListOfImages } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'batch' });

export const addSelectionAddedToBatchListener = () => {
  startAppListening({
    actionCreator: selectionAddedToBatch,
    effect: async (action, { dispatch, getState, take }) => {
      const { selection } = getState().gallery;

      const { requestId } = dispatch(receivedListOfImages(selection));

      const [{ payload }] = await take(
        (action): action is ReturnType<typeof receivedListOfImages.fulfilled> =>
          action.meta.requestId === requestId
      );

      moduleLog.debug({ data: { payload } }, 'receivedListOfImages');

      dispatch(imagesAddedToBatch(payload.image_dtos));

      payload.image_dtos.forEach((image) => {
        dispatch(
          imagesApi.util.upsertQueryData('getImageDTO', image.image_name, image)
        );
      });
    },
  });
};
