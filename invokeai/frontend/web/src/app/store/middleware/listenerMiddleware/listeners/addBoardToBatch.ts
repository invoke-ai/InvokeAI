import { createAction } from '@reduxjs/toolkit';
import { log } from 'app/logging/useLogger';
import { imagesAddedToBatch } from 'features/batch/store/batchSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { boardImageNamesReceived } from 'services/api/thunks/boardImages';
import { receivedListOfImages } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'batch' });

export const boardAddedToBatch = createAction<{ board_id: string }>(
  'batch/boardAddedToBatch'
);

export const addAddBoardToBatchListener = () => {
  startAppListening({
    actionCreator: boardAddedToBatch,
    effect: async (action, { dispatch, getState, take }) => {
      const { board_id } = action.payload;

      const { requestId } = dispatch(boardImageNamesReceived({ board_id }));

      const [{ payload }] = await take(
        (
          action
        ): action is ReturnType<typeof boardImageNamesReceived.fulfilled> =>
          action.meta.requestId === requestId
      );

      moduleLog.debug({ data: { payload } }, 'boardImageNamesReceived');

      const { requestId: requestId2 } = dispatch(
        receivedListOfImages(payload.image_names)
      );

      const [{ payload: payload2 }] = await take(
        (action): action is ReturnType<typeof receivedListOfImages.fulfilled> =>
          action.meta.requestId === requestId2
      );

      moduleLog.debug({ data: { payload2 } }, 'receivedListOfImages');

      dispatch(imagesAddedToBatch(payload2.image_dtos));

      payload2.image_dtos.forEach((image) => {
        dispatch(
          imagesApi.util.upsertQueryData('getImageDTO', image.image_name, image)
        );
      });
    },
  });
};
