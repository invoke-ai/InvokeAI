import { createAction } from '@reduxjs/toolkit';
import { log } from 'app/logging/useLogger';
import { imagesAddedToBatch } from 'features/batch/store/batchSlice';
import { boardImageNamesReceived } from 'services/api/thunks/boardImages';
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

      dispatch(imagesAddedToBatch(payload.image_names));
    },
  });
};
