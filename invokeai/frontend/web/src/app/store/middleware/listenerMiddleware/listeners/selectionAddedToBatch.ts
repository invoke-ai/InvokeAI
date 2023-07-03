import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import {
  imagesAddedToBatch,
  selectionAddedToBatch,
} from 'features/batch/store/batchSlice';

const moduleLog = log.child({ namespace: 'batch' });

export const addSelectionAddedToBatchListener = () => {
  startAppListening({
    actionCreator: selectionAddedToBatch,
    effect: (action, { dispatch, getState }) => {
      const { selection } = getState().gallery;

      dispatch(imagesAddedToBatch(selection));
    },
  });
};
