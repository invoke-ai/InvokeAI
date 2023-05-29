import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { receivedResultImagesPage } from 'services/thunks/gallery';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedResultImagesPageFulfilledListener = () => {
  startAppListening({
    actionCreator: receivedResultImagesPage.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const page = action.payload;
      moduleLog.debug(
        { data: { page } },
        `Received ${page.items.length} results`
      );
    },
  });
};

export const addReceivedResultImagesPageRejectedListener = () => {
  startAppListening({
    actionCreator: receivedResultImagesPage.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        moduleLog.debug(
          { data: { error: serializeError(action.payload.error) } },
          'Problem receiving results'
        );
      }
    },
  });
};
