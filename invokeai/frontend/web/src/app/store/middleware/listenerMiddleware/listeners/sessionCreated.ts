import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { sessionCreated } from 'services/thunks/session';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'session' });

export const addSessionCreatedPendingListener = () => {
  startAppListening({
    actionCreator: sessionCreated.pending,
    effect: (action, { getState, dispatch }) => {
      //
    },
  });
};

export const addSessionCreatedFulfilledListener = () => {
  startAppListening({
    actionCreator: sessionCreated.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const session = action.payload;
      moduleLog.debug({ data: { session } }, `Session created (${session.id})`);
    },
  });
};

export const addSessionCreatedRejectedListener = () => {
  startAppListening({
    actionCreator: sessionCreated.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        const { arg, error } = action.payload;
        moduleLog.error(
          {
            data: {
              arg,
              error: serializeError(error),
            },
          },
          `Problem creating session`
        );
      }
    },
  });
};
