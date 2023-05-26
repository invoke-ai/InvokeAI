import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { sessionCanceled } from 'services/thunks/session';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'session' });

export const addSessionCanceledPendingListener = () => {
  startAppListening({
    actionCreator: sessionCanceled.pending,
    effect: (action, { getState, dispatch }) => {
      //
    },
  });
};

export const addSessionCanceledFulfilledListener = () => {
  startAppListening({
    actionCreator: sessionCanceled.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const { sessionId } = action.meta.arg;
      moduleLog.debug(
        { data: { sessionId } },
        `Session canceled (${sessionId})`
      );
    },
  });
};

export const addSessionCanceledRejectedListener = () => {
  startAppListening({
    actionCreator: sessionCanceled.rejected,
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
          `Problem canceling session`
        );
      }
    },
  });
};
