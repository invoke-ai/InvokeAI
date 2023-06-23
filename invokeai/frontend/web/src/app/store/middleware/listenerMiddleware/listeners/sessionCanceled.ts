import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { sessionCanceled } from 'services/api/thunks/session';
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
      const { session_id } = action.meta.arg;
      moduleLog.debug(
        { data: { session_id } },
        `Session canceled (${session_id})`
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
