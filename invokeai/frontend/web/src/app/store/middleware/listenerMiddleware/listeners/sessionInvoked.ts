import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { sessionInvoked } from 'services/api/thunks/session';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'session' });

export const addSessionInvokedPendingListener = () => {
  startAppListening({
    actionCreator: sessionInvoked.pending,
    effect: (action, { getState, dispatch }) => {
      //
    },
  });
};

export const addSessionInvokedFulfilledListener = () => {
  startAppListening({
    actionCreator: sessionInvoked.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const { session_id } = action.meta.arg;
      moduleLog.debug(
        { data: { session_id } },
        `Session invoked (${session_id})`
      );
    },
  });
};

export const addSessionInvokedRejectedListener = () => {
  startAppListening({
    actionCreator: sessionInvoked.rejected,
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
          `Problem invoking session`
        );
      }
    },
  });
};
