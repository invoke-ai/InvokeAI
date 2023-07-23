import { logger } from 'app/logging/logger';
import { serializeError } from 'serialize-error';
import { sessionCanceled } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const addSessionCanceledPendingListener = () => {
  startAppListening({
    actionCreator: sessionCanceled.pending,
    effect: () => {
      //
    },
  });
};

export const addSessionCanceledFulfilledListener = () => {
  startAppListening({
    actionCreator: sessionCanceled.fulfilled,
    effect: (action) => {
      const log = logger('session');
      const { session_id } = action.meta.arg;
      log.debug({ session_id }, `Session canceled (${session_id})`);
    },
  });
};

export const addSessionCanceledRejectedListener = () => {
  startAppListening({
    actionCreator: sessionCanceled.rejected,
    effect: (action) => {
      const log = logger('session');
      const { session_id } = action.meta.arg;
      if (action.payload) {
        const { error } = action.payload;
        log.error(
          {
            session_id,
            error: serializeError(error),
          },
          `Problem canceling session`
        );
      }
    },
  });
};
