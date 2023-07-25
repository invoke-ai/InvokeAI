import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { serializeError } from 'serialize-error';
import { sessionCreated } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const addSessionCreatedPendingListener = () => {
  startAppListening({
    actionCreator: sessionCreated.pending,
    effect: () => {
      //
    },
  });
};

export const addSessionCreatedFulfilledListener = () => {
  startAppListening({
    actionCreator: sessionCreated.fulfilled,
    effect: (action) => {
      const log = logger('session');
      const session = action.payload;
      log.debug(
        { session: parseify(session) },
        `Session created (${session.id})`
      );
    },
  });
};

export const addSessionCreatedRejectedListener = () => {
  startAppListening({
    actionCreator: sessionCreated.rejected,
    effect: (action) => {
      const log = logger('session');
      if (action.payload) {
        const { error, status } = action.payload;
        const graph = parseify(action.meta.arg);
        log.error(
          { graph, status, error: serializeError(error) },
          `Problem creating session`
        );
      }
    },
  });
};
