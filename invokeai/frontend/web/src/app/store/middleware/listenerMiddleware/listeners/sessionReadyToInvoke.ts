import { logger } from 'app/logging/logger';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { sessionInvoked } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const addSessionReadyToInvokeListener = () => {
  startAppListening({
    actionCreator: sessionReadyToInvoke,
    effect: (action, { getState, dispatch }) => {
      const log = logger('session');
      const { sessionId: session_id } = getState().system;
      if (session_id) {
        log.debug({ session_id }, `Session ready to invoke (${session_id})})`);
        dispatch(sessionInvoked({ session_id }));
      }
    },
  });
};
