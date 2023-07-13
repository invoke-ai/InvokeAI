import { startAppListening } from '..';
import { sessionInvoked } from 'services/api/thunks/session';
import { log } from 'app/logging/useLogger';
import { sessionReadyToInvoke } from 'features/system/store/actions';

const moduleLog = log.child({ namespace: 'session' });

export const addSessionReadyToInvokeListener = () => {
  startAppListening({
    actionCreator: sessionReadyToInvoke,
    effect: (action, { getState, dispatch }) => {
      const { sessionId: session_id } = getState().system;
      if (session_id) {
        moduleLog.debug(
          { session_id },
          `Session ready to invoke (${session_id})})`
        );
        dispatch(sessionInvoked({ session_id }));
      }
    },
  });
};
