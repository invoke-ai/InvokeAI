import { startAppListening } from '..';
import { sessionInvoked } from 'services/thunks/session';
import { log } from 'app/logging/useLogger';
import { sessionReadyToInvoke } from 'features/system/store/actions';

const moduleLog = log.child({ namespace: 'session' });

export const addSessionReadyToInvokeListener = () => {
  startAppListening({
    actionCreator: sessionReadyToInvoke,
    effect: (action, { getState, dispatch }) => {
      const { sessionId } = getState().system;
      if (sessionId) {
        moduleLog.debug(
          { sessionId },
          `Session ready to invoke (${sessionId})})`
        );
        dispatch(sessionInvoked({ sessionId }));
      }
    },
  });
};
