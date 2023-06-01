import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { commitStagingAreaImage } from 'features/canvas/store/canvasSlice';
import { sessionCanceled } from 'services/thunks/session';

const moduleLog = log.child({ namespace: 'canvas' });

export const addCommitStagingAreaImageListener = () => {
  startAppListening({
    actionCreator: commitStagingAreaImage,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();
      const { sessionId, isProcessing } = state.system;
      const canvasSessionId = action.payload;

      if (!isProcessing) {
        // Only need to cancel if we are processing
        return;
      }

      if (!canvasSessionId) {
        moduleLog.debug('No canvas session, skipping cancel');
        return;
      }

      if (canvasSessionId !== sessionId) {
        moduleLog.debug(
          {
            data: {
              canvasSessionId,
              sessionId,
            },
          },
          'Canvas session does not match global session, skipping cancel'
        );
        return;
      }

      dispatch(sessionCanceled({ sessionId }));
    },
  });
};
