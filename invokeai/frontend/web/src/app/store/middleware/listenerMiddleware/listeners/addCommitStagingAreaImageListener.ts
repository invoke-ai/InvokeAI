import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { commitStagingAreaImage } from 'features/canvas/store/canvasSlice';
import { sessionCanceled } from 'services/api/thunks/session';

const moduleLog = log.child({ namespace: 'canvas' });

export const addCommitStagingAreaImageListener = () => {
  startAppListening({
    actionCreator: commitStagingAreaImage,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();
      const { sessionId: session_id, isProcessing } = state.system;
      const canvasSessionId = action.payload;

      if (!isProcessing) {
        // Only need to cancel if we are processing
        return;
      }

      if (!canvasSessionId) {
        moduleLog.debug('No canvas session, skipping cancel');
        return;
      }

      if (canvasSessionId !== session_id) {
        moduleLog.debug(
          {
            data: {
              canvasSessionId,
              session_id,
            },
          },
          'Canvas session does not match global session, skipping cancel'
        );
        return;
      }

      dispatch(sessionCanceled({ session_id }));
    },
  });
};
