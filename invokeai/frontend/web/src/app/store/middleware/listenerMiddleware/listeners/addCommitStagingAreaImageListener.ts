import { logger } from 'app/logging/logger';
import { commitStagingAreaImage } from 'features/canvas/store/canvasSlice';
import { sessionCanceled } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const addCommitStagingAreaImageListener = () => {
  startAppListening({
    actionCreator: commitStagingAreaImage,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();
      const { sessionId: session_id, isProcessing } = state.system;
      const canvasSessionId = action.payload;

      if (!isProcessing) {
        // Only need to cancel if we are processing
        return;
      }

      if (!canvasSessionId) {
        log.debug('No canvas session, skipping cancel');
        return;
      }

      if (canvasSessionId !== session_id) {
        log.debug(
          {
            canvasSessionId,
            session_id,
          },
          'Canvas session does not match global session, skipping cancel'
        );
        return;
      }

      dispatch(sessionCanceled({ session_id }));
    },
  });
};
