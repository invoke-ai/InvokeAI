import { logger } from 'app/logging/logger';
import { matchAnyStagingAreaDismissed } from 'features/canvas/store/canvasSlice';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';

import { startAppListening } from '..';

export const addCommitStagingAreaImageListener = () => {
  startAppListening({
    matcher: matchAnyStagingAreaDismissed,
    effect: async (_, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();
      const { canvasBatchIds } = state.progress;

      try {
        const req = dispatch(
          queueApi.endpoints.cancelByBatchIds.initiate(
            { batch_ids: canvasBatchIds },
            { fixedCacheKey: 'cancelByBatchIds' }
          )
        );
        const { canceled } = await req.unwrap();
        req.reset();
        if (canceled > 0) {
          log.debug(`Canceled ${canceled} canvas batches`);
          dispatch(
            addToast({
              title: t('queue.cancelBatchSucceeded'),
              status: 'success',
            })
          );
        }
      } catch {
        log.error('Failed to cancel canvas batches');
        dispatch(
          addToast({
            title: t('queue.cancelBatchFailed'),
            status: 'error',
          })
        );
      }
    },
  });
};
