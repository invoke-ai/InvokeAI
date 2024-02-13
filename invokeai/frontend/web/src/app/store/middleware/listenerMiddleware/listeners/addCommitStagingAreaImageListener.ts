import { isAnyOf } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { canvasBatchIdsReset, commitStagingAreaImage, discardStagedImages } from 'features/canvas/store/canvasSlice';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';

import { startAppListening } from '..';

const matcher = isAnyOf(commitStagingAreaImage, discardStagedImages);

export const addCommitStagingAreaImageListener = () => {
  startAppListening({
    matcher,
    effect: async (_, { dispatch, getState }) => {
      const log = logger('canvas');
      const state = getState();
      const { batchIds } = state.canvas;

      try {
        const req = dispatch(
          queueApi.endpoints.cancelByBatchIds.initiate({ batch_ids: batchIds }, { fixedCacheKey: 'cancelByBatchIds' })
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
        dispatch(canvasBatchIdsReset());
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
