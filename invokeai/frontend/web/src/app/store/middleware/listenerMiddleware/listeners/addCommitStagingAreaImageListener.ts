import { isAnyOf } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { canvasReset, newSessionRequested } from 'features/controlLayers/store/actions';
import { stagingAreaReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';

const log = logger('canvas');

const matchCanvasOrStagingAreaReset = isAnyOf(stagingAreaReset, canvasReset, newSessionRequested);

export const addStagingListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: matchCanvasOrStagingAreaReset,
    effect: async (_, { dispatch }) => {
      try {
        const req = dispatch(
          queueApi.endpoints.cancelByBatchDestination.initiate(
            { destination: 'canvas' },
            { fixedCacheKey: 'cancelByBatchOrigin' }
          )
        );
        const { canceled } = await req.unwrap();
        req.reset();

        if (canceled > 0) {
          log.debug(`Canceled ${canceled} canvas batches`);
          toast({
            id: 'CANCEL_BATCH_SUCCEEDED',
            title: t('queue.cancelBatchSucceeded'),
            status: 'success',
          });
        }
      } catch {
        log.error('Failed to cancel canvas batches');
        toast({
          id: 'CANCEL_BATCH_FAILED',
          title: t('queue.cancelBatchFailed'),
          status: 'error',
        });
      }
    },
  });
};
