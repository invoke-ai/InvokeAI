import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { SerializableObject } from 'common/types';
import { zPydanticValidationError } from 'features/system/store/zodSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { truncate, upperFirst } from 'lodash-es';
import { serializeError } from 'serialize-error';
import { queueApi } from 'services/api/endpoints/queue';

const log = logger('queue');

export const addBatchEnqueuedListener = (startAppListening: AppStartListening) => {
  // success
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchFulfilled,
    effect: (action) => {
      const enqueueResult = action.payload;
      const arg = action.meta.arg.originalArgs;
      log.debug({ enqueueResult } as SerializableObject, 'Batch enqueued');

      toast({
        id: 'QUEUE_BATCH_SUCCEEDED',
        title: t('queue.batchQueued'),
        status: 'success',
        description: t('queue.batchQueuedDesc', {
          count: enqueueResult.enqueued,
          direction: arg.prepend ? t('queue.front') : t('queue.back'),
        }),
      });
    },
  });

  // error
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchRejected,
    effect: (action) => {
      const response = action.payload;
      const batchConfig = action.meta.arg.originalArgs;

      if (!response) {
        toast({
          id: 'QUEUE_BATCH_FAILED',
          title: t('queue.batchFailedToQueue'),
          status: 'error',
          description: t('common.unknownError'),
        });
        log.error({ batchConfig } as SerializableObject, t('queue.batchFailedToQueue'));
        return;
      }

      const result = zPydanticValidationError.safeParse(response);
      if (result.success) {
        result.data.data.detail.map((e) => {
          toast({
            id: 'QUEUE_BATCH_FAILED',
            title: truncate(upperFirst(e.msg), { length: 128 }),
            status: 'error',
            description: truncate(
              `Path:
              ${e.loc.join('.')}`,
              { length: 128 }
            ),
          });
        });
      } else if (response.status !== 403) {
        toast({
          id: 'QUEUE_BATCH_FAILED',
          title: t('queue.batchFailedToQueue'),
          status: 'error',
          description: t('common.unknownError'),
        });
      }
      log.error({ batchConfig, error: serializeError(response) } as SerializableObject, t('queue.batchFailedToQueue'));
    },
  });
};
