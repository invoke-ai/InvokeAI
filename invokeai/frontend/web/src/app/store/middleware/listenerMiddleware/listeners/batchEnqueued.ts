import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/store';
import { truncate } from 'es-toolkit/compat';
import { addGalleryProgressItems } from 'features/gallery/store/galleryProgressStore';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { zPydanticValidationError } from 'features/system/store/zodSchemas';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { serializeError } from 'serialize-error';
import { queueApi } from 'services/api/endpoints/queue';
import type { JsonObject } from 'type-fest';

const log = logger('queue');

export const addBatchEnqueuedListener = (startAppListening: AppStartListening) => {
  // success
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchFulfilled,
    effect: (action, listenerApi) => {
      const enqueueResult = action.payload;
      const arg = action.meta.arg.originalArgs;
      log.debug({ enqueueResult } as JsonObject, 'Batch enqueued');

      toast({
        id: 'QUEUE_BATCH_SUCCEEDED',
        title: t('queue.batchQueued'),
        status: 'success',
        description: t('queue.batchQueuedDesc', {
          count: enqueueResult.enqueued,
          direction: arg.prepend ? t('queue.front') : t('queue.back'),
        }),
      });

      // Track generate-destination batches as progress items in the gallery
      const destination = enqueueResult.batch.destination;
      if (destination === 'generate') {
        const state = listenerApi.getState();
        const targetBoardId = selectAutoAddBoardId(state);
        addGalleryProgressItems(enqueueResult.item_ids, enqueueResult.batch.batch_id ?? '', targetBoardId);
      }
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
        log.error({ batchConfig } as JsonObject, t('queue.batchFailedToQueue'));
        return;
      }

      const result = zPydanticValidationError.safeParse(response);
      if (result.success) {
        result.data.data.detail.map((e) => {
          const description = truncate(e.msg.replace(/^(Value|Index|Key) error, /i, ''), { length: 256 });
          toast({
            id: 'QUEUE_BATCH_FAILED',
            title: t('queue.batchFailedToQueue'),
            status: 'error',
            description,
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
      log.error({ batchConfig, error: serializeError(response) } as JsonObject, t('queue.batchFailedToQueue'));
    },
  });
};
