import { createStandaloneToast, theme, TOAST_OPTIONS } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { zPydanticValidationError } from 'features/system/store/zodSchemas';
import { t } from 'i18next';
import { truncate, upperFirst } from 'lodash-es';
import { queueApi } from 'services/api/endpoints/queue';

import { startAppListening } from '..';

const { toast } = createStandaloneToast({
  theme: theme,
  defaultOptions: TOAST_OPTIONS.defaultOptions,
});

export const addBatchEnqueuedListener = () => {
  // success
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchFulfilled,
    effect: async (action) => {
      const response = action.payload;
      const arg = action.meta.arg.originalArgs;
      logger('queue').debug({ enqueueResult: parseify(response) }, 'Batch enqueued');

      if (!toast.isActive('batch-queued')) {
        toast({
          id: 'batch-queued',
          title: t('queue.batchQueued'),
          description: t('queue.batchQueuedDesc', {
            count: response.enqueued,
            direction: arg.prepend ? t('queue.front') : t('queue.back'),
          }),
          duration: 1000,
          status: 'success',
        });
      }
    },
  });

  // error
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchRejected,
    effect: async (action) => {
      const response = action.payload;
      const arg = action.meta.arg.originalArgs;

      if (!response) {
        toast({
          title: t('queue.batchFailedToQueue'),
          status: 'error',
          description: 'Unknown Error',
        });
        logger('queue').error({ batchConfig: parseify(arg), error: parseify(response) }, t('queue.batchFailedToQueue'));
        return;
      }

      const result = zPydanticValidationError.safeParse(response);
      if (result.success) {
        result.data.data.detail.map((e) => {
          toast({
            id: 'batch-failed-to-queue',
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
          title: t('queue.batchFailedToQueue'),
          description: t('common.unknownError'),
          status: 'error',
        });
      }
      logger('queue').error({ batchConfig: parseify(arg), error: parseify(response) }, t('queue.batchFailedToQueue'));
    },
  });
};
