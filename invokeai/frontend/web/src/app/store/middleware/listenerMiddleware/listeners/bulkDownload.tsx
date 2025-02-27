import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';

const log = logger('gallery');

export const addBulkDownloadListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.bulkDownloadImages.matchFulfilled,
    effect: (action) => {
      log.debug(action.payload, 'Bulk download requested');

      // If we have an item name, we are processing the bulk download locally and should use it as the toast id to
      // prevent multiple toasts for the same item.
      toast({
        id: action.payload.bulk_download_item_name ?? undefined,
        title: t('gallery.bulkDownloadRequested'),
        status: 'success',
        // Show the response message if it exists, otherwise show the default message
        description: action.payload.response || t('gallery.bulkDownloadRequestedDesc'),
        duration: null,
      });
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.bulkDownloadImages.matchRejected,
    effect: () => {
      log.debug('Bulk download request failed');

      // There isn't any toast to update if we get this event.
      toast({
        id: 'BULK_DOWNLOAD_REQUEST_FAILED',
        title: t('gallery.bulkDownloadRequestFailed'),
        status: 'error',
      });
    },
  });
};
