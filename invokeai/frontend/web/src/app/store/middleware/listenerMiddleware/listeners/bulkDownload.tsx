import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/store';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';

const log = logger('gallery');

export const addBulkDownloadListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.bulkDownloadImages.matchFulfilled,
    effect: (action) => {
      log.debug(action.payload, 'Bulk download requested');

      // Use a "preparing:" prefix so this toast cannot collide with the
      // "ready to download" toast that arrives via the bulk_download_complete
      // socket event.  The background task can complete in under 20ms, so the
      // socket event may arrive *before* this Redux middleware runs — without
      // distinct IDs the "preparing" toast would overwrite the "ready" toast.
      const itemName = action.payload.bulk_download_item_name;
      toast({
        id: itemName ? `preparing:${itemName}` : undefined,
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
