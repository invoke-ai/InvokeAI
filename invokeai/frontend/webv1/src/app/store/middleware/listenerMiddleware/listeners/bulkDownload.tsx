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
      // Read through optionals: the payload is typed as a model, but `fetchBaseQuery` resolves
      // an empty response entity as `data: null`, so a proxy that strips the body off the 202
      // fulfils this action with nothing in it.
      const itemName = action.payload?.bulk_download_item_name;
      if (!itemName) {
        // Nothing to key the toast on, so there must be no toast. This one is raised with
        // `duration: null` and is dismissed by name when the zip arrives
        // (`toastApi.close(\`preparing:${name}\`)` in setEventListeners); raising it without an
        // id gets it a random one instead, which that close call can never match — a permanent
        // "preparing your download" banner for a download that has already landed. The
        // download itself is unaffected: it was scheduled server-side, and its completion
        // toast arrives over the socket.
        return;
      }
      toast({
        id: `preparing:${itemName}`,
        title: t('gallery.bulkDownloadRequested'),
        status: 'success',
        // Show the response message if it exists, otherwise show the default message
        description: action.payload?.response || t('gallery.bulkDownloadRequestedDesc'),
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
