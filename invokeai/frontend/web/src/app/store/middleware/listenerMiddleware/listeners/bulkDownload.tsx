import type { UseToastOptions } from '@invoke-ai/ui-library';
import { ExternalLink } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { toast } from 'common/util/toast';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';
import {
  socketBulkDownloadCompleted,
  socketBulkDownloadFailed,
  socketBulkDownloadStarted,
} from 'services/events/actions';

const log = logger('images');

export const addBulkDownloadListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.bulkDownloadImages.matchFulfilled,
    effect: async (action) => {
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
        isClosable: true,
      });
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.bulkDownloadImages.matchRejected,
    effect: async () => {
      log.debug('Bulk download request failed');

      // There isn't any toast to update if we get this event.
      toast({
        title: t('gallery.bulkDownloadRequestFailed'),
        status: 'success',
        isClosable: true,
      });
    },
  });

  startAppListening({
    actionCreator: socketBulkDownloadStarted,
    effect: async (action) => {
      // This should always happen immediately after the bulk download request, so we don't need to show a toast here.
      log.debug(action.payload.data, 'Bulk download preparation started');
    },
  });

  startAppListening({
    actionCreator: socketBulkDownloadCompleted,
    effect: async (action) => {
      log.debug(action.payload.data, 'Bulk download preparation completed');

      const { bulk_download_item_name } = action.payload.data;

      // TODO(psyche): This URL may break in in some environments (e.g. Nvidia workbench) but we need to test it first
      const url = `/api/v1/images/download/${bulk_download_item_name}`;

      const toastOptions: UseToastOptions = {
        id: bulk_download_item_name,
        title: t('gallery.bulkDownloadReady', 'Download ready'),
        status: 'success',
        description: (
          <ExternalLink
            label={t('gallery.clickToDownload', 'Click here to download')}
            href={url}
            download={bulk_download_item_name}
          />
        ),
        duration: null,
        isClosable: true,
      };

      if (toast.isActive(bulk_download_item_name)) {
        toast.update(bulk_download_item_name, toastOptions);
      } else {
        toast(toastOptions);
      }
    },
  });

  startAppListening({
    actionCreator: socketBulkDownloadFailed,
    effect: async (action) => {
      log.debug(action.payload.data, 'Bulk download preparation failed');

      const { bulk_download_item_name } = action.payload.data;

      const toastOptions: UseToastOptions = {
        id: bulk_download_item_name,
        title: t('gallery.bulkDownloadFailed'),
        status: 'error',
        description: action.payload.data.error,
        duration: null,
        isClosable: true,
      };

      if (toast.isActive(bulk_download_item_name)) {
        toast.update(bulk_download_item_name, toastOptions);
      } else {
        toast(toastOptions);
      }
    },
  });
};
