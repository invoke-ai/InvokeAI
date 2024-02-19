import { logger } from 'app/logging/logger';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { socketBulkDownloadCompleted } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addBulkDownloadCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketBulkDownloadCompleted,
    effect: async (action, { dispatch }) => {
      log.debug(action.payload, 'Bulk download complete');

      const bulk_download_item_name = action.payload.data.bulk_download_item_name;

      const url = `/api/v1/images/download/${bulk_download_item_name}`;
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = bulk_download_item_name;
      document.body.appendChild(a);
      a.click();

      dispatch(
        addToast({
          title: t('gallery.bulkDownloadStarting'),
          status: 'success',
          ...(action.payload
            ? {
                description: bulk_download_item_name,
                duration: null,
                isClosable: true,
              }
            : {}),
        })
      );
    },
  });
};
