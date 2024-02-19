import { logger } from 'app/logging/logger';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { socketBulkDownloadFailed } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addBulkDownloadFailedEventListener = () => {
  startAppListening({
    actionCreator: socketBulkDownloadFailed,
    effect: async (action, { dispatch }) => {
      log.debug(action.payload, 'Bulk download error');


      dispatch(
        addToast({
          title: t('gallery.bulkDownloadFailed'),
          status: 'error',
          ...(action.payload
            ? {
                description: action.payload.data.error,
                duration: null,
                isClosable: true,
              }
            : {}),
        })
      );
    },
  });
};
