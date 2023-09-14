import { logger } from 'app/logging/logger';
import { queueApi } from 'services/api/endpoints/queue';
import {
  appSocketQueueStatusChanged,
  socketQueueStatusChanged,
} from 'services/events/actions';
import { startAppListening } from '../..';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';

export const addSocketQueueStatusChangedEventListener = () => {
  startAppListening({
    actionCreator: socketQueueStatusChanged,
    effect: (action, { dispatch, getOriginalState }) => {
      const log = logger('socketio');
      log.debug(action.payload, `Queue status updated`);
      // pass along the socket event as an application action
      dispatch(appSocketQueueStatusChanged(action.payload));

      const { data: oldQueueStatus } =
        queueApi.endpoints.getQueueStatus.select()(getOriginalState());

      if (
        oldQueueStatus?.started === false &&
        oldQueueStatus?.stop_after_current === true &&
        action.payload.data.started === false &&
        action.payload.data.stop_after_current === false
      ) {
        dispatch(
          addToast({ title: t('queue.stopSucceeded'), status: 'success' })
        );
      }

      dispatch(queueApi.util.invalidateTags(['SessionQueueStatus']));
    },
  });
};
