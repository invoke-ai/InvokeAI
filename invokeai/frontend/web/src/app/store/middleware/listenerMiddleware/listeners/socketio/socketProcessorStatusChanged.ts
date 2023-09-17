import { logger } from 'app/logging/logger';
import { queueApi } from 'services/api/endpoints/queue';
import {
  appSocketProcessorStatusChanged,
  socketProcessorStatusChanged,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addProcessorStatusChangedEventListener = () => {
  startAppListening({
    actionCreator: socketProcessorStatusChanged,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug(action.payload, 'Processor status changed');
      dispatch(appSocketProcessorStatusChanged(action.payload));
      dispatch(queueApi.util.invalidateTags(['SessionProcessorStatus']));
    },
  });
};
