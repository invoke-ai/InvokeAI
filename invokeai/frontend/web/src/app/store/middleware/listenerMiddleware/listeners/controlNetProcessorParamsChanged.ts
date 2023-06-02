import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import {
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
} from 'features/controlNet/store/controlNetSlice';

const moduleLog = log.child({ namespace: 'controlNet' });

export const addControlNetProcessorParamsChangedListener = () => {
  startAppListening({
    predicate: (action) =>
      controlNetProcessorParamsChanged.match(action) ||
      controlNetProcessorTypeChanged.match(action),
    effect: async (action, { dispatch, cancelActiveListeners, delay }) => {
      const { controlNetId } = action.payload;
      // Cancel any in-progress instances of this listener
      cancelActiveListeners();

      // Delay before starting actual work
      await delay(1000);

      dispatch(controlNetImageProcessed({ controlNetId }));
    },
  });
};
