import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import {
  controlNetImageChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
} from 'features/controlNet/store/controlNetSlice';

const moduleLog = log.child({ namespace: 'controlNet' });

/**
 * Listener that automatically processes a ControlNet image when its processor parameters are changed.
 *
 * The network request is debounced by 1 second.
 */
export const addControlNetAutoProcessListener = () => {
  startAppListening({
    predicate: (action) =>
      controlNetProcessorParamsChanged.match(action) ||
      controlNetImageChanged.match(action) ||
      controlNetProcessorTypeChanged.match(action),
    effect: async (
      action,
      { dispatch, getState, cancelActiveListeners, delay }
    ) => {
      const state = getState();
      if (!state.controlNet.shouldAutoProcess) {
        // silently skip
        return;
      }

      if (state.system.isProcessing) {
        moduleLog.trace('System busy, skipping ControlNet auto-processing');
        return;
      }

      const { controlNetId } = action.payload;

      if (!state.controlNet.controlNets[controlNetId].controlImage) {
        moduleLog.trace(
          { data: { controlNetId } },
          'No ControlNet image to auto-process'
        );
        return;
      }

      // Cancel any in-progress instances of this listener
      cancelActiveListeners();

      // Delay before starting actual work
      await delay(300);

      dispatch(controlNetImageProcessed({ controlNetId }));
    },
  });
};
