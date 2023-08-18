import { AnyListenerPredicate } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import {
  controlNetAutoConfigToggled,
  controlNetImageChanged,
  controlNetModelChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { startAppListening } from '..';

const predicate: AnyListenerPredicate<RootState> = (
  action,
  state,
  prevState
) => {
  const isActionMatched =
    controlNetProcessorParamsChanged.match(action) ||
    controlNetModelChanged.match(action) ||
    controlNetImageChanged.match(action) ||
    controlNetProcessorTypeChanged.match(action) ||
    controlNetAutoConfigToggled.match(action);

  if (!isActionMatched) {
    return false;
  }

  if (controlNetAutoConfigToggled.match(action)) {
    // do not process if the user just disabled auto-config
    if (
      prevState.controlNet.controlNets[action.payload.controlNetId]
        ?.shouldAutoConfig === true
    ) {
      return false;
    }
  }

  const cn = state.controlNet.controlNets[action.payload.controlNetId];

  if (!cn) {
    // something is wrong, the controlNet should exist
    return false;
  }

  const { controlImage, processorType, shouldAutoConfig } = cn;
  if (controlNetModelChanged.match(action) && !shouldAutoConfig) {
    // do not process if the action is a model change but the processor settings are dirty
    return false;
  }

  const isProcessorSelected = processorType !== 'none';

  const isBusy = state.system.isProcessing;

  const hasControlImage = Boolean(controlImage);

  return isProcessorSelected && !isBusy && hasControlImage;
};

/**
 * Listener that automatically processes a ControlNet image when its processor parameters are changed.
 *
 * The network request is debounced by 1 second.
 */
export const addControlNetAutoProcessListener = () => {
  startAppListening({
    predicate,
    effect: async (action, { dispatch, cancelActiveListeners, delay }) => {
      const log = logger('session');
      const { controlNetId } = action.payload;

      // Cancel any in-progress instances of this listener
      cancelActiveListeners();
      log.trace('ControlNet auto-process triggered');
      // Delay before starting actual work
      await delay(300);

      dispatch(controlNetImageProcessed({ controlNetId }));
    },
  });
};
