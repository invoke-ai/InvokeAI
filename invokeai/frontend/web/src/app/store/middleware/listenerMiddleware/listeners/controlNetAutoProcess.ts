import { AnyListenerPredicate } from '@reduxjs/toolkit';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import {
  controlNetAutoConfigToggled,
  controlNetImageChanged,
  controlNetModelChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { RootState } from 'app/store/store';

const moduleLog = log.child({ namespace: 'controlNet' });

const predicate: AnyListenerPredicate<RootState> = (action, state) => {
  const isActionMatched =
    controlNetProcessorParamsChanged.match(action) ||
    controlNetModelChanged.match(action) ||
    controlNetImageChanged.match(action) ||
    controlNetProcessorTypeChanged.match(action) ||
    controlNetAutoConfigToggled.match(action);

  if (!isActionMatched) {
    return false;
  }

  const { controlImage, processorType, shouldAutoConfig } =
    state.controlNet.controlNets[action.payload.controlNetId];

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
    effect: async (
      action,
      { dispatch, getState, cancelActiveListeners, delay }
    ) => {
      const { controlNetId } = action.payload;

      // Cancel any in-progress instances of this listener
      cancelActiveListeners();
      moduleLog.trace(
        { data: action.payload },
        'ControlNet auto-process triggered'
      );
      // Delay before starting actual work
      await delay(300);

      dispatch(controlNetImageProcessed({ controlNetId }));
    },
  });
};
