import { AnyAction } from '@reduxjs/toolkit';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import {
  controlNetImageChanged,
  controlNetProcessorParamsChanged,
  controlNetProcessorTypeChanged,
  isControlNetImagePreprocessedToggled,
} from 'features/controlNet/store/controlNetSlice';
import { RootState } from 'app/store/store';

const moduleLog = log.child({ namespace: 'controlNet' });

const predicate = (action: AnyAction, state: RootState) => {
  const isActionMatched =
    controlNetProcessorParamsChanged.match(action) ||
    controlNetImageChanged.match(action) ||
    controlNetProcessorTypeChanged.match(action) ||
    isControlNetImagePreprocessedToggled.match(action);

  if (!isActionMatched) {
    return false;
  }

  const { controlNetId } = action.payload;

  const shouldAutoProcess =
    !state.controlNet.controlNets[controlNetId].isPreprocessed;

  const isBusy = state.system.isProcessing;

  const hasControlImage = Boolean(
    state.controlNet.controlNets[controlNetId].controlImage
  );

  return shouldAutoProcess && !isBusy && hasControlImage;
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

      // Delay before starting actual work
      await delay(300);

      dispatch(controlNetImageProcessed({ controlNetId }));
    },
  });
};
