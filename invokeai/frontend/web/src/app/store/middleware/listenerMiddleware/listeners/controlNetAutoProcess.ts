import type { AnyListenerPredicate } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { controlAdapterImageProcessed } from 'features/controlAdapters/store/actions';
import {
  controlAdapterAutoConfigToggled,
  controlAdapterImageChanged,
  controlAdapterModelChanged,
  controlAdapterProcessorParamsChanged,
  controlAdapterProcessortTypeChanged,
  selectControlAdapterById,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';

import { startAppListening } from '..';

type AnyControlAdapterParamChangeAction =
  | ReturnType<typeof controlAdapterProcessorParamsChanged>
  | ReturnType<typeof controlAdapterModelChanged>
  | ReturnType<typeof controlAdapterImageChanged>
  | ReturnType<typeof controlAdapterProcessortTypeChanged>
  | ReturnType<typeof controlAdapterAutoConfigToggled>;

const predicate: AnyListenerPredicate<RootState> = (action, state, prevState) => {
  const isActionMatched =
    controlAdapterProcessorParamsChanged.match(action) ||
    controlAdapterModelChanged.match(action) ||
    controlAdapterImageChanged.match(action) ||
    controlAdapterProcessortTypeChanged.match(action) ||
    controlAdapterAutoConfigToggled.match(action);

  if (!isActionMatched) {
    return false;
  }

  const { id } = action.payload;
  const prevCA = selectControlAdapterById(prevState.controlAdapters, id);
  const ca = selectControlAdapterById(state.controlAdapters, id);
  if (!prevCA || !isControlNetOrT2IAdapter(prevCA) || !ca || !isControlNetOrT2IAdapter(ca)) {
    return false;
  }

  if (controlAdapterAutoConfigToggled.match(action)) {
    // do not process if the user just disabled auto-config
    if (prevCA.shouldAutoConfig === true) {
      return false;
    }
  }

  const { controlImage, processorType, shouldAutoConfig } = ca;
  if (controlAdapterModelChanged.match(action) && !shouldAutoConfig) {
    // do not process if the action is a model change but the processor settings are dirty
    return false;
  }

  const isProcessorSelected = processorType !== 'none';

  const hasControlImage = Boolean(controlImage);

  return isProcessorSelected && hasControlImage;
};

const DEBOUNCE_MS = 300;

/**
 * Listener that automatically processes a ControlNet image when its processor parameters are changed.
 *
 * The network request is debounced.
 */
export const addControlNetAutoProcessListener = () => {
  startAppListening({
    predicate,
    effect: async (action, { dispatch, cancelActiveListeners, delay }) => {
      const log = logger('session');
      const { id } = (action as AnyControlAdapterParamChangeAction).payload;

      // Cancel any in-progress instances of this listener
      cancelActiveListeners();
      log.trace('ControlNet auto-process triggered');
      // Delay before starting actual work
      await delay(DEBOUNCE_MS);

      dispatch(controlAdapterImageProcessed({ id }));
    },
  });
};
