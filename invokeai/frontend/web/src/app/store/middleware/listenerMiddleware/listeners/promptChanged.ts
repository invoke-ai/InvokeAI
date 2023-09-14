import { isAnyOf } from '@reduxjs/toolkit';
import {
  combinatorialToggled,
  isEnabledToggled,
  isErrorChanged,
  isLoadingChanged,
  maxPromptsChanged,
  maxPromptsReset,
  parsingErrorChanged,
  promptsChanged,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { setPositivePrompt } from 'features/parameters/store/generationSlice';
import { utilitiesApi } from 'services/api/endpoints/utilities';
import { startAppListening } from '..';
import { appSocketConnected } from 'services/events/actions';

const matcher = isAnyOf(
  setPositivePrompt,
  combinatorialToggled,
  maxPromptsChanged,
  maxPromptsReset,
  isEnabledToggled,
  appSocketConnected
);

export const addDynamicPromptsListener = () => {
  startAppListening({
    matcher,
    effect: async (
      action,
      { dispatch, getState, cancelActiveListeners, delay }
    ) => {
      // Cancel any in-progress instances of this listener
      cancelActiveListeners();
      // Delay before starting actual work (debounce)
      await delay(1000);

      const state = getState();
      const { positivePrompt } = state.generation;
      const { isEnabled, maxPrompts, combinatorial } = state.dynamicPrompts;

      if (!isEnabled) {
        return;
      }

      dispatch(isLoadingChanged(true));

      try {
        const req = dispatch(
          utilitiesApi.endpoints.dynamicPrompts.initiate({
            prompt: positivePrompt,
            max_prompts: maxPrompts,
            combinatorial: combinatorial,
          })
        );

        const res = await req.unwrap();
        req.unsubscribe();

        dispatch(promptsChanged(res.prompts));
        dispatch(parsingErrorChanged(res.error));
        dispatch(isErrorChanged(false));
        dispatch(isLoadingChanged(false));
      } catch {
        dispatch(isErrorChanged(true));
        dispatch(isLoadingChanged(false));
        //no-op
      }
    },
  });
};
