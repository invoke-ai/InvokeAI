import { isAnyOf } from '@reduxjs/toolkit';
import {
  combinatorialToggled,
  isErrorChanged,
  isLoadingChanged,
  maxPromptsChanged,
  maxPromptsReset,
  parsingErrorChanged,
  promptsChanged,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { setPositivePrompt } from 'features/parameters/store/generationSlice';
import { utilitiesApi } from 'services/api/endpoints/utilities';
import { appSocketConnected } from 'services/events/actions';
import { startAppListening } from '..';

const matcher = isAnyOf(
  setPositivePrompt,
  combinatorialToggled,
  maxPromptsChanged,
  maxPromptsReset,
  appSocketConnected
);

export const addDynamicPromptsListener = () => {
  startAppListening({
    matcher,
    effect: async (
      action,
      { dispatch, getState, cancelActiveListeners, delay }
    ) => {
      // debounce request
      cancelActiveListeners();
      await delay(1000);

      const state = getState();

      if (state.config.disabledFeatures.includes('dynamicPrompting')) {
        return;
      }

      const { positivePrompt } = state.generation;
      const { maxPrompts, combinatorial } = state.dynamicPrompts;

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
      }
    },
  });
};
