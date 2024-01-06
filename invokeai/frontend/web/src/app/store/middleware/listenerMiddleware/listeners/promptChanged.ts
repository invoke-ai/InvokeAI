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
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
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
      cancelActiveListeners();
      const state = getState();
      const { positivePrompt } = state.generation;
      const { maxPrompts } = state.dynamicPrompts;

      if (state.config.disabledFeatures.includes('dynamicPrompting')) {
        return;
      }

      const cachedPrompts = utilitiesApi.endpoints.dynamicPrompts.select({
        prompt: positivePrompt,
        max_prompts: maxPrompts,
      })(getState()).data;

      if (cachedPrompts) {
        dispatch(promptsChanged(cachedPrompts.prompts));
        return;
      }

      if (!getShouldProcessPrompt(state.generation.positivePrompt)) {
        if (state.dynamicPrompts.isLoading) {
          dispatch(isLoadingChanged(false));
        }
        dispatch(promptsChanged([state.generation.positivePrompt]));
        return;
      }

      if (!state.dynamicPrompts.isLoading) {
        dispatch(isLoadingChanged(true));
      }

      // debounce request
      await delay(1000);

      try {
        const req = dispatch(
          utilitiesApi.endpoints.dynamicPrompts.initiate({
            prompt: positivePrompt,
            max_prompts: maxPrompts,
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
