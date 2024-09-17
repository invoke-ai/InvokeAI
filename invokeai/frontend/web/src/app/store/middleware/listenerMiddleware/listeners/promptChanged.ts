import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { positivePromptChanged } from 'features/controlLayers/store/paramsSlice';
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
import { getPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import { activeStylePresetIdChanged } from 'features/stylePresets/store/stylePresetSlice';
import { stylePresetsApi } from 'services/api/endpoints/stylePresets';
import { utilitiesApi } from 'services/api/endpoints/utilities';

import { socketConnected } from './socketConnected';

const matcher = isAnyOf(
  positivePromptChanged,
  combinatorialToggled,
  maxPromptsChanged,
  maxPromptsReset,
  socketConnected,
  activeStylePresetIdChanged,
  stylePresetsApi.endpoints.listStylePresets.matchFulfilled
);

export const addDynamicPromptsListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher,
    effect: async (action, { dispatch, getState, cancelActiveListeners, delay }) => {
      cancelActiveListeners();
      const state = getState();
      const { positivePrompt } = getPresetModifiedPrompts(state);
      const { maxPrompts } = state.dynamicPrompts;

      if (state.config.disabledFeatures.includes('dynamicPrompting')) {
        return;
      }

      const cachedPrompts = utilitiesApi.endpoints.dynamicPrompts.select({
        prompt: positivePrompt,
        max_prompts: maxPrompts,
      })(state).data;

      if (cachedPrompts) {
        dispatch(promptsChanged(cachedPrompts.prompts));
        dispatch(parsingErrorChanged(cachedPrompts.error));
        return;
      }

      if (!getShouldProcessPrompt(positivePrompt)) {
        dispatch(promptsChanged([positivePrompt]));
        dispatch(parsingErrorChanged(undefined));
        dispatch(isErrorChanged(false));
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
      } catch {
        dispatch(isErrorChanged(true));
        dispatch(isLoadingChanged(false));
      }
    },
  });
};
