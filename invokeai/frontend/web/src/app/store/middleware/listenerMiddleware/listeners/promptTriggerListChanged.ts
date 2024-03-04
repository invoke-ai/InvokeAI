import type { ComboboxOption } from '@invoke-ai/ui-library';
import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { loraAdded, loraIsEnabledChanged, loraRecalled, loraRemoved } from 'features/lora/store/loraSlice';
import { modelChanged, triggerPhrasesChanged } from 'features/parameters/store/generationSlice';
import { modelsApi } from 'services/api/endpoints/models';

const matcher = isAnyOf(loraAdded, loraRemoved, loraRecalled, loraIsEnabledChanged, modelChanged);

export const addPromptTriggerListChanged = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher,
    effect: async (action, { dispatch, getState, cancelActiveListeners }) => {
      cancelActiveListeners();
      const state = getState();
      const { model: mainModel } = state.generation;
      const { loras } = state.lora;

      let triggerPhrases: ComboboxOption[] = [];

      if (!mainModel) {
        dispatch(triggerPhrasesChanged([]));
        return;
      }

      const { data: mainModelData } = await dispatch(modelsApi.endpoints.getModelMetadata.initiate(mainModel.key));
      triggerPhrases = (mainModelData?.trigger_phrases || []).map((phrase) => ({ label: phrase, value: phrase }));

      for (let index = 0; index < Object.values(loras).length; index++) {
        const lora = Object.values(loras)[index];
        if (lora && lora.isEnabled) {
          const { data: loraMetadata } = await dispatch(modelsApi.endpoints.getModelMetadata.initiate(lora.model.key));
          const { data: loraConfig } = modelsApi.endpoints.getModelConfig.select(lora.model.key)(state);
          const loraTriggerPhrases = (loraMetadata?.trigger_phrases || []).map((phrase) => ({
            label: phrase,
            value: phrase,
            description: loraConfig?.name ? `(${loraConfig?.name})` : '',
          }));
          triggerPhrases = [...triggerPhrases, ...loraTriggerPhrases];
        }
      }

      dispatch(triggerPhrasesChanged(triggerPhrases));
    },
  });
};
