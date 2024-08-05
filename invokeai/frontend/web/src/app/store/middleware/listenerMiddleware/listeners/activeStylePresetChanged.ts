import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { negativePromptChanged, positivePromptChanged, } from 'features/controlLayers/store/controlLayersSlice';
import { activeStylePresetChanged, calculatedNegPromptChanged, calculatedPosPromptChanged } from '../../../../../features/stylePresets/store/stylePresetSlice';
import { isAnyOf } from '@reduxjs/toolkit';

export const addActiveStylePresetChanged = (startAppListening: AppStartListening) => {
    startAppListening({
        matcher: isAnyOf(activeStylePresetChanged, positivePromptChanged, negativePromptChanged),
        effect: async (action, { dispatch, getState }) => {
            const state = getState();

            const activeStylePreset = state.stylePreset.activeStylePreset;
            const positivePrompt = state.controlLayers.present.positivePrompt
            const negativePrompt = state.controlLayers.present.negativePrompt

            if (!activeStylePreset) {
                return;
            }

            const { positive_prompt: presetPositivePrompt, negative_prompt: presetNegativePrompt } = activeStylePreset.preset_data;

            const calculatedPosPrompt = presetPositivePrompt.includes('{prompt}') ? presetPositivePrompt.replace('{prompt}', positivePrompt) : `${positivePrompt} ${presetPositivePrompt}`

            const calculatedNegPrompt = presetNegativePrompt.includes('{prompt}') ? presetNegativePrompt.replace('{prompt}', negativePrompt) : `${negativePrompt} ${presetNegativePrompt}`

            dispatch(calculatedPosPromptChanged(calculatedPosPrompt))

            dispatch(calculatedNegPromptChanged(calculatedNegPrompt))
        },
    });
};
