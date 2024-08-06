import { useAppSelector } from "app/store/storeHooks"

export const PRESET_PLACEHOLDER = `{prompt}`

export const buildPresetModifiedPrompt = (presetPrompt: string, currentPrompt: string) => {
    return presetPrompt.includes(PRESET_PLACEHOLDER) ? presetPrompt.replace(new RegExp(PRESET_PLACEHOLDER), currentPrompt) : `${currentPrompt} ${presetPrompt}`
}


export const usePresetModifiedPrompts = () => {
    const activeStylePreset = useAppSelector(s => s.stylePreset.activeStylePreset)
    const { positivePrompt, negativePrompt } = useAppSelector(s => s.controlLayers.present)

    if (!activeStylePreset) {
        return { presetModifiedPositivePrompt: positivePrompt, presetModifiedNegativePrompt: negativePrompt }
    }

    const { positive_prompt: presetPositivePrompt, negative_prompt: presetNegativePrompt } = activeStylePreset.preset_data;

    const presetModifiedPositivePrompt = buildPresetModifiedPrompt(presetPositivePrompt, positivePrompt)

    const presetModifiedNegativePrompt = buildPresetModifiedPrompt(presetNegativePrompt, negativePrompt)

    return { presetModifiedPositivePrompt, presetModifiedNegativePrompt }
}