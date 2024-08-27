import { useAppSelector } from 'app/store/storeHooks';
import { selectNegativePrompt, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { selectStylePresetActivePresetId } from 'features/stylePresets/store/stylePresetSlice';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

export const PRESET_PLACEHOLDER = '{prompt}';

export const buildPresetModifiedPrompt = (presetPrompt: string, currentPrompt: string) => {
  return presetPrompt.includes(PRESET_PLACEHOLDER)
    ? presetPrompt.replace(PRESET_PLACEHOLDER, currentPrompt)
    : `${currentPrompt} ${presetPrompt}`;
};

export const usePresetModifiedPrompts = () => {
  const positivePrompt = useAppSelector(selectPositivePrompt);
  const negativePrompt = useAppSelector(selectNegativePrompt);
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);

  const { activeStylePreset } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data }) => {
      let activeStylePreset = null;
      if (data) {
        activeStylePreset = data.find((sp) => sp.id === activeStylePresetId);
      }
      return { activeStylePreset };
    },
  });

  if (!activeStylePreset) {
    return { presetModifiedPositivePrompt: positivePrompt, presetModifiedNegativePrompt: negativePrompt };
  }

  const { positive_prompt: presetPositivePrompt, negative_prompt: presetNegativePrompt } =
    activeStylePreset.preset_data;

  const presetModifiedPositivePrompt = buildPresetModifiedPrompt(presetPositivePrompt, positivePrompt);

  const presetModifiedNegativePrompt = buildPresetModifiedPrompt(presetNegativePrompt, negativePrompt);

  return { presetModifiedPositivePrompt, presetModifiedNegativePrompt };
};
