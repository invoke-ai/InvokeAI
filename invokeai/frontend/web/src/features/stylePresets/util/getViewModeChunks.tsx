import { PRESET_PLACEHOLDER } from 'features/stylePresets/hooks/usePresetModifiedPrompts';

export const getViewModeChunks = (currentPrompt: string, presetPrompt?: string): [string, string, string] => {
  if (!presetPrompt || !presetPrompt.length) {
    return ['', currentPrompt, ''];
  }

  const [before, after] = presetPrompt.split(PRESET_PLACEHOLDER, 2);

  if (!before || !after) {
    return ['', `${currentPrompt} `, presetPrompt];
  }

  return [before ?? '', currentPrompt, after ?? ''];
};
