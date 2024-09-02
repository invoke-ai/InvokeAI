import { PRESET_PLACEHOLDER } from 'features/stylePresets/hooks/usePresetModifiedPrompts';

export const getViewModeChunks = (currentPrompt: string, presetPrompt?: string): [string, string, string] => {
  if (!presetPrompt || !presetPrompt.length) {
    return ['', currentPrompt, ''];
  }

  // When preset prompt does not contain the placeholder, we append the preset to the current prompt
  if (!presetPrompt.includes(PRESET_PLACEHOLDER)) {
    return ['', `${currentPrompt} `, presetPrompt];
  }

  // Otherwise, we split the preset prompt into 3 parts: before, current, and after the placeholder
  const [before, ...after] = presetPrompt.split(PRESET_PLACEHOLDER);

  return [before || '', currentPrompt, after.join(PRESET_PLACEHOLDER) || ''];
};
