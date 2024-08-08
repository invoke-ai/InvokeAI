import { PRESET_PLACEHOLDER } from '../hooks/usePresetModifiedPrompts';

export const getViewModeChunks = (currentPrompt: string, presetPrompt?: string) => {
  if (!presetPrompt || !presetPrompt.length) {
    return ['', currentPrompt, ''];
  }

  const chunks = presetPrompt.split(PRESET_PLACEHOLDER);

  if (chunks.length === 1) {
    return ['', currentPrompt, chunks[0]];
  } else {
    return [chunks[0], currentPrompt, chunks[1]];
  }
};
