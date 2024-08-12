import { PRESET_PLACEHOLDER } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
export const getViewModeChunks = (currentPrompt: string, presetPrompt?: string): [string, string, string] => {
  if (!presetPrompt || !presetPrompt.length) {
    return ['', currentPrompt, ''];
  }
  const [firstPart, ...remainingParts] = presetPrompt.split(PRESET_PLACEHOLDER);
  const chunks = [firstPart, remainingParts.join(PRESET_PLACEHOLDER)];
  if (chunks.length === 1) {
    return ['', currentPrompt, chunks[0] ?? ''];
  } else {
    return [chunks[0] ?? '', currentPrompt, chunks[1] ?? ''];
  }
};
