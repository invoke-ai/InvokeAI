import { getViewModeChunks } from 'features/stylePresets/util/getViewModeChunks';
import { describe, expect, it } from 'vitest';

describe('getViewModeChunks', () => {
  it('should return empty strings when presetPrompt is not provided', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = undefined;
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['', currentPrompt, '']);
  });

  it('should return empty strings when presetPrompt is empty', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = '';
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['', currentPrompt, '']);
  });

  it('should append presetPrompt to currentPrompt when presetPrompt does not contain PRESET_PLACEHOLDER', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = 'preset prompt';
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['', `${currentPrompt} `, presetPrompt]);
  });

  it('should split presetPrompt into 3 parts when presetPrompt contains PRESET_PLACEHOLDER', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = 'before {prompt} after';
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['before ', currentPrompt, ' after']);
  });

  it('should split presetPrompt into 3 parts when presetPrompt contains multiple PRESET_PLACEHOLDER', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = 'before {prompt} middle {prompt} after';
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['before ', currentPrompt, ' middle {prompt} after']);
  });

  it('should handle the PRESET_PLACEHOLDER being at the start of the presetPrompt', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = '{prompt} after';
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['', currentPrompt, ' after']);
  });

  it('should handle the PRESET_PLACEHOLDER being at the end of the presetPrompt', () => {
    const currentPrompt = 'current prompt';
    const presetPrompt = 'before {prompt}';
    const result = getViewModeChunks(currentPrompt, presetPrompt);
    expect(result).toEqual(['before ', currentPrompt, '']);
  });
});
