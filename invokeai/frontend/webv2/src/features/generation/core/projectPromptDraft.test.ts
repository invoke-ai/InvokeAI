import { describe, expect, it } from 'vitest';

import { applyProjectPromptDraft, getPromptDraftFromValues, migrateProjectPromptDraft } from './projectPromptDraft';

describe('project prompt draft', () => {
  it('normalizes missing prompt values', () => {
    expect(getPromptDraftFromValues({})).toEqual({
      negativePrompt: '',
      negativePromptEnabled: true,
      positivePrompt: '',
    });
  });

  it('patches only shared prompt fields', () => {
    const values = { negativePrompt: 'old', positivePrompt: 'old', positivePromptHeightPx: 120 };

    expect(applyProjectPromptDraft(values, { positivePrompt: 'new' })).toEqual({
      negativePrompt: 'old',
      negativePromptEnabled: true,
      positivePrompt: 'new',
      positivePromptHeightPx: 120,
    });
  });

  it('seeds an empty Generate prompt from a legacy Upscale prompt', () => {
    expect(
      migrateProjectPromptDraft(
        { positivePrompt: '', positivePromptHeightPx: 96 },
        { negativePrompt: 'blur', negativePromptEnabled: false, positivePrompt: 'fine detail' }
      )
    ).toEqual({
      negativePrompt: 'blur',
      negativePromptEnabled: false,
      positivePrompt: 'fine detail',
      positivePromptHeightPx: 96,
    });
  });

  it('keeps a non-empty Generate prompt during migration', () => {
    const generateValues = { negativePrompt: '', positivePrompt: 'generate prompt' };

    expect(migrateProjectPromptDraft(generateValues, { positivePrompt: 'upscale prompt' })).toBe(generateValues);
  });
});
