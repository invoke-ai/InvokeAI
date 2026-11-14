import { describe, expect, it } from 'vitest';

import { applyProjectPromptDraft, getPromptDraftFromValues, migrateProjectPromptDraft } from './projectPromptDraft';

describe('project prompt draft', () => {
  it('normalizes missing prompt values', () => {
    expect(getPromptDraftFromValues({})).toEqual({
      dynamicPromptsCombinatorial: true,
      dynamicPromptsEnabled: true,
      dynamicPromptsMaxPrompts: 100,
      dynamicPromptsResample: true,
      dynamicPromptsSampleSeed: 0,
      dynamicPromptsSeedBehaviour: 'per-iteration',
      negativePrompt: '',
      negativePromptEnabled: true,
      positivePrompt: '',
      promptTemplate: null,
      promptTemplateViewMode: false,
    });
  });

  it('reads the template and its merged view as Generate stores them', () => {
    const template = { id: 'cinematic', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, film' };

    expect(getPromptDraftFromValues({ promptTemplate: template, promptTemplateViewMode: true })).toMatchObject({
      promptTemplate: template,
      promptTemplateViewMode: true,
    });
    // Nothing to show merged without a template.
    expect(getPromptDraftFromValues({ promptTemplate: { name: 'no id' }, promptTemplateViewMode: true })).toMatchObject(
      { promptTemplate: null, promptTemplateViewMode: false }
    );
  });

  it('applies a template and dynamic prompts setting without touching the other values', () => {
    const template = { id: 'cinematic', name: 'Cinematic', negativePrompt: 'blur', positivePrompt: '{prompt}, film' };
    const values = { dynamicPromptsMaxPrompts: 20, positivePrompt: 'a fox', positivePromptHeightPx: 120 };
    const applied = applyProjectPromptDraft(values, { dynamicPromptsCombinatorial: false, promptTemplate: template });

    expect(applied).toEqual({
      dynamicPromptsCombinatorial: false,
      dynamicPromptsMaxPrompts: 20,
      negativePrompt: '',
      negativePromptEnabled: true,
      positivePrompt: 'a fox',
      positivePromptHeightPx: 120,
      promptTemplate: template,
      promptTemplateViewMode: false,
    });
  });

  it('leaves the merged view when the template is cleared', () => {
    const template = { id: 'cinematic', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, film' };
    const values = { positivePrompt: 'a fox', promptTemplate: template, promptTemplateViewMode: true };

    expect(applyProjectPromptDraft(values, { promptTemplate: null })).toMatchObject({
      promptTemplate: null,
      promptTemplateViewMode: false,
    });
  });

  it('returns the same values when a patch changes nothing', () => {
    const values = { dynamicPromptsEnabled: false, positivePrompt: 'a fox' };

    expect(applyProjectPromptDraft(values, { dynamicPromptsEnabled: false, positivePrompt: 'a fox' })).toBe(values);
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

  it("moves only the legacy prompt text, leaving Generate's template in place", () => {
    const template = { id: 'cinematic', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, film' };

    expect(
      migrateProjectPromptDraft(
        { dynamicPromptsEnabled: false, promptTemplate: template },
        { positivePrompt: 'detail' }
      )
    ).toMatchObject({ dynamicPromptsEnabled: false, positivePrompt: 'detail', promptTemplate: template });
  });

  it('keeps a non-empty Generate prompt during migration', () => {
    const generateValues = { negativePrompt: '', positivePrompt: 'generate prompt' };

    expect(migrateProjectPromptDraft(generateValues, { positivePrompt: 'upscale prompt' })).toBe(generateValues);
  });
});
