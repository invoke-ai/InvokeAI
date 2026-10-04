import { describe, expect, it } from 'vitest';

import {
  applyPromptTemplate,
  flattenPromptTemplateExpansion,
  getEffectivePrompts,
  getPromptTemplateChunks,
  isCanonicalPromptTemplateSnapshot,
  sanitizePromptTemplateSnapshot,
  syncPromptTemplateWithCatalog,
  type PromptTemplateSnapshot,
} from './promptTemplates';

const template: PromptTemplateSnapshot = {
  id: 'template-1',
  name: 'Cinematic',
  negativePrompt: '{prompt}, blurry',
  positivePrompt: 'photo of {prompt}, 35mm',
};

describe('prompt template expansion', () => {
  it.each([
    ['before {prompt} after', 'subject', 'before subject after'],
    ['{prompt} after', 'subject', 'subject after'],
    ['before {prompt}', 'subject', 'before subject'],
    ['before {prompt} after {prompt}', 'subject', 'before subject after {prompt}'],
    ['cinematic', 'subject', 'subject cinematic'],
    ['', 'subject', 'subject '],
    ['{prompt}', 'costs $$$ and $&', 'costs $$$ and $&'],
  ])('applies %j to %j', (templatePrompt, prompt, expected) => {
    expect(applyPromptTemplate(templatePrompt, prompt)).toBe(expected);
  });

  it('merges both prompt sides and passes them through when no template is active', () => {
    expect(getEffectivePrompts({ negativePrompt: 'noise', positivePrompt: 'owl' })).toEqual({
      negativePrompt: 'noise',
      positivePrompt: 'owl',
    });
    expect(getEffectivePrompts({ negativePrompt: 'noise', positivePrompt: 'owl', promptTemplate: template })).toEqual({
      negativePrompt: 'noise, blurry',
      positivePrompt: 'photo of owl, 35mm',
    });
  });

  it('flattens with the selected positive expansion and clears template state', () => {
    expect(
      flattenPromptTemplateExpansion({
        authoredNegativePrompt: 'noise',
        selectedPositivePrompt: 'chosen expansion',
        template,
      })
    ).toEqual({
      negativePrompt: 'noise, blurry',
      positivePrompt: 'chosen expansion',
      promptTemplate: null,
      promptTemplateViewMode: false,
    });
  });

  it.each([
    ['subject', '', ['', 'subject ', '']],
    ['subject', 'cinematic', ['', 'subject ', 'cinematic']],
    ['subject', 'before {prompt} after', ['before ', 'subject', ' after']],
    ['subject', '{prompt} and {prompt}', ['', 'subject', ' and {prompt}']],
  ])('splits %j around the authored prompt', (prompt, templatePrompt, expected) => {
    const chunks = getPromptTemplateChunks(prompt, templatePrompt);

    expect(chunks).toEqual(expected);
    expect(chunks.join('')).toBe(applyPromptTemplate(templatePrompt, prompt));
  });
});

describe('prompt template snapshots', () => {
  it('sanitizes untrusted values and rejects unusable identities', () => {
    expect(sanitizePromptTemplateSnapshot(template)).toEqual(template);
    expect(sanitizePromptTemplateSnapshot({ id: 'template-1', name: 'Cinematic' })).toEqual({
      id: 'template-1',
      name: 'Cinematic',
      negativePrompt: '',
      positivePrompt: '',
    });
    expect(sanitizePromptTemplateSnapshot(null)).toBeNull();
    expect(sanitizePromptTemplateSnapshot({ id: '', name: 'Cinematic' })).toBeNull();
    expect(sanitizePromptTemplateSnapshot({ id: 'template-1', name: 3 })).toBeNull();
  });

  it('recognizes only exact canonical snapshots', () => {
    expect(isCanonicalPromptTemplateSnapshot(template)).toBe(true);
    expect(isCanonicalPromptTemplateSnapshot({ ...template, extra: true })).toBe(false);
    expect(isCanonicalPromptTemplateSnapshot({ ...template, id: '' })).toBe(false);
    expect(isCanonicalPromptTemplateSnapshot({ ...template, positivePrompt: null })).toBe(false);
    expect(isCanonicalPromptTemplateSnapshot(null)).toBe(false);
  });

  it('refreshes changed catalog fields without persisting record metadata', () => {
    const current = { ...template, name: 'Updated', hasImage: true };

    expect(syncPromptTemplateWithCatalog(template, [current])).toEqual({
      id: template.id,
      name: 'Updated',
      negativePrompt: template.negativePrompt,
      positivePrompt: template.positivePrompt,
    });
  });

  it('preserves identity when unchanged and keeps deleted snapshots', () => {
    expect(syncPromptTemplateWithCatalog(template, [{ ...template }])).toBe(template);
    expect(syncPromptTemplateWithCatalog(template, [])).toBe(template);
    expect(syncPromptTemplateWithCatalog(null, [template])).toBeNull();
  });
});
