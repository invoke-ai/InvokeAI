import { describe, expect, it } from 'vitest';

import {
  applyPromptTemplate,
  EMPTY_PROMPT_TEMPLATE_FIELD_VALUES,
  flattenPromptTemplateExpansion,
  getEffectivePrompts,
  getPositivePromptTemplateSegments,
  getPromptTemplateFields,
  getPromptTemplateSegments,
  isCanonicalPromptTemplateSnapshot,
  joinPromptTemplateSegments,
  normalizePromptTemplateFieldValues,
  retainPromptTemplateFieldValues,
  sanitizePromptTemplateSnapshot,
  setPromptTemplateFieldValue,
  syncPromptTemplateWithCatalog,
  type PromptTemplateSnapshot,
} from './promptTemplates';

const NO_VALUES = EMPTY_PROMPT_TEMPLATE_FIELD_VALUES;

const template: PromptTemplateSnapshot = {
  id: 'template-1',
  name: 'Cinematic',
  negativePrompt: '{prompt}, blurry',
  positivePrompt: 'photo of {prompt}, 35mm',
};

const fieldTemplate: PromptTemplateSnapshot = {
  id: 'template-2',
  name: 'Film still',
  negativePrompt: '{prompt}, not {style}',
  positivePrompt: 'a still of {subject}. {prompt}. {Style} style, {art_medium}',
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
    expect(applyPromptTemplate(templatePrompt, prompt, NO_VALUES)).toBe(expected);
  });

  it('keeps a JSON positive prompt whole under a template and still applies the negative side', () => {
    const layout = '```json\n{"bbox_coord_space":"abs_centralized","items":[]}\n```';
    expect(
      getEffectivePrompts({
        negativePrompt: 'noise',
        positivePrompt: layout,
        promptTemplate: fieldTemplate,
        promptTemplateFieldValues: { style: 'noir', subject: 'a fox' },
      })
    ).toEqual({ negativePrompt: 'noise, not noir', positivePrompt: layout });
  });

  it('splits the merged positive view as it generates, with a JSON prompt whole', () => {
    const layout = '{"bbox_coord_space":"abs_centralized","items":[]}';
    expect(getPositivePromptTemplateSegments(layout, template, NO_VALUES)).toEqual([{ kind: 'prompt', text: layout }]);
    expect(getPositivePromptTemplateSegments('owl', template, NO_VALUES)).toEqual([
      { kind: 'template', text: 'photo of ' },
      { kind: 'prompt', text: 'owl' },
      { kind: 'template', text: ', 35mm' },
    ]);
  });

  it('merges both prompt sides and passes them through when no template is active', () => {
    expect(getEffectivePrompts({ negativePrompt: 'noise', positivePrompt: 'owl' })).toEqual({
      negativePrompt: 'noise',
      positivePrompt: 'owl',
    });
    expect(
      getEffectivePrompts({
        negativePrompt: 'noise',
        positivePrompt: 'owl',
        promptTemplate: template,
        promptTemplateFieldValues: NO_VALUES,
      })
    ).toEqual({ negativePrompt: 'noise, blurry', positivePrompt: 'photo of owl, 35mm' });
  });

  it('flattens with the selected positive expansion and clears template state', () => {
    expect(
      flattenPromptTemplateExpansion({
        authoredNegativePrompt: 'noise',
        fieldValues: { style: 'noir' },
        selectedPositivePrompt: 'chosen expansion',
        template: fieldTemplate,
      })
    ).toEqual({
      negativePrompt: 'noise, not noir',
      positivePrompt: 'chosen expansion',
      promptTemplate: null,
      promptTemplateViewMode: false,
    });
  });

  it.each([
    ['subject', '', [{ kind: 'prompt', text: 'subject ' }]],
    [
      'subject',
      'cinematic',
      [
        { kind: 'prompt', text: 'subject ' },
        { kind: 'template', text: 'cinematic' },
      ],
    ],
    [
      'subject',
      '{prompt} and {prompt}',
      [
        { kind: 'prompt', text: 'subject' },
        { kind: 'template', text: ' and {prompt}' },
      ],
    ],
  ])('splits %j around the authored prompt', (prompt, templatePrompt, expected) => {
    expect(getPromptTemplateSegments(prompt, templatePrompt, NO_VALUES)).toEqual(expected);
  });
});

describe('prompt template fields', () => {
  it('lists each custom placeholder once, positive side first, under its first spelling', () => {
    expect(getPromptTemplateFields(fieldTemplate)).toEqual([
      { label: 'Subject', name: 'subject' },
      { label: 'Style', name: 'style' },
      { label: 'Art medium', name: 'art_medium' },
    ]);
    expect(getPromptTemplateFields({ negativePrompt: 'not {mood}', positivePrompt: '{prompt}' })).toEqual([
      { label: 'Mood', name: 'mood' },
    ]);
  });

  it('reads only bare names as fields', () => {
    const notFields =
      // eslint-disable-next-line no-template-curly-in-string -- `${season}` is dynamic-prompts syntax, not interpolation
      '{prompt} {Prompt} {red|blue} {2$$a|b} ${season} ${season=winter} {"key": 1} {two words} {} {9lives}';

    expect(getPromptTemplateFields({ negativePrompt: '', positivePrompt: notFields })).toEqual([]);
    expect(applyPromptTemplate(notFields, 'owl', { prompt: 'x', season: 'x' })).toBe(
      notFields.replace('{prompt}', 'owl')
    );
  });

  it('fills every occurrence on both sides and leaves an unfilled field empty', () => {
    expect(
      getEffectivePrompts({
        negativePrompt: 'noise',
        positivePrompt: 'at dusk',
        promptTemplate: fieldTemplate,
        promptTemplateFieldValues: { style: 'noir', subject: 'a fox' },
      })
    ).toEqual({
      negativePrompt: 'noise, not noir',
      positivePrompt: 'a still of a fox. at dusk. noir style, ',
    });
  });

  it('reads only a value of its own for a field named like an inherited property', () => {
    const templatePrompt = '{prompt} by {constructor}, {toString}';

    expect(applyPromptTemplate(templatePrompt, 'owl', NO_VALUES)).toBe('owl by , ');
    expect(applyPromptTemplate(templatePrompt, 'owl', { constructor: 'hand' })).toBe('owl by hand, ');
  });

  it('places a value as typed, without reading placeholders or replacement patterns in it', () => {
    const segments = getPromptTemplateSegments('{style} owl', '{subject}: {prompt} in {style}', {
      style: '$& {prompt}',
      subject: '{style} {prompt}',
    });

    expect(segments).toEqual([
      { kind: 'field', text: '{style} {prompt}' },
      { kind: 'template', text: ': ' },
      { kind: 'prompt', text: '{style} owl' },
      { kind: 'template', text: ' in ' },
      { kind: 'field', text: '$& {prompt}' },
    ]);
    expect(joinPromptTemplateSegments(segments)).toBe('{style} {prompt}: {style} owl in $& {prompt}');
  });

  it('normalizes stored values to usable entries, keeping canonical identity', () => {
    const canonical = { style: 'noir' };

    expect(normalizePromptTemplateFieldValues(canonical)).toBe(canonical);
    expect(normalizePromptTemplateFieldValues(undefined)).toBe(EMPTY_PROMPT_TEMPLATE_FIELD_VALUES);
    expect(normalizePromptTemplateFieldValues({ style: '' })).toBe(EMPTY_PROMPT_TEMPLATE_FIELD_VALUES);
    expect(
      normalizePromptTemplateFieldValues({ Mood: 'calm', prompt: 'x', style: 'noir', subject: 3, 'two words': 'x' })
    ).toEqual({ style: 'noir' });
  });

  it('stores values under sorted keys, so clearing and retyping one saves the same record', () => {
    const cleared = setPromptTemplateFieldValue({ style: 'noir', subject: 'a fox' }, 'style', '');

    expect(cleared).toEqual({ subject: 'a fox' });
    expect(JSON.stringify(setPromptTemplateFieldValue(cleared, 'style', 'noir'))).toBe(
      '{"style":"noir","subject":"a fox"}'
    );
    expect(JSON.stringify(normalizePromptTemplateFieldValues({ subject: 'a fox', style: 'noir' }))).toBe(
      '{"style":"noir","subject":"a fox"}'
    );
  });

  it('keeps only the values a newly chosen template declares', () => {
    const values = { style: 'noir', subject: 'a fox' };

    expect(retainPromptTemplateFieldValues(values, fieldTemplate)).toBe(values);
    expect(
      retainPromptTemplateFieldValues(values, { negativePrompt: '', positivePrompt: '{prompt}, {STYLE}' })
    ).toEqual({
      style: 'noir',
    });
    expect(retainPromptTemplateFieldValues(values, template)).toBe(EMPTY_PROMPT_TEMPLATE_FIELD_VALUES);
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
