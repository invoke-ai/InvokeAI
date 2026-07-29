import { describe, expect, it } from 'vitest';

import type { PromptTriggerOption } from './promptTriggerOptions';

import {
  filterPromptTriggerOptions,
  filterPromptTriggerOptionsByKind,
  getInlineTriggerOptions,
  getPromptTriggerOptions,
} from './promptTriggerOptions';

const OPTIONS: PromptTriggerOption[] = [
  { group: 'Wildcards', kind: 'wildcard', label: 'colours', value: '__colours__' },
  { group: 'Wildcards', kind: 'wildcard', label: 'moods', value: '__moods__' },
  { group: 'Compatible embeddings', kind: 'embedding', label: 'easynegative', value: '<easynegative>' },
  { group: 'My LoRA', kind: 'phrase', label: 'a photo of sks', value: 'a photo of sks' },
];

describe('getPromptTriggerOptions', () => {
  const build = () =>
    getPromptTriggerOptions({
      compatibleEmbeddingsLabel: 'Compatible embeddings',
      loras: [
        { isEnabled: true, model: { name: 'My LoRA', trigger_phrases: ['sks'] } },
        { isEnabled: false, model: { name: 'Off LoRA', trigger_phrases: ['nope'] } },
      ] as never,
      mainModelLabel: 'Main model',
      models: [
        { base: 'sdxl', name: 'easynegative', type: 'embedding' },
        { base: 'sd-1', name: 'wrongbase', type: 'embedding' },
      ] as never,
      selectedModel: { base: 'sdxl', name: 'Juggernaut', trigger_phrases: ['jugg'] } as never,
      wildcards: [{ name: 'colours', values: ['red'] }],
      wildcardsLabel: 'Wildcards',
    });

  it('tags every option with what it inserts', () => {
    expect(build().map((option) => [option.kind, option.value])).toEqual([
      ['wildcard', '__colours__'],
      ['phrase', 'jugg'],
      ['phrase', 'sks'],
      ['embedding', '<easynegative>'],
    ]);
  });

  it('skips disabled LoRAs and embeddings of another base', () => {
    const labels = build().map((option) => option.label);

    expect(labels).not.toContain('nope');
    expect(labels).not.toContain('wrongbase');
  });
});

describe('getInlineTriggerOptions', () => {
  // `<` opens an embedding token and `__` a wildcard reference, so offering the
  // other kind under either is offering something the user cannot have meant.
  it('offers only wildcards for `__`', () => {
    expect(getInlineTriggerOptions(OPTIONS, '_', '').map((option) => option.label)).toEqual(['colours', 'moods']);
  });

  it('offers only embeddings for `<`', () => {
    expect(getInlineTriggerOptions(OPTIONS, '<', '').map((option) => option.label)).toEqual(['easynegative']);
  });

  // They belong to no delimiter, and stay reachable through the `+` button.
  it('offers trigger phrases under neither', () => {
    const everything = [...getInlineTriggerOptions(OPTIONS, '_', ''), ...getInlineTriggerOptions(OPTIONS, '<', '')];

    expect(everything.some((option) => option.kind === 'phrase')).toBe(false);
  });

  it('narrows by label', () => {
    expect(getInlineTriggerOptions(OPTIONS, '_', 'mo').map((option) => option.label)).toEqual(['moods']);
    expect(getInlineTriggerOptions(OPTIONS, '_', 'MO').map((option) => option.label)).toEqual(['moods']);
  });

  // `__w` matching every wildcard on the strength of the word "Wildcards" is
  // what folding the group into the query would do.
  it('ignores the group name, unlike the browse search', () => {
    expect(getInlineTriggerOptions(OPTIONS, '_', 'wildcard')).toEqual([]);
    expect(filterPromptTriggerOptions(OPTIONS, 'wildcard')).toHaveLength(2);
  });

  it('returns nothing when the query matches no label', () => {
    expect(getInlineTriggerOptions(OPTIONS, '_', 'zzz')).toEqual([]);
  });
});

describe('filterPromptTriggerOptionsByKind', () => {
  it('keeps phrases and embeddings while excluding wildcards for a negative prompt', () => {
    expect(
      filterPromptTriggerOptionsByKind(OPTIONS, new Set(['embedding', 'phrase'])).map((option) => option.kind)
    ).toEqual(['embedding', 'phrase']);
  });
});
