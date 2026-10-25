import { describe, expect, it } from 'vitest';

import type { GenerateLora, GenerateModelConfig, LoraModelConfig } from './types';

import { addConcept, type ConceptUpdate, removeConcept, updateConcept } from './conceptList';

const lora = (key: string, base = 'sdxl', extra: Partial<LoraModelConfig> = {}): LoraModelConfig => ({
  base,
  key,
  name: key,
  type: 'lora',
  ...extra,
});

const SDXL: GenerateModelConfig = { base: 'sdxl', key: 'sdxl-main', name: 'SDXL', type: 'main' };
const INK = lora('ink');
const CHALK = lora('chalk');
const LIST: GenerateLora[] = [
  { isEnabled: true, model: INK, weight: 0.5 },
  { isEnabled: false, model: CHALK, weight: 1.25 },
];

describe('addConcept', () => {
  it.each([
    ['a compatible concept at the default weight', lora('glow'), 0.75],
    [
      'a compatible concept at its own default weight',
      lora('glow', 'sdxl', { default_settings: { weight: 1.4 } }),
      1.4,
    ],
  ])('appends %s, enabled', (_, model, weight) => {
    expect(addConcept(LIST, model, SDXL)).toEqual([...LIST, { isEnabled: true, model, weight }]);
  });

  it.each([
    ['is already listed', lora('chalk', 'sdxl', { name: 'Chalk renamed' }), SDXL],
    ['belongs to another family', lora('glow', 'sd-1'), SDXL],
    ['has no main model to load it', lora('glow'), null],
  ])('leaves the list unchanged when the concept %s', (_, model, mainModel) => {
    expect(addConcept(LIST, model, mainModel)).toBe(LIST);
  });

  it('checks Wan LoRA variants against the main model in the correct direction', () => {
    const model = lora('wan-5b', 'wan', { variant: '5b' });
    const mainModel: GenerateModelConfig = {
      base: 'wan',
      key: 'wan-main',
      name: 'Wan',
      type: 'main',
      variant: 't2v_a14b',
    };

    expect(addConcept(LIST, model, mainModel)).toBe(LIST);
    expect(addConcept(LIST, model, { ...mainModel, variant: 'ti2v_5b' })).toEqual([
      ...LIST,
      { isEnabled: true, model, weight: 0.75 },
    ]);
  });

  it('supports compatible video concepts outside the Generate model families', () => {
    const model = lora('minimax-lora', 'minimax-h3');
    const mainModel: GenerateModelConfig = {
      base: 'minimax-h3',
      key: 'minimax-main',
      name: 'MiniMax',
      type: 'main',
    };

    expect(addConcept(LIST, model, mainModel)).toEqual([...LIST, { isEnabled: true, model, weight: 0.75 }]);
  });
});

describe('updateConcept', () => {
  it.each<[string, ConceptUpdate, GenerateLora]>([
    ['weight', { weight: 0.9 }, { isEnabled: false, model: CHALK, weight: 0.9 }],
    ['enabled state', { isEnabled: true }, { isEnabled: true, model: CHALK, weight: 1.25 }],
    ['both fields', { isEnabled: true, weight: -1 }, { isEnabled: true, model: CHALK, weight: -1 }],
  ])('changes only the matching row’s %s', (_, update, expected) => {
    const next = updateConcept(LIST, 'chalk', update);

    expect(next).toEqual([LIST[0], expected]);
    expect(next[0]).toBe(LIST[0]);
  });

  it.each<[string, string, ConceptUpdate]>([
    ['the row was removed', 'gone', { weight: 0.9 }],
    ['the values already match', 'chalk', { isEnabled: false, weight: 1.25 }],
    ['the update is empty', 'ink', {}],
  ])('leaves the list unchanged when %s', (_, key, update) => {
    expect(updateConcept(LIST, key, update)).toBe(LIST);
  });

  it('updates every matching key in a loaded list without removing duplicate rows', () => {
    const list = [...LIST, { isEnabled: true, model: CHALK, weight: -0.5 }];
    const next = updateConcept(list, 'chalk', { isEnabled: false, weight: 0.9 });

    expect(next).toEqual([
      LIST[0],
      { isEnabled: false, model: CHALK, weight: 0.9 },
      { isEnabled: false, model: CHALK, weight: 0.9 },
    ]);
    expect(next[0]).toBe(list[0]);
  });

  it('updates a later duplicate when the first matching row already has the requested values', () => {
    const list = [...LIST, { isEnabled: true, model: CHALK, weight: -0.5 }];
    const next = updateConcept(list, 'chalk', { isEnabled: false, weight: 1.25 });

    expect(next).toEqual([...LIST, { isEnabled: false, model: CHALK, weight: 1.25 }]);
    expect(next[0]).toBe(list[0]);
    expect(next[1]).toBe(list[1]);
  });

  it('leaves a duplicate-key list unchanged when all matching rows already match', () => {
    const list = [...LIST, { isEnabled: false, model: CHALK, weight: 1.25 }];

    expect(updateConcept(list, 'chalk', { isEnabled: false, weight: 1.25 })).toBe(list);
  });
});

describe('removeConcept', () => {
  it('drops the matching row', () => {
    expect(removeConcept(LIST, 'ink')).toEqual([LIST[1]]);
  });

  it('leaves the list unchanged when the row is already gone', () => {
    expect(removeConcept(LIST, 'gone')).toBe(LIST);
  });
});
