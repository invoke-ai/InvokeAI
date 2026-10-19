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
});

describe('removeConcept', () => {
  it('drops the matching row', () => {
    expect(removeConcept(LIST, 'ink')).toEqual([LIST[1]]);
  });

  it('leaves the list unchanged when the row is already gone', () => {
    expect(removeConcept(LIST, 'gone')).toBe(LIST);
  });
});
