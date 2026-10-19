import { describe, expect, it } from 'vitest';

import type { StarterModel } from './types';

import { DEFAULT_STARTER_MODEL_FILTERS, filterStarterModels } from './starters';

const createStarter = (overrides: Partial<StarterModel>): StarterModel => ({
  base: 'sdxl',
  description: 'A starter model',
  is_installed: false,
  name: 'Model',
  source: 'owner/repo',
  type: 'main',
  ...overrides,
});

const catalog: StarterModel[] = [
  createStarter({ base: 'sdxl', name: 'Juggernaut XL', type: 'main' }),
  createStarter({ base: 'flux', description: 'Fast FLUX finetune', name: 'Alpha FLUX', type: 'main' }),
  createStarter({ base: 'sd-1', name: 'Detail Tweaker', type: 'lora', variant: 'inpaint' }),
];

describe('filterStarterModels', () => {
  it('keeps catalog order under the default sort', () => {
    expect(filterStarterModels(catalog, DEFAULT_STARTER_MODEL_FILTERS, '').map((m) => m.name)).toEqual([
      'Juggernaut XL',
      'Alpha FLUX',
      'Detail Tweaker',
    ]);
  });

  it('matches search terms against name, description, taxonomy, and variant', () => {
    expect(filterStarterModels(catalog, DEFAULT_STARTER_MODEL_FILTERS, 'finetune').map((m) => m.name)).toEqual([
      'Alpha FLUX',
    ]);
    expect(filterStarterModels(catalog, DEFAULT_STARTER_MODEL_FILTERS, 'inpaint lora').map((m) => m.name)).toEqual([
      'Detail Tweaker',
    ]);
    expect(filterStarterModels(catalog, DEFAULT_STARTER_MODEL_FILTERS, 'nonexistent')).toEqual([]);
  });

  it('filters by type and base', () => {
    expect(
      filterStarterModels(catalog, { ...DEFAULT_STARTER_MODEL_FILTERS, typeFilter: 'lora' }, '').map((m) => m.name)
    ).toEqual(['Detail Tweaker']);
    expect(
      filterStarterModels(catalog, { ...DEFAULT_STARTER_MODEL_FILTERS, baseFilter: 'flux' }, '').map((m) => m.name)
    ).toEqual(['Alpha FLUX']);
  });

  it('sorts by name in either direction', () => {
    const byName = { ...DEFAULT_STARTER_MODEL_FILTERS, sortField: 'name' as const };

    expect(filterStarterModels(catalog, byName, '').map((m) => m.name)).toEqual([
      'Alpha FLUX',
      'Detail Tweaker',
      'Juggernaut XL',
    ]);
    expect(filterStarterModels(catalog, { ...byName, sortDirection: 'desc' as const }, '').map((m) => m.name)).toEqual([
      'Juggernaut XL',
      'Detail Tweaker',
      'Alpha FLUX',
    ]);
  });

  it('sorts by base', () => {
    // Note: label order and raw-id order coincide for every registry base, so
    // this cannot distinguish the two — it pins the ordering, not the source.
    expect(
      filterStarterModels(catalog, { ...DEFAULT_STARTER_MODEL_FILTERS, sortField: 'base' as const }, '').map(
        (m) => m.base
      )
    ).toEqual(['flux', 'sd-1', 'sdxl']);
  });

  it('sorts unknown bases deterministically via their fallback labels', () => {
    // 'zeta-arch' has no registry entry; its display label is "Zeta Arch".
    const withUnknown = [...catalog, createStarter({ base: 'zeta-arch', name: 'Zeta Model' })];

    expect(
      filterStarterModels(withUnknown, { ...DEFAULT_STARTER_MODEL_FILTERS, sortField: 'base' as const }, '').map(
        (m) => m.base
      )
    ).toEqual(['flux', 'sd-1', 'sdxl', 'zeta-arch']);
  });
});
