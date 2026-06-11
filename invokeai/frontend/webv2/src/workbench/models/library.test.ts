import { describe, expect, it } from 'vitest';

import {
  collectBases,
  collectTypes,
  DEFAULT_LIBRARY_FILTERS,
  filterModels,
  flattenGroupsToRows,
  groupModelsByType,
} from './library';
import type { ModelConfig } from './types';

const createModel = (overrides: Partial<ModelConfig>): ModelConfig => ({
  base: 'sdxl',
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash',
  key: 'key',
  name: 'Model',
  path: '/models/model.safetensors',
  source: '/models/model.safetensors',
  source_type: 'path',
  type: 'main',
  ...overrides,
});

const library: ModelConfig[] = [
  createModel({ base: 'sdxl', file_size: 300, key: 'a', name: 'Juggernaut XL', type: 'main' }),
  createModel({ base: 'flux', file_size: 100, key: 'b', name: 'FLUX dev', type: 'main' }),
  createModel({
    base: 'sd-1',
    file_size: 200,
    key: 'c',
    name: 'Detail Tweaker',
    trigger_phrases: ['detail'],
    type: 'lora',
  }),
  createModel({ base: 'any', file_size: 50, key: 'd', name: 'Some VAE', type: 'vae' }),
];

const NO_MISSING: ReadonlySet<string> = new Set();

describe('filterModels', () => {
  it('matches search terms against name, taxonomy, and trigger phrases', () => {
    expect(
      filterModels(library, { ...DEFAULT_LIBRARY_FILTERS, searchTerm: 'flux' }, NO_MISSING).map((m) => m.key)
    ).toEqual(['b']);
    expect(
      filterModels(library, { ...DEFAULT_LIBRARY_FILTERS, searchTerm: 'detail' }, NO_MISSING).map((m) => m.key)
    ).toEqual(['c']);
  });

  it('filters by type and base independently', () => {
    expect(
      filterModels(library, { ...DEFAULT_LIBRARY_FILTERS, typeFilter: 'lora' }, NO_MISSING).map((m) => m.key)
    ).toEqual(['c']);
    expect(
      filterModels(library, { ...DEFAULT_LIBRARY_FILTERS, baseFilter: 'sdxl' }, NO_MISSING).map((m) => m.key)
    ).toEqual(['a']);
  });

  it('filters to missing models only', () => {
    const missing = new Set(['d']);

    expect(filterModels(library, { ...DEFAULT_LIBRARY_FILTERS, missingOnly: true }, missing).map((m) => m.key)).toEqual(
      ['d']
    );
  });

  it('sorts by size descending', () => {
    const sorted = filterModels(
      library,
      { ...DEFAULT_LIBRARY_FILTERS, sortDirection: 'desc', sortField: 'size' },
      NO_MISSING
    );

    expect(sorted.map((m) => m.key)).toEqual(['a', 'c', 'b', 'd']);
  });
});

describe('groupModelsByType', () => {
  it('groups in canonical category order with headers flattened for virtualization', () => {
    const groups = groupModelsByType(filterModels(library, DEFAULT_LIBRARY_FILTERS, NO_MISSING));

    expect(groups.map((group) => group.type)).toEqual(['main', 'lora', 'vae']);

    const rows = flattenGroupsToRows(groups);

    expect(rows[0]).toMatchObject({ kind: 'header' });
    expect(rows).toHaveLength(groups.length + library.length);
  });
});

describe('collect helpers', () => {
  it('collects distinct bases alphabetically and types in category order', () => {
    expect(collectBases(library)).toEqual(['any', 'flux', 'sd-1', 'sdxl']);
    expect(collectTypes(library)).toEqual(['main', 'lora', 'vae']);
  });
});
