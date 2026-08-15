import { describe, expect, it } from 'vitest';

import type { ModelConfig } from './types';

import {
  collectBases,
  collectBasesForDisplay,
  collectTypes,
  DEFAULT_LIBRARY_FILTERS,
  filterModels,
  flattenGroupsToRows,
  getModelPickerGroups,
  groupModelsByType,
} from './library';

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

describe('getModelPickerGroups', () => {
  it('filters by allowed type, exclusions, and custom predicate', () => {
    const result = getModelPickerGroups(library, {
      excludeKeys: new Set(['b']),
      filter: (model) => model.base !== 'any',
      modelTypes: ['main', 'lora', 'vae'],
    });

    expect(result.candidates.map((m) => m.key)).toEqual(['a', 'c']);
    expect(result.groups.map((group) => group.base)).toEqual(['sd-1', 'sdxl']);
  });

  // Type is not a grouping axis: a cross-type picker would otherwise repeat a
  // "Main Models" header above every base section.
  it('groups by base alone, in base display order', () => {
    const result = getModelPickerGroups(library, {
      modelTypes: ['main', 'lora', 'vae'],
    });

    expect(result.groups.map((group) => [group.key, group.models.map((m) => m.key)])).toEqual([
      ['sd-1', ['c']],
      ['sdxl', ['a']],
      ['flux', ['b']],
      ['any', ['d']],
    ]);
  });

  it('keeps several types in one base group, ordered by taxonomy rank then name', () => {
    const mixed = [
      createModel({ base: 'sdxl', key: 'lora-1', name: 'Alpha LoRA', type: 'lora' }),
      createModel({ base: 'sdxl', key: 'main-1', name: 'Zulu Main', type: 'main' }),
    ];
    const result = getModelPickerGroups(mixed, { modelTypes: ['main', 'lora'] });

    expect(result.groups).toHaveLength(1);
    expect(result.groups[0]?.models.map((m) => m.key)).toEqual(['main-1', 'lora-1']);
  });

  it('floats models related to the current selection to the top of their group', () => {
    const related = [
      createModel({ base: 'sdxl', key: 'aaa', name: 'Aaa', type: 'lora' }),
      createModel({ base: 'sdxl', key: 'zzz', name: 'Zzz', type: 'lora' }),
    ];
    const result = getModelPickerGroups(related, {
      modelTypes: ['lora'],
      relatedKeys: new Set(['zzz']),
    });

    expect(result.groups[0]?.models.map((m) => m.key)).toEqual(['zzz', 'aaa']);
  });

  it('filters visible models by the selected bases, empty set showing all', () => {
    const onlySdxl = getModelPickerGroups(library, {
      baseFilter: new Set(['sdxl']),
      modelTypes: ['main', 'lora', 'vae'],
    });

    expect(onlySdxl.groups.flatMap((group) => group.models.map((m) => m.key))).toEqual(['a']);

    const all = getModelPickerGroups(library, {
      baseFilter: new Set(),
      modelTypes: ['main', 'lora', 'vae'],
    });

    // Base display order, matching `availableBases`: sd-1, sdxl, flux, any.
    expect(all.groups.flatMap((group) => group.models.map((m) => m.key))).toEqual(['c', 'a', 'b', 'd']);
  });

  it('exposes availableBases that stay stable across base filtering', () => {
    const expected = ['sd-1', 'sdxl', 'flux', 'any'];

    const base = getModelPickerGroups(library, { modelTypes: ['main', 'lora', 'vae'] });
    expect(base.availableBases).toEqual(expected);

    const filtered = getModelPickerGroups(library, {
      baseFilter: new Set(['flux']),
      modelTypes: ['main', 'lora', 'vae'],
    });
    expect(filtered.availableBases).toEqual(expected);
  });

  it('drops bases removed by excludeKeys or the custom predicate from availableBases', () => {
    const result = getModelPickerGroups(library, {
      excludeKeys: new Set(['b']),
      filter: (model) => model.base !== 'any',
      modelTypes: ['main', 'lora', 'vae'],
    });

    expect(result.availableBases).toEqual(['sd-1', 'sdxl']);
  });
});

describe('collect helpers', () => {
  it('collects distinct bases alphabetically and types in category order', () => {
    expect(collectBases(library)).toEqual(['any', 'flux', 'sd-1', 'sdxl']);
    expect(collectTypes(library)).toEqual(['main', 'lora', 'vae']);
  });

  it('orders bases for display by registry order with any/external/unknown last', () => {
    expect(collectBasesForDisplay(library)).toEqual(['sd-1', 'sdxl', 'flux', 'any']);
    expect(
      collectBasesForDisplay([{ base: 'unknown' }, { base: 'flux' }, { base: 'external' }, { base: 'sd-1' }])
    ).toEqual(['sd-1', 'flux', 'external', 'unknown']);
  });
});

describe('sort fields', () => {
  it('sorts by path with direction applied', () => {
    const pathModels = [
      createModel({ key: 'p1', path: '/models/zeta.safetensors' }),
      createModel({ key: 'p2', path: '/loras/alpha.safetensors' }),
      createModel({ key: 'p3', path: '/models/beta.safetensors' }),
    ];

    const asc = filterModels(
      pathModels,
      { ...DEFAULT_LIBRARY_FILTERS, sortDirection: 'asc', sortField: 'path' },
      NO_MISSING
    ).map((m) => m.key);

    expect(asc).toEqual(['p2', 'p3', 'p1']);

    const desc = filterModels(
      pathModels,
      { ...DEFAULT_LIBRARY_FILTERS, sortDirection: 'desc', sortField: 'path' },
      NO_MISSING
    ).map((m) => m.key);

    expect(desc).toEqual(['p1', 'p3', 'p2']);
  });

  it('sorts by type in category-rank order, then name', () => {
    const sorted = filterModels(
      library,
      { ...DEFAULT_LIBRARY_FILTERS, sortDirection: 'asc', sortField: 'type' },
      NO_MISSING
    ).map((m) => m.type);

    expect(sorted).toEqual(['main', 'main', 'lora', 'vae']);
  });
});
