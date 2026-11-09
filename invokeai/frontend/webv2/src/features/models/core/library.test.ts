import { describe, expect, it } from 'vitest';

import type { ModelConfig } from './types';

import {
  collectBases,
  collectBasesForDisplay,
  collectTypes,
  DEFAULT_LIBRARY_FILTERS,
  filterModels,
  getModelPickerGroups,
  getQualifyingType,
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

const textLlm = createModel({
  base: 'any',
  key: 'llm',
  name: 'Qwen3.5 9B VLM',
  roles: [{ type: 'qwen3_5_encoder', variant: 'qwen3_5_9b' }],
  type: 'text_llm',
});
const roleLibrary = [...library, textLlm];

describe('role-serving models', () => {
  it('filters by role type as well as primary type', () => {
    const keys = (typeFilter: ModelConfig['type']) =>
      filterModels(roleLibrary, { ...DEFAULT_LIBRARY_FILTERS, typeFilter }, NO_MISSING).map((m) => m.key);

    expect(keys('qwen3_5_encoder')).toEqual(['llm']);
    expect(keys('text_llm')).toEqual(['llm']);
    expect(keys('qwen3_encoder')).toEqual([]);
  });

  it('finds a model by its role type in the search', () => {
    expect(
      filterModels(roleLibrary, { ...DEFAULT_LIBRARY_FILTERS, searchTerm: 'qwen3_5_encoder' }, NO_MISSING).map(
        (m) => m.key
      )
    ).toEqual(['llm']);
  });

  it('lists the model in its primary and each role group, and only in the filtered one under a type filter', () => {
    const groups = groupModelsByType(roleLibrary);
    const entriesOf = (type: string) => groups.find((group) => group.type === type)?.entries;

    expect(entriesOf('text_llm')).toEqual([{ model: textLlm, viaType: null }]);
    expect(entriesOf('qwen3_5_encoder')).toEqual([{ model: textLlm, viaType: 'qwen3_5_encoder' }]);

    const filtered = groupModelsByType([textLlm], 'qwen3_5_encoder');

    expect(filtered.map((group) => group.type)).toEqual(['qwen3_5_encoder']);
    // The only row of the model, so it keeps its plain identity.
    expect(filtered[0]?.entries).toEqual([{ model: textLlm, viaType: null }]);
    expect(groupModelsByType([textLlm], 'text_llm')[0]?.entries).toEqual([{ model: textLlm, viaType: null }]);
  });

  it('leads a picker with models of the asked type, then those that only serve as it', () => {
    const encoder = createModel({ base: 'any', key: 'enc', name: 'Zeta Encoder', type: 'qwen3_5_encoder' });
    const roleEncoder = createModel({
      base: 'any',
      key: 'role-enc',
      name: 'Alpha Encoder',
      roles: [{ type: 'text_llm' }],
      type: 'qwen3_5_encoder',
    });
    const realLlm = createModel({ base: 'any', key: 'llm-z', name: 'Zeta LLM', type: 'text_llm' });
    const keysFor = (models: ModelConfig[], modelTypes: ModelConfig['type'][]) =>
      getModelPickerGroups(models, { modelTypes }).groups.flatMap((group) => group.models.map((m) => m.key));

    // Name order alone would put the role candidates first.
    expect(keysFor([roleEncoder, realLlm], ['text_llm'])).toEqual(['llm-z', 'role-enc']);
    expect(keysFor([textLlm, encoder], ['qwen3_5_encoder'])).toEqual(['enc', 'llm']);
    expect(getQualifyingType(textLlm, ['qwen3_5_encoder'])).toBe('qwen3_5_encoder');
    expect(getQualifyingType(textLlm, ['text_llm', 'qwen3_5_encoder'])).toBe('text_llm');
  });

  it('collects role types', () => {
    expect(collectTypes(roleLibrary)).toEqual(expect.arrayContaining(['text_llm', 'qwen3_5_encoder']));
  });

  it('offers a role-serving model to a picker of the role type and nothing else', () => {
    const picker = (modelTypes: ModelConfig['type'][]) =>
      getModelPickerGroups(roleLibrary, { modelTypes }).candidates.map((m) => m.key);

    expect(picker(['qwen3_5_encoder'])).toEqual(['llm']);
    expect(picker(['qwen3_encoder'])).toEqual([]);
    expect(
      getModelPickerGroups(roleLibrary, {
        filter: (model) => model.roles?.[0]?.variant === 'qwen3_5_4b',
        modelTypes: ['qwen3_5_encoder'],
      }).candidates
    ).toEqual([]);
  });
});

describe('groupModelsByType', () => {
  it('groups in canonical category order', () => {
    const groups = groupModelsByType(filterModels(library, DEFAULT_LIBRARY_FILTERS, NO_MISSING));

    expect(groups.map((group) => group.type)).toEqual(['main', 'lora', 'vae']);
    expect(groups.reduce((count, group) => count + group.entries.length, 0)).toBe(library.length);
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

  // Group by base alone so cross-type pickers do not repeat type headings.
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

  it('ignores a base filter this picker does not offer', () => {
    const result = getModelPickerGroups(library, {
      baseFilter: new Set(['sdxl']),
      filter: (model) => model.base === 'sd-1',
      modelTypes: ['main', 'lora', 'vae'],
    });

    expect(result.groups.flatMap((group) => group.models.map((m) => m.key))).toEqual(['c']);
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
