import { describe, expect, it } from 'vitest';

import type { StarterModel } from './types';

import { DEFAULT_STARTER_MODEL_FILTERS, filterStarterModels, findBundlePreviewDecoder } from './starters';

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
  it('sorts by name, ignoring case, unless the user picks another order', () => {
    const mixedCase = [...catalog, createStarter({ name: 'bright LoRA', type: 'lora' })];

    expect(filterStarterModels(mixedCase, DEFAULT_STARTER_MODEL_FILTERS, '').map((m) => m.name)).toEqual([
      'Alpha FLUX',
      'bright LoRA',
      'Detail Tweaker',
      'Juggernaut XL',
    ]);
  });

  it('keeps catalog order under the catalog sort', () => {
    const catalogOrder = { ...DEFAULT_STARTER_MODEL_FILTERS, sortField: 'default' as const };

    expect(filterStarterModels(catalog, catalogOrder, '').map((m) => m.name)).toEqual([
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

  it('shows a model under the other bases it serves, as well as its own', () => {
    const withDecoder = [
      ...catalog,
      createStarter({ base: 'flux', compatible_bases: ['z-image'], name: 'TAEF1', type: 'tiny_autoencoder' }),
      createStarter({ base: 'z-image', name: 'Z-Image Turbo', type: 'main' }),
    ];
    const byBase = (baseFilter: string) =>
      filterStarterModels(withDecoder, { ...DEFAULT_STARTER_MODEL_FILTERS, baseFilter }, '').map((m) => m.name);

    expect(byBase('z-image')).toEqual(['TAEF1', 'Z-Image Turbo']);
    expect(byBase('flux')).toEqual(['Alpha FLUX', 'TAEF1']);
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

describe('findBundlePreviewDecoder', () => {
  const taef1 = createStarter({
    base: 'flux',
    compatible_bases: ['z-image'],
    name: 'TAEF1',
    source: 'taef1',
    type: 'tiny_autoencoder',
  });
  const taesdxl = createStarter({ base: 'sdxl', name: 'TAESDXL', source: 'taesdxl', type: 'tiny_autoencoder' });
  const starters = [taesdxl, taef1];
  const bundle = (...models: StarterModel[]) => ({ models, name: 'Bundle' });
  const encoder = createStarter({ base: 'any', name: 'Encoder', type: 'qwen3_encoder' });

  it('finds the decoder a family shares through its compatible bases', () => {
    expect(findBundlePreviewDecoder(bundle(createStarter({ base: 'z-image' }), encoder), starters)).toEqual({
      decoder: taef1,
      family: 'z-image',
    });
  });

  it('finds the decoder filed under the family itself', () => {
    expect(findBundlePreviewDecoder(bundle(createStarter({ base: 'flux' })), starters)?.decoder).toBe(taef1);
  });

  it('gives nothing once the decoder is installed', () => {
    const installed = [taesdxl, { ...taef1, is_installed: true }];
    expect(findBundlePreviewDecoder(bundle(createStarter({ base: 'flux' })), installed)).toBeNull();
  });

  it('names the next missing decoder when a family has two', () => {
    const taew2_1 = createStarter({ base: 'wan', is_installed: true, name: 'TAEW2.1', type: 'tiny_autoencoder' });
    const taew2_2 = createStarter({ base: 'wan', name: 'TAEW2.2', type: 'tiny_autoencoder' });
    expect(findBundlePreviewDecoder(bundle(createStarter({ base: 'wan' })), [taew2_1, taew2_2])?.decoder).toBe(taew2_2);
  });

  it('gives nothing for a family without a decoder', () => {
    expect(findBundlePreviewDecoder(bundle(createStarter({ base: 'cogview4' })), starters)).toBeNull();
  });

  it('gives nothing when the main models name no single family', () => {
    const mixed = bundle(createStarter({ base: 'flux' }), createStarter({ base: 'sdxl' }));
    expect(findBundlePreviewDecoder(mixed, starters)).toBeNull();
    expect(findBundlePreviewDecoder(bundle(encoder), starters)).toBeNull();
  });
});
