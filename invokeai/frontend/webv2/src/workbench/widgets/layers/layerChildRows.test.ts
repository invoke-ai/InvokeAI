import type { CanvasAdjustmentEntry, RegionalGuidanceReferenceImage } from '@workbench/canvas-engine/api';

import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import {
  getLayerChildItem,
  layerChildDropCommand,
  layerChildRowCommand,
  layerChildRowKey,
  projectLayerChildRows,
} from './layerChildRows';
import { buildLayerStackRows } from './layerTreeRows';

const referenceImage = (id: string, overrides: Partial<RegionalGuidanceReferenceImage> = {}) => ({
  config: {
    beginEndStepPct: [0, 1] as [number, number],
    clipVisionModel: 'ViT-H' as const,
    image: null,
    method: 'full' as const,
    model: null,
    type: 'ip_adapter' as const,
    weight: 1,
  },
  id,
  isEnabled: true,
  ...overrides,
});

const regionalWith = (refs: RegionalGuidanceReferenceImage[], overrides = {}) =>
  layerContract('rg1', 'regional_guidance', { referenceImages: refs, ...overrides });

const rowsOf = (document: ReturnType<typeof documentFrom>, id = 'rg1') => {
  const built = buildLayerStackRows(document.stacks, new Set());
  return built.regional_guidance.rows.find((row) => row.id === id)!;
};

describe('projectLayerChildRows', () => {
  it('projects one row per reference image, in order, with enablement and position facts', () => {
    const withImage = referenceImage('ref2', {
      config: { ...referenceImage('ref2').config, image: { imageName: 'img.png', thumbnailUrl: '/thumb.png' } },
      isEnabled: false,
    } as Partial<RegionalGuidanceReferenceImage>);
    const document = documentFrom([regionalWith([referenceImage('ref1'), withImage])]);
    const rows = projectLayerChildRows(rowsOf(document).vm);
    expect(rows.map((row) => row.itemId)).toEqual(['ref1', 'ref2']);
    expect(rows[0]).toMatchObject({
      depth: 1,
      image: null,
      isEnabled: true,
      key: layerChildRowKey('rg1', 'ref1'),
      kind: 'reference-image',
      layerId: 'rg1',
      parentContributing: true,
      posInSet: 1,
      setSize: 2,
      stack: 'regional_guidance',
    });
    expect(rows[1]).toMatchObject({
      image: { imageName: 'img.png', thumbnailUrl: '/thumb.png' },
      isEnabled: false,
      posInSet: 2,
    });
  });

  it('projects nothing for layers without reference images and for other layer types', () => {
    const document = documentFrom([regionalWith([]), layerContract('r1'), layerContract('m1', 'inpaint_mask')]);
    const built = buildLayerStackRows(document.stacks, new Set());
    for (const stack of Object.values(built)) {
      for (const row of stack.rows) {
        expect(projectLayerChildRows(row.vm)).toEqual([]);
      }
    }
  });

  it('reports a disabled or ancestor-disabled parent as not contributing', () => {
    const document = documentFrom([regionalWith([referenceImage('ref1')], { isEnabled: false })]);
    expect(projectLayerChildRows(rowsOf(document).vm)[0]!.parentContributing).toBe(false);
  });

  it('returns the identical array for an unchanged node', () => {
    const document = documentFrom([regionalWith([referenceImage('ref1')])]);
    const vm = rowsOf(document).vm;
    expect(projectLayerChildRows(vm)).toBe(projectLayerChildRows(vm));
  });
});

describe('layerChildRowCommand', () => {
  const target = { itemId: 'ref1', layerId: 'rg1' };

  it('builds a patch-config toggling one item and leaving the others untouched', () => {
    const document = documentFrom([regionalWith([referenceImage('ref1'), referenceImage('ref2')])]);
    const command = layerChildRowCommand(document, target, { isEnabled: false, type: 'set-enabled' });
    expect(command).toMatchObject({ id: 'rg1', type: 'patch-config' });
    const config = command!.config as { referenceImages: RegionalGuidanceReferenceImage[] };
    const before = command!.before as { referenceImages: RegionalGuidanceReferenceImage[] };
    expect(config.referenceImages.map((ref) => [ref.id, ref.isEnabled])).toEqual([
      ['ref1', false],
      ['ref2', true],
    ]);
    expect(before.referenceImages.map((ref) => ref.isEnabled)).toEqual([true, true]);
    // Untouched items keep identity; the toggled one is replaced, never mutated.
    expect(config.referenceImages[1]).toBe(before.referenceImages[1]);
    expect(before.referenceImages[0]!.isEnabled).toBe(true);
  });

  it('builds a removal that drops exactly the addressed item', () => {
    const document = documentFrom([regionalWith([referenceImage('ref1'), referenceImage('ref2')])]);
    const command = layerChildRowCommand(document, target, { type: 'remove' });
    const config = command!.config as { referenceImages: RegionalGuidanceReferenceImage[] };
    expect(config.referenceImages.map((ref) => ref.id)).toEqual(['ref2']);
  });

  it('returns null for a missing layer, a wrong-typed layer, a missing item, and a no-op toggle', () => {
    const document = documentFrom([regionalWith([referenceImage('ref1')]), layerContract('r1')]);
    expect(layerChildRowCommand(document, { itemId: 'ref1', layerId: 'gone' }, { type: 'remove' })).toBeNull();
    expect(layerChildRowCommand(document, { itemId: 'ref1', layerId: 'r1' }, { type: 'remove' })).toBeNull();
    expect(layerChildRowCommand(document, { itemId: 'gone', layerId: 'rg1' }, { type: 'remove' })).toBeNull();
    expect(layerChildRowCommand(document, target, { isEnabled: true, type: 'set-enabled' })).toBeNull();
  });
});

describe('mask modifier rows', () => {
  const maskWith = (overrides = {}) => layerContract('m1', 'inpaint_mask', overrides);
  const maskRow = (document: ReturnType<typeof documentFrom>) =>
    buildLayerStackRows(document.stacks, new Set()).inpaint_mask.rows.find((row) => row.id === 'm1')!;

  it('projects noise then denoise rows with their values and enablement', () => {
    const document = documentFrom([
      maskWith({ denoise: { isEnabled: false, limit: 0.8 }, noise: { isEnabled: true, level: 0.25 } }),
    ]);
    const rows = projectLayerChildRows(maskRow(document).vm);
    expect(rows.map((row) => [row.kind, row.itemId, row.isEnabled, row.detail, row.posInSet, row.setSize])).toEqual([
      ['mask-noise', 'noise', true, '25%', 1, 2],
      ['mask-denoise', 'denoise', false, '80%', 2, 2],
    ]);
    expect(projectLayerChildRows(maskRow(documentFrom([maskWith()])).vm)).toEqual([]);
  });

  it('toggles and removes a modifier through null-clearing patches', () => {
    const document = documentFrom([maskWith({ noise: { isEnabled: true, level: 0.25 } })]);
    const toggle = layerChildRowCommand(
      document,
      { itemId: 'noise', layerId: 'm1' },
      { isEnabled: false, type: 'set-enabled' }
    );
    expect(toggle).toMatchObject({
      before: { layerType: 'inpaint_mask', noise: { isEnabled: true, level: 0.25 } },
      config: { layerType: 'inpaint_mask', noise: { isEnabled: false, level: 0.25 } },
      id: 'm1',
    });
    const remove = layerChildRowCommand(document, { itemId: 'noise', layerId: 'm1' }, { type: 'remove' });
    expect(remove).toMatchObject({ config: { layerType: 'inpaint_mask', noise: null } });
    expect(layerChildRowCommand(document, { itemId: 'denoise', layerId: 'm1' }, { type: 'remove' })).toBeNull();
    expect(
      layerChildRowCommand(document, { itemId: 'noise', layerId: 'm1' }, { isEnabled: true, type: 'set-enabled' })
    ).toBeNull();
  });

  it('resolves live child items across kinds', () => {
    const document = documentFrom([
      maskWith({ noise: { isEnabled: false, level: 0.5 } }),
      regionalWith([referenceImage('ref1')]),
    ]);
    expect(getLayerChildItem(document, 'm1', 'noise')).toEqual({ isEnabled: false, kind: 'mask-noise' });
    expect(getLayerChildItem(document, 'm1', 'denoise')).toBeNull();
    expect(getLayerChildItem(document, 'rg1', 'ref1')).toEqual({ isEnabled: true, kind: 'reference-image' });
    expect(getLayerChildItem(document, 'rg1', 'gone')).toBeNull();
    expect(getLayerChildItem(document, 'gone', 'noise')).toBeNull();
  });
});

describe('group adjustment child rows', () => {
  const groupWith = (adjustments: CanvasAdjustmentEntry[]) =>
    groupContract('g1', [layerContract('r1')], { adjustments });
  const groupRow = (document: ReturnType<typeof documentFrom>) =>
    buildLayerStackRows(document.stacks, new Set()).raster.rows.find((row) => row.id === 'g1')!;
  const stack = (): CanvasAdjustmentEntry[] => [
    { brightness: 0.2, contrast: 0, id: 'ga1', isEnabled: true, type: 'brightness-contrast' },
    { id: 'ga2', isEnabled: false, type: 'invert' },
  ];

  it("projects a group's adjustment entries exactly like a raster layer's", () => {
    const rows = projectLayerChildRows(groupRow(documentFrom([groupWith(stack())])).vm);
    expect(rows.map((row) => [row.kind, row.itemId, row.isEnabled, row.layerId])).toEqual([
      ['adjustment-brightness-contrast', 'ga1', true, 'g1'],
      ['adjustment-invert', 'ga2', false, 'g1'],
    ]);
  });

  it('emits group-arm patches for toggle, move, rename, and drop reorder', () => {
    const document = documentFrom([groupWith(stack())]);
    const toggled = layerChildRowCommand(
      document,
      { itemId: 'ga2', layerId: 'g1' },
      { isEnabled: true, type: 'set-enabled' }
    );
    expect(toggled).toMatchObject({
      before: { layerType: 'group' },
      config: { layerType: 'group' },
      id: 'g1',
      type: 'patch-config',
    });
    expect((toggled!.config as unknown as { adjustments: CanvasAdjustmentEntry[] }).adjustments[1]!.isEnabled).toBe(
      true
    );

    const renamed = layerChildRowCommand(
      document,
      { itemId: 'ga1', layerId: 'g1' },
      { name: 'Warmth', type: 'rename' }
    );
    expect((renamed!.config as unknown as { adjustments: CanvasAdjustmentEntry[] }).adjustments[0]!.name).toBe(
      'Warmth'
    );

    const dropped = layerChildDropCommand(
      document,
      { itemId: 'ga1', kind: 'adjustment-brightness-contrast', layerId: 'g1' },
      { beforeItemId: null, layerId: 'g1' }
    );
    expect(dropped).toMatchObject({ config: { layerType: 'group' }, id: 'g1' });
    expect(
      (dropped as unknown as { config: { adjustments: CanvasAdjustmentEntry[] } }).config.adjustments.map(
        (entry) => entry.id
      )
    ).toEqual(['ga2', 'ga1']);
  });

  it('resolves group-owned child items', () => {
    const document = documentFrom([groupWith(stack())]);
    expect(getLayerChildItem(document, 'g1', 'ga1')).toEqual({
      isEnabled: true,
      kind: 'adjustment-brightness-contrast',
    });
  });
});

describe('adjustment rows', () => {
  const entries = () => [
    { brightness: 0.1, contrast: 0, id: 'a1', isEnabled: true, type: 'brightness-contrast' as const },
    { id: 'a2', isEnabled: false, saturation: -0.4, type: 'hsl' as const },
    { curves: {}, id: 'a3', isEnabled: true, type: 'curves' as const },
  ];
  const rasterWith = (adjustments: CanvasAdjustmentEntry[] = entries()) =>
    layerContract('r1', 'raster', { adjustments });
  const rasterRow = (document: ReturnType<typeof documentFrom>) =>
    buildLayerStackRows(document.stacks, new Set()).raster.rows.find((row) => row.id === 'r1')!;

  it('projects one row per entry, in stack order, with per-kind identity and details', () => {
    const rows = projectLayerChildRows(rasterRow(documentFrom([rasterWith()])).vm);
    expect(rows.map((row) => [row.kind, row.itemId, row.isEnabled, row.detail])).toEqual([
      ['adjustment-brightness-contrast', 'a1', true, null],
      ['adjustment-hsl', 'a2', false, '-40%'],
      ['adjustment-curves', 'a3', true, null],
    ]);
    expect(rows[1]).toMatchObject({ posInSet: 2, setSize: 3 });
    expect(projectLayerChildRows(rasterRow(documentFrom([rasterWith([])])).vm)).toEqual([]);
  });

  it('projects levels, hue and invert entries with their kinds and details', () => {
    const rows = projectLayerChildRows(
      rasterRow(
        documentFrom([
          rasterWith([
            {
              gamma: 1.2,
              id: 'l1',
              inBlack: 10,
              inWhite: 240,
              isEnabled: true,
              outBlack: 0,
              outWhite: 255,
              type: 'levels',
            },
            { id: 'h1', isEnabled: true, rotation: 90, type: 'hue' },
            { id: 'i1', isEnabled: false, type: 'invert' },
            { id: 'e1', isEnabled: true, stops: 1.5, type: 'exposure' },
            {
              channel: 'r',
              gamma: 1,
              id: 'l2',
              inBlack: 0,
              inWhite: 255,
              isEnabled: true,
              outBlack: 0,
              outWhite: 200,
              type: 'levels',
            },
          ]),
        ])
      ).vm
    );
    expect(rows.map((row) => [row.kind, row.itemId, row.isEnabled, row.detail])).toEqual([
      ['adjustment-levels', 'l1', true, null],
      ['adjustment-hue', 'h1', true, '90°'],
      ['adjustment-invert', 'i1', false, null],
      ['adjustment-exposure', 'e1', true, '+1.5 EV'],
      ['adjustment-levels', 'l2', true, 'R'],
    ]);
  });

  it('renames an entry, clears the name with null, and refuses no-op renames and non-adjustment kinds', () => {
    const document = documentFrom([rasterWith(), regionalWith([referenceImage('ref1')])]);
    const named = layerChildRowCommand(document, { itemId: 'a1', layerId: 'r1' }, { name: 'Warm up', type: 'rename' });
    const entriesOf = (command: NonNullable<typeof named>) =>
      (command.config as unknown as { adjustments: CanvasAdjustmentEntry[] }).adjustments;
    expect(entriesOf(named!).map((entry) => entry.name)).toEqual(['Warm up', undefined, undefined]);
    expect(layerChildRowCommand(document, { itemId: 'a1', layerId: 'r1' }, { name: null, type: 'rename' })).toBeNull();
    const renamed = documentFrom([
      rasterWith([{ brightness: 0, contrast: 0, id: 'a1', isEnabled: true, name: 'Old', type: 'brightness-contrast' }]),
    ]);
    const cleared = layerChildRowCommand(renamed, { itemId: 'a1', layerId: 'r1' }, { name: null, type: 'rename' });
    expect('name' in entriesOf(cleared!)[0]!).toBe(false);
    expect(
      layerChildRowCommand(document, { itemId: 'ref1', layerId: 'rg1' }, { name: 'Ref', type: 'rename' })
    ).toBeNull();
  });

  it('projects the custom name of a renamed entry', () => {
    const rows = projectLayerChildRows(
      rasterRow(
        documentFrom([rasterWith([{ id: 'h1', isEnabled: true, name: 'Shift teal', rotation: 30, type: 'hue' }])])
      ).vm
    );
    expect(rows[0]).toMatchObject({ customName: 'Shift teal', detail: '30°' });
  });

  it('moves an entry within the stack and refuses moves past the ends', () => {
    const document = documentFrom([rasterWith()]);
    const moved = layerChildRowCommand(document, { itemId: 'a2', layerId: 'r1' }, { direction: -1, type: 'move' });
    expect(
      (moved!.config as unknown as { adjustments: { id: string }[] }).adjustments.map((entry) => entry.id)
    ).toEqual(['a2', 'a1', 'a3']);
    expect(layerChildRowCommand(document, { itemId: 'a1', layerId: 'r1' }, { direction: -1, type: 'move' })).toBeNull();
    expect(layerChildRowCommand(document, { itemId: 'a3', layerId: 'r1' }, { direction: 1, type: 'move' })).toBeNull();
  });

  it('duplicates an entry directly after itself with the minted id', () => {
    const document = documentFrom([rasterWith()]);
    const command = layerChildRowCommand(
      document,
      { itemId: 'a1', layerId: 'r1' },
      { newId: 'a1-copy', type: 'duplicate' }
    );
    const ids = (command!.config as unknown as { adjustments: { id: string }[] }).adjustments.map((entry) => entry.id);
    expect(ids).toEqual(['a1', 'a1-copy', 'a2', 'a3']);
  });

  it('toggles and removes entries, and refuses move/duplicate for non-adjustment kinds', () => {
    const document = documentFrom([rasterWith(), regionalWith([referenceImage('ref1')])]);
    const toggled = layerChildRowCommand(
      document,
      { itemId: 'a2', layerId: 'r1' },
      { isEnabled: true, type: 'set-enabled' }
    );
    expect((toggled!.config as unknown as { adjustments: { isEnabled: boolean }[] }).adjustments[1]!.isEnabled).toBe(
      true
    );
    const removed = layerChildRowCommand(document, { itemId: 'a2', layerId: 'r1' }, { type: 'remove' });
    expect(
      (removed!.config as unknown as { adjustments: { id: string }[] }).adjustments.map((entry) => entry.id)
    ).toEqual(['a1', 'a3']);
    expect(
      layerChildRowCommand(document, { itemId: 'ref1', layerId: 'rg1' }, { direction: 1, type: 'move' })
    ).toBeNull();
    expect(
      layerChildRowCommand(document, { itemId: 'ref1', layerId: 'rg1' }, { newId: 'x', type: 'duplicate' })
    ).toBeNull();
    expect(getLayerChildItem(document, 'r1', 'a3')).toEqual({ isEnabled: true, kind: 'adjustment-curves' });
  });
});

describe('layerChildDropCommand', () => {
  const entries = () => [
    { brightness: 0.1, contrast: 0, id: 'a1', isEnabled: true, type: 'brightness-contrast' as const },
    { id: 'a2', isEnabled: false, saturation: -0.4, type: 'hsl' as const },
    { curves: {}, id: 'a3', isEnabled: true, type: 'curves' as const },
  ];
  const twoRegions = () => [
    layerContract('rg1', 'regional_guidance', { referenceImages: [referenceImage('ref1'), referenceImage('ref2')] }),
    layerContract('rg2', 'regional_guidance', { referenceImages: [referenceImage('ref3')] }),
    layerContract('r1', 'raster', { adjustments: entries() }),
  ];
  const adjustment = { itemId: 'a3', kind: 'adjustment-curves' as const, layerId: 'r1' };
  const reference = { itemId: 'ref1', kind: 'reference-image' as const, layerId: 'rg1' };

  it('reorders an adjustment within its layer and refuses cross-layer or no-op landings', () => {
    const document = documentFrom(twoRegions());
    const moved = layerChildDropCommand(document, adjustment, { beforeItemId: 'a1', layerId: 'r1' });
    expect(
      ((moved as { config: unknown }).config as { adjustments: { id: string }[] }).adjustments.map((e) => e.id)
    ).toEqual(['a3', 'a1', 'a2']);
    expect(layerChildDropCommand(document, adjustment, { beforeItemId: null, layerId: 'r1' })).toBeNull();
    expect(layerChildDropCommand(document, adjustment, { beforeItemId: 'a3', layerId: 'r1' })).toBeNull();
    expect(layerChildDropCommand(document, adjustment, { beforeItemId: null, layerId: 'rg1' })).toBeNull();
  });

  it('reorders a reference image within its layer', () => {
    const document = documentFrom(twoRegions());
    const moved = layerChildDropCommand(document, reference, { beforeItemId: null, layerId: 'rg1' });
    expect(
      ((moved as { config: unknown }).config as { referenceImages: { id: string }[] }).referenceImages.map((r) => r.id)
    ).toEqual(['ref2', 'ref1']);
  });

  it('moves a reference image between regional layers as one atomic batch', () => {
    const document = documentFrom(twoRegions());
    const command = layerChildDropCommand(document, reference, { beforeItemId: 'ref3', layerId: 'rg2' });
    expect(command).toMatchObject({ type: 'patch-config-batch' });
    const patches = (command as unknown as { patches: { id: string; config: { referenceImages: { id: string }[] } }[] })
      .patches;
    expect(patches.map((patch) => [patch.id, patch.config.referenceImages.map((r) => r.id)])).toEqual([
      ['rg1', ['ref2']],
      ['rg2', ['ref1', 'ref3']],
    ]);
    expect(layerChildDropCommand(document, reference, { beforeItemId: null, layerId: 'r1' })).toBeNull();
    expect(
      layerChildDropCommand(document, { ...reference, itemId: 'gone' }, { beforeItemId: null, layerId: 'rg2' })
    ).toBeNull();
  });
});

describe('layerChildDropCommand id collisions', () => {
  it('refuses a cross-layer move whose destination already holds the item id', () => {
    const document = documentFrom([
      layerContract('rg1', 'regional_guidance', { referenceImages: [referenceImage('ref1')] }),
      layerContract('rg2', 'regional_guidance', { referenceImages: [referenceImage('ref1')] }),
    ]);
    expect(
      layerChildDropCommand(
        document,
        { itemId: 'ref1', kind: 'reference-image', layerId: 'rg1' },
        { beforeItemId: null, layerId: 'rg2' }
      )
    ).toBeNull();
  });
});

describe('layer regenerate region child row', () => {
  const region = (overrides = {}) => ({
    isEnabled: true,
    fill: { color: '#e07575', style: 'diagonal' as const },
    ...overrides,
  });
  const rasterWithRegion = (overrides = {}) => layerContract('r1', 'raster', { inpaint: region(overrides) });
  const row = (document: ReturnType<typeof documentFrom>) =>
    buildLayerStackRows(document.stacks, new Set()).raster.rows.find((r) => r.id === 'r1')!;

  it('projects the region above the adjustment rows with a combined set', () => {
    const document = documentFrom([
      layerContract('r1', 'raster', {
        adjustments: [{ id: 'a1', isEnabled: true, type: 'invert' }],
        inpaint: region({ name: 'Face' }),
      }),
    ]);
    const rows = projectLayerChildRows(row(document).vm);
    expect(rows.map((r) => [r.kind, r.itemId, r.posInSet, r.setSize, r.customName])).toEqual([
      ['layer-region', 'inpaint', 1, 2, 'Face'],
      ['adjustment-invert', 'a1', 2, 2, null],
    ]);
    expect(rows[0]?.orderedPosInSet).toBeUndefined();
    expect(rows[1]).toMatchObject({ orderedPosInSet: 1, orderedSetSize: 1 });
  });

  it('refuses an adjustment drop landing above the region row', () => {
    const document = documentFrom([
      layerContract('r1', 'raster', {
        adjustments: [
          { id: 'a1', isEnabled: true, type: 'invert' },
          { id: 'a2', isEnabled: true, type: 'invert' },
        ],
        inpaint: region(),
      }),
    ]);
    const child = { itemId: 'a2', kind: 'adjustment-invert' as const, layerId: 'r1' };
    expect(layerChildDropCommand(document, child, { beforeItemId: 'inpaint', layerId: 'r1' })).toBeNull();
    const landed = layerChildDropCommand(document, child, { beforeItemId: 'a1', layerId: 'r1' });
    expect(landed).toMatchObject({ config: { adjustments: [{ id: 'a2' }, { id: 'a1' }] } });
  });

  it('routes a persisted adjustment whose id is "inpaint" to the adjustments when no region exists', () => {
    const document = documentFrom([
      layerContract('r1', 'raster', { adjustments: [{ id: 'inpaint', isEnabled: true, type: 'invert' }] }),
    ]);
    expect(getLayerChildItem(document, 'r1', 'inpaint')).toEqual({ isEnabled: true, kind: 'adjustment-invert' });
    const toggled = layerChildRowCommand(
      document,
      { itemId: 'inpaint', layerId: 'r1' },
      {
        isEnabled: false,
        type: 'set-enabled',
      }
    );
    expect(toggled).toMatchObject({ config: { adjustments: [{ id: 'inpaint', isEnabled: false }] } });
  });

  it('hides a foreign "inpaint"-id adjustment row while a region exists', () => {
    const document = documentFrom([
      layerContract('r1', 'raster', {
        adjustments: [
          { id: 'inpaint', isEnabled: true, type: 'invert' },
          { id: 'a1', isEnabled: true, type: 'invert' },
        ],
        inpaint: region(),
      }),
    ]);
    const rows = projectLayerChildRows(row(document).vm);
    expect(rows.map((r) => [r.kind, r.itemId, r.posInSet, r.setSize])).toEqual([
      ['layer-region', 'inpaint', 1, 2],
      ['adjustment-invert', 'a1', 2, 2],
    ]);
  });

  it('resolves, toggles, renames, and null-removes the singleton; move and duplicate refuse', () => {
    const document = documentFrom([rasterWithRegion()]);
    expect(getLayerChildItem(document, 'r1', 'inpaint')).toEqual({ isEnabled: true, kind: 'layer-region' });

    const target = { itemId: 'inpaint', layerId: 'r1' };
    const toggled = layerChildRowCommand(document, target, { isEnabled: false, type: 'set-enabled' });
    expect(toggled).toMatchObject({ config: { inpaint: { isEnabled: false }, layerType: 'raster' }, id: 'r1' });

    const renamed = layerChildRowCommand(document, target, { name: 'Sky', type: 'rename' });
    expect(renamed).toMatchObject({ config: { inpaint: { name: 'Sky' } } });

    const removed = layerChildRowCommand(document, target, { type: 'remove' });
    expect(removed).toMatchObject({ config: { inpaint: null, layerType: 'raster' } });

    expect(layerChildRowCommand(document, target, { direction: 1, type: 'move' })).toBeNull();
    expect(layerChildRowCommand(document, target, { newId: 'x', type: 'duplicate' })).toBeNull();
  });
});
