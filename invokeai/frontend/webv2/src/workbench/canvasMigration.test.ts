import type { CanvasStateContractV3 } from '@workbench/canvas-engine/api';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import type { CanvasLoadResult } from './canvasLoadContracts';

import {
  createEmptyCanvasDocument,
  createEmptyCanvasState,
  createNewCanvasState,
  DEFAULT_CANVAS_DOCUMENT_HEIGHT,
  DEFAULT_CANVAS_DOCUMENT_WIDTH,
  loadCanvasState,
  normalizeCanvasDocumentContract,
  placementToTransform,
} from './canvasMigration';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
  nextInpaintMaskName,
} from './widgets/layers/layerOps';

const load = (raw: unknown): CanvasStateContractV3 => {
  const result = loadCanvasState(raw);
  if (result.status !== 'loaded') {
    throw new Error(`Expected canvas state to load, got ${JSON.stringify(result)}.`);
  }
  return result.value;
};

const refusal = (raw: unknown): Exclude<CanvasLoadResult<CanvasStateContractV3>, { status: 'loaded' }> => {
  const result = loadCanvasState(raw);
  if (result.status === 'loaded') {
    throw new Error('Expected canvas state to be refused.');
  }
  return result;
};

const withNodes = (nodes: unknown[], stack: 'raster' | 'control' | 'regional_guidance' | 'inpaint_mask' = 'raster') => {
  const state = createEmptyCanvasState();
  return { ...state, document: { ...state.document, stacks: { ...state.document.stacks, [stack]: nodes } } };
};

describe('placementToTransform', () => {
  it('maps a placement rect to a scale-based transform relative to the source image size', () => {
    expect(placementToTransform({ height: 200, width: 400, x: 10, y: 20 }, 200, 100)).toEqual({
      rotation: 0,
      scaleX: 2,
      scaleY: 2,
      x: 10,
      y: 20,
    });
  });

  it('falls back to a 1:1 scale when the source image has no dimensions', () => {
    expect(placementToTransform({ height: 200, width: 400, x: 0, y: 0 }, 0, 0)).toEqual({
      rotation: 0,
      scaleX: 1,
      scaleY: 1,
      x: 0,
      y: 0,
    });
  });
});

describe('loadCanvasState', () => {
  it('round-trips z-image controls without rewriting persisted adapter kinds', () => {
    const adapters = [
      { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: 'sd-control', weight: 0.75 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 't2i_adapter', model: 't2i', weight: 1 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 'control_lora', model: 'flux-control', weight: 0.75 },
      { beginEndStepPct: [0.2, 0.9], controlMode: null, kind: 'z_image_control', model: 'z-control', weight: 0.7 },
    ] as const;
    const layers = adapters.map((adapter, index) => ({
      ...createControlLayer(`Control ${index}`, `control-${index}`),
      adapter,
    }));

    const loaded = load(withNodes(layers, 'control'));

    expect(loaded.document.stacks.control.map((layer) => (layer.type === 'control' ? layer.adapter : null))).toEqual(
      adapters
    );
  });

  it('normalizes an incomplete persisted Z-Image control with backend defaults', () => {
    const loaded = load(
      withNodes(
        [{ ...createControlLayer('Z Control', 'z-control'), adapter: { kind: 'z_image_control', model: 'z-control' } }],
        'control'
      )
    );
    const layer = loaded.document.stacks.control[0];

    expect(layer?.type === 'control' ? layer.adapter : null).toEqual({
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'z_image_control',
      model: 'z-control',
      weight: 0.75,
    });
  });

  it('round-trips a layer regenerate region and drops a malformed one without failing the layer', () => {
    const inpaint = {
      fill: { color: '#e07575', style: 'diagonal' },
      isEnabled: true,
      name: 'Face',
    };
    const loaded = load(withNodes([{ ...createEmptyPaintLayer('Region', 'region'), inpaint }]));
    const layer = loaded.document.stacks.raster[0];
    expect(layer?.type === 'raster' ? layer.inpaint : null).toEqual(inpaint);

    const malformed = load(withNodes([{ ...createEmptyPaintLayer('Bad', 'bad'), inpaint: { isEnabled: 'yes' } }]));
    const badLayer = malformed.document.stacks.raster[0];
    expect(badLayer?.type === 'raster' ? badLayer.inpaint : null).toBeUndefined();
  });

  it('round-trips triangle and star shape sources', () => {
    const shape = (kind: string) => ({
      ...createEmptyPaintLayer(kind, kind),
      source: { fill: '#ff0000', height: 10, kind, stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
    });
    const loaded = load(withNodes([shape('triangle'), shape('star')]));
    const kinds = loaded.document.stacks.raster.map((layer) =>
      layer.type === 'raster' && layer.source.type === 'shape' ? layer.source.kind : null
    );
    expect(kinds).toEqual(['triangle', 'star']);
  });

  it('round-trips a placed gradient without rounding its anchor, and refuses a zero span', () => {
    const gradient = (span: number) => ({
      ...createEmptyPaintLayer('Gradient', 'gradient'),
      source: {
        angle: 30,
        center: { x: 12.5, y: 7 },
        height: 64,
        kind: 'radial',
        span,
        stops: [
          { color: '#000000ff', offset: 0 },
          { color: '#00000000', offset: 1 },
        ],
        type: 'gradient',
        width: 64,
      },
    });
    const loaded = load(withNodes([gradient(40)])).document.stacks.raster[0];
    expect(loaded?.type === 'raster' ? loaded.source : null).toMatchObject({ center: { x: 12.5, y: 7 }, span: 40 });
    expect(refusal(withNodes([gradient(0)]))).toMatchObject({
      diagnostics: [{ message: expect.stringContaining('span'), path: 'document.stacks.raster[0]' }],
      status: 'invalid',
    });
  });

  it('round-trips group adjustments in the raster stack, strips them from overlay groups, and drops a malformed group stack', () => {
    const stack = [{ brightness: 0.1, contrast: 0, id: 'ga1', isEnabled: true, type: 'brightness-contrast' }];
    const rasterGroup = {
      adjustments: stack,
      children: [createEmptyPaintLayer('Inside', 'inside')],
      id: 'rg',
      isEnabled: true,
      isLocked: false,
      name: 'Adjusted',
      type: 'group',
    };
    const rasterLoaded = load(withNodes([rasterGroup]));
    const loadedGroup = rasterLoaded.document.stacks.raster[0];
    expect(loadedGroup?.type === 'group' ? loadedGroup.adjustments : null).toEqual(stack);

    const overlayGroup = { ...rasterGroup, children: [], id: 'og' };
    const overlayLoaded = load(withNodes([overlayGroup], 'control'));
    const loadedOverlay = overlayLoaded.document.stacks.control[0];
    expect(loadedOverlay?.type === 'group' ? Object.hasOwn(loadedOverlay, 'adjustments') : null).toBe(false);

    const malformed = { ...rasterGroup, adjustments: [{ id: 'bad', type: 'nope' }], id: 'mg' };
    const malformedLoaded = load(withNodes([malformed]));
    const loadedMalformed = malformedLoaded.document.stacks.raster[0];
    expect(loadedMalformed?.type === 'group' ? loadedMalformed.adjustments : null).toBeUndefined();
    expect(loadedMalformed?.type === 'group' ? loadedMalformed.children.length : 0).toBe(1);
  });

  it('round-trips color labels on leaves and groups in any stack, and drops malformed values', () => {
    const labelled = load(withNodes([{ ...createEmptyPaintLayer('Tagged', 'tagged'), colorLabel: 'violet' }]));
    const leaf = labelled.document.stacks.raster[0];
    expect(leaf && 'colorLabel' in leaf ? leaf.colorLabel : null).toBe('violet');

    const overlayGroup = {
      children: [],
      colorLabel: 'green',
      id: 'og',
      isEnabled: true,
      isLocked: false,
      name: 'Overlay',
      type: 'group',
    };
    const overlayLoaded = load(withNodes([overlayGroup], 'control'));
    const loadedOverlay = overlayLoaded.document.stacks.control[0];
    expect(loadedOverlay?.type === 'group' ? loadedOverlay.colorLabel : null).toBe('green');

    const malformed = load(withNodes([{ ...createEmptyPaintLayer('Bad', 'bad'), colorLabel: 'magenta' }]));
    const badLeaf = malformed.document.stacks.raster[0];
    expect(badLeaf && 'colorLabel' in badLeaf ? badLeaf.colorLabel : null).toBeUndefined();
  });

  it('round-trips group opacity and blend in the raster stack, strips them from overlay groups, and drops malformed values', () => {
    const rasterGroup = {
      blendMode: 'multiply',
      children: [createEmptyPaintLayer('Inside', 'inside')],
      id: 'rg',
      isEnabled: true,
      isLocked: false,
      name: 'Faded',
      opacity: 0.5,
      type: 'group',
    };
    const rasterLoaded = load(withNodes([rasterGroup]));
    const loadedGroup = rasterLoaded.document.stacks.raster[0];
    expect(loadedGroup?.type === 'group' ? loadedGroup.opacity : null).toBe(0.5);
    expect(loadedGroup?.type === 'group' ? loadedGroup.blendMode : null).toBe('multiply');

    const overlayGroup = { ...rasterGroup, children: [], id: 'og' };
    const overlayLoaded = load(withNodes([overlayGroup], 'control'));
    const loadedOverlay = overlayLoaded.document.stacks.control[0];
    expect(loadedOverlay?.type === 'group' ? Object.hasOwn(loadedOverlay, 'opacity') : null).toBe(false);
    expect(loadedOverlay?.type === 'group' ? Object.hasOwn(loadedOverlay, 'blendMode') : null).toBe(false);

    const malformed = { ...rasterGroup, blendMode: 'plasma', id: 'mg', opacity: 7 };
    const malformedLoaded = load(withNodes([malformed]));
    const loadedMalformed = malformedLoaded.document.stacks.raster[0];
    expect(loadedMalformed?.type === 'group' ? loadedMalformed.opacity : null).toBeUndefined();
    expect(loadedMalformed?.type === 'group' ? loadedMalformed.blendMode : null).toBeUndefined();
    expect(loadedMalformed?.type === 'group' ? loadedMalformed.children.length : 0).toBe(1);
  });

  it('round-trips a valid adjustment stack, drops the pre-stack object shape, and drops a stack with one malformed entry', () => {
    const stack = [
      { brightness: 0.2, contrast: -0.1, id: 'a1', isEnabled: true, type: 'brightness-contrast' },
      { id: 'a2', isEnabled: false, saturation: 0.4, type: 'hsl' },
      {
        curves: {
          r: [
            [0, 10],
            [255, 255],
          ],
        },
        id: 'a3',
        isEnabled: true,
        type: 'curves',
      },
      { gamma: 1.4, id: 'a4', inBlack: 12, inWhite: 240, isEnabled: true, outBlack: 5, outWhite: 250, type: 'levels' },
      { id: 'a5', isEnabled: true, rotation: -45, type: 'hue' },
      { id: 'a6', isEnabled: false, type: 'invert' },
      { id: 'a7', isEnabled: true, stops: -1.5, type: 'exposure' },
      {
        channel: 'g',
        gamma: 1,
        id: 'a8',
        inBlack: 0,
        inWhite: 255,
        isEnabled: true,
        outBlack: 0,
        outWhite: 128,
        type: 'levels',
      },
    ];
    const adjustmentsOf = (loaded: CanvasStateContractV3) => {
      const layer = loaded.document.stacks.raster[0];
      return layer?.type === 'raster' ? layer.adjustments : null;
    };

    const valid = load(withNodes([{ ...createEmptyPaintLayer('Adjusted', 'adjusted'), adjustments: stack }]));
    expect(adjustmentsOf(valid)).toEqual(stack);

    const legacy = load(
      withNodes([
        {
          ...createEmptyPaintLayer('Legacy', 'legacy'),
          adjustments: { brightness: 0.2, contrast: 0, saturation: 0 },
        },
      ])
    );
    expect(adjustmentsOf(legacy)).toBeUndefined();

    // One typo'd entry drops the WHOLE stack rather than failing the document — the accepted blast radius.
    const malformed = load(
      withNodes([
        {
          ...createEmptyPaintLayer('Broken', 'broken'),
          adjustments: [stack[0], { id: 'bad', isEnabled: true, saturation: 'high', type: 'hsl' }],
        },
      ])
    );
    expect(adjustmentsOf(malformed)).toBeUndefined();
    expect(malformed.document.stacks.raster).toHaveLength(1);

    // Levels cross-field invariants are enforced: an inverted input range or non-positive gamma is malformed.
    const invalidLevels = load(
      withNodes([
        {
          ...createEmptyPaintLayer('Inverted', 'inverted'),
          adjustments: [
            {
              gamma: 1,
              id: 'l1',
              inBlack: 200,
              inWhite: 100,
              isEnabled: true,
              outBlack: 0,
              outWhite: 255,
              type: 'levels',
            },
          ],
        },
      ])
    );
    expect(adjustmentsOf(invalidLevels)).toBeUndefined();
  });

  it('normalizes control adapters in both the live document and saved snapshots', () => {
    const invalidLayer = {
      ...createControlLayer('Z Control', 'z-control'),
      adapter: { beginEndStepPct: [0.8, 0.2], controlMode: null, kind: 'z_image_control', model: null, weight: -1 },
    };
    const state = withNodes([invalidLayer], 'control');
    const loaded = load({
      ...state,
      snapshots: [{ createdAt: 'now', document: state.document, id: 'snapshot', name: 'Snapshot' }],
    });

    const getAdapter = (doc: (typeof loaded)['document']) => {
      const layer = doc.stacks.control[0];
      return layer?.type === 'control' ? layer.adapter : null;
    };
    expect(getAdapter(loaded.document)).toMatchObject({ beginEndStepPct: [0, 1], weight: 0.75 });
    expect(getAdapter(loaded.snapshots[0]!.document)).toMatchObject({ beginEndStepPct: [0, 1], weight: 0.75 });
  });

  it('refuses a state whose snapshot entries are malformed instead of discarding them', () => {
    const state = createEmptyCanvasState();
    const validDocument = {
      ...state.document,
      stacks: stacksFrom([createControlLayer('Snapshot Control', 'snapshot-control')]),
    };
    const valid = { createdAt: 'now', document: validDocument, id: 'valid', name: 'Valid' };
    const base = withNodes([createEmptyPaintLayer('Live', 'live')]);

    expect(load({ ...base, snapshots: [valid] }).snapshots).toEqual([valid]);

    for (const malformed of [
      null,
      'malformed',
      {},
      { createdAt: 'now', id: 'missing-document', name: 'Missing document' },
      { createdAt: 'now', document: null, id: 'null-document', name: 'Null document' },
      {
        createdAt: 'now',
        document: { ...state.document, stacks: { raster: { bad: true } } },
        id: 'bad-layers',
        name: 'Bad',
      },
    ]) {
      const raw = { ...base, snapshots: [valid, malformed] };
      expect(refusal(raw)).toMatchObject({ raw, scope: 'snapshot', status: 'invalid' });
    }
  });

  it.each([
    ['missing discriminant and base fields', { id: 'broken' }],
    ['unknown layer type', { ...createEmptyPaintLayer('Mystery', 'mystery'), type: 'mystery' }],
    ['raster with invalid source', { ...createEmptyPaintLayer('Raster', 'raster'), source: null }],
    ['a leaf of another stack', createControlLayer('Control', 'control')],
    ['a group without children', { ...groupContract('g'), children: null }],
    ['a group with a malformed child', groupContract('g', [{ ...createEmptyPaintLayer('Bad', 'bad'), opacity: 4 }])],
  ])('refuses a snapshot containing a malformed node: %s', (_label, malformedNode) => {
    const state = createEmptyCanvasState();
    const raw = {
      ...state,
      snapshots: [
        {
          createdAt: 'now',
          document: { ...state.document, stacks: { raster: [createEmptyPaintLayer('Fine', 'fine'), malformedNode] } },
          id: 'malformed',
          name: 'Malformed',
        },
      ],
    };

    expect(refusal(raw)).toMatchObject({
      diagnostics: [{ path: expect.stringContaining('snapshots[0].document.stacks.raster[1]') }],
      raw,
      scope: 'snapshot',
      status: 'invalid',
    });
  });

  it('round-trips valid snapshots containing every layer type and nested groups', () => {
    const state = createEmptyCanvasState();
    const stacks = {
      control: [groupContract('cg', [createControlLayer('Control', 'control')], { isHidden: true })],
      inpaint_mask: [createInpaintMaskLayer('Mask', 'mask')],
      raster: [groupContract('rg', [createEmptyPaintLayer('Raster', 'raster'), groupContract('inner', [])])],
      regional_guidance: [createRegionalGuidanceLayer('Region', 0, 'region')],
    };
    const snapshot = {
      createdAt: 'now',
      document: { ...state.document, selectedLayerId: 'rg', stacks },
      id: 'valid',
      name: 'Valid',
    };

    const loaded = load({ ...state, document: snapshot.document, snapshots: [snapshot] });

    expect(loaded.document).toEqual(snapshot.document);
    expect(loaded.snapshots).toEqual([snapshot]);
  });

  it('refuses a live document containing an unknown node type instead of dropping it', () => {
    const valid = createEmptyPaintLayer('Valid', 'valid');
    const raw = withNodes([valid, { ...valid, id: 'mystery', type: 'mystery' }]);

    expect(refusal(raw)).toMatchObject({
      diagnostics: [{ message: expect.stringContaining('mystery node is invalid'), path: 'document.stacks.raster[1]' }],
      raw,
      scope: 'document',
      status: 'invalid',
    });
  });

  it('refuses a live document containing a malformed leaf with a diagnostic naming the field', () => {
    expect(refusal(withNodes([{ ...createEmptyPaintLayer('Bad', 'bad'), opacity: 4 }]))).toMatchObject({
      diagnostics: [{ message: expect.stringContaining('opacity'), path: 'document.stacks.raster[0]' }],
      status: 'invalid',
    });
  });

  it.each([
    [
      'a duplicated id across stacks',
      { control: [createControlLayer('C', 'dup')], raster: [createEmptyPaintLayer('R', 'dup')] },
      'document.stacks.control[0].id',
    ],
    [
      'a duplicated id inside a group',
      { raster: [groupContract('g', [createEmptyPaintLayer('A', 'a'), createEmptyPaintLayer('B', 'a')])] },
      'document.stacks.raster[0].children[1].id',
    ],
    [
      'a display-hidden raster group',
      { raster: [groupContract('g', [], { isHidden: true })] },
      'document.stacks.raster[0].isHidden',
    ],
    ['a leaf in the wrong stack', { raster: [createControlLayer('C', 'c')] }, 'document.stacks.raster[0]'],
    [
      'a group nested past the depth limit',
      {
        raster: [
          Array.from({ length: 11 }).reduce<unknown>(
            (child, _, index) => groupContract(`g${index}`, child ? [child as never] : []),
            null
          ),
        ],
      },
      'document.stacks.raster[0]',
    ],
  ])('refuses %s', (_label, stacks, path) => {
    const state = createEmptyCanvasState();
    const raw = { ...state, document: { ...state.document, stacks } };
    expect(refusal(raw)).toMatchObject({
      diagnostics: [{ path: expect.stringContaining(path) }],
      scope: 'document',
      status: 'invalid',
    });
  });

  it('accepts a group nested exactly at the depth limit', () => {
    const nested = Array.from({ length: 10 }).reduce<unknown>(
      (child, _, index) => groupContract(`g${index}`, child ? [child as never] : []),
      null
    );
    expect(load(withNodes([nested])).document.stacks.raster[0]).toEqual(nested);
  });

  it.each([
    ['a future outer version', { ...createEmptyCanvasState(), version: 5 }, 'state', 5],
    ['an older outer version', { ...createEmptyCanvasState(), version: 2 }, 'state', 2],
    ['a legacy version', { ...createEmptyCanvasState(), version: 1 }, 'state', 1],
    [
      'a nested document of another version',
      { ...createEmptyCanvasState(), document: { ...createEmptyCanvasDocument(), version: 2 } },
      'document',
      2,
    ],
    [
      'a snapshot of another version',
      {
        ...createEmptyCanvasState(),
        snapshots: [{ createdAt: 'now', document: { ...createEmptyCanvasDocument(), version: 5 }, id: 'f', name: 'F' }],
      },
      'snapshot',
      5,
    ],
  ])('refuses %s before parsing anything', (_label, raw, scope, version) => {
    expect(refusal(raw)).toEqual({ raw, scope, status: 'unsupported-version', version });
  });

  it.each([
    ['a string', '3'],
    ['a fraction', 2.5],
    ['zero', 0],
    ['a negative number', -1],
    ['null', null],
    ['absent', undefined],
  ])('treats a malformed declared version as invalid rather than guessing: %s', (_label, version) => {
    const raw = { ...createEmptyCanvasState(), version };
    expect(refusal(raw)).toMatchObject({ diagnostics: [{ path: 'version' }], raw, scope: 'state', status: 'invalid' });
  });

  it('treats absent state as empty, fills missing snapshots, and refuses a document without its stacks', () => {
    const empty = createEmptyCanvasState();

    expect(load(undefined)).toEqual(empty);
    expect(load(null)).toEqual(empty);
    expect(refusal({ version: 3 })).toMatchObject({ scope: 'document', status: 'invalid' });
    // A v2-shaped body under a v3 envelope must not load as an empty canvas and then be saved over.
    expect(
      refusal({
        document: { height: 1024, layers: [createEmptyPaintLayer('Raster', 'r')], version: 3, width: 1024 },
        version: 3,
      })
    ).toMatchObject({ diagnostics: [{ path: 'document.stacks' }], scope: 'document', status: 'invalid' });
    expect(load({ ...empty, snapshots: undefined }).snapshots).toEqual([]);
    expect(refusal({ ...empty, snapshots: null })).toMatchObject({ scope: 'state', status: 'invalid' });
    expect(refusal({ ...empty, document: null })).toMatchObject({ scope: 'document', status: 'invalid' });
    expect(refusal({ ...empty, document: { ...empty.document, stacks: [] } })).toMatchObject({
      scope: 'document',
      status: 'invalid',
    });
  });

  it('reads a v3 canvas and keeps the compatibility floor when no v4 typography is present', () => {
    const current = createEmptyCanvasState();
    const legacy = {
      ...current,
      document: { ...current.document, version: 3 as const },
      version: 3 as const,
    };

    expect(load(legacy)).toEqual(legacy);
  });

  it('adopts v4 when a v3 canvas contains custom-font typography', () => {
    const current = createEmptyCanvasState();
    const textLayer = {
      ...createEmptyPaintLayer('Text', 'text'),
      source: {
        align: 'left',
        color: '#fff',
        content: 'hello',
        fontFamily: 'Example',
        fontRef: { contentHash: 'a'.repeat(64), family: 'Example', id: 'font-1', label: 'Example Regular' },
        fontSize: 24,
        fontStyle: 'italic' as const,
        fontVariations: { wght: 650 },
        fontWeight: 650,
        lineHeight: 1.2,
        type: 'text' as const,
      },
    };
    const legacy = {
      ...current,
      document: {
        ...current.document,
        stacks: { ...current.document.stacks, raster: [textLayer] },
        version: 3 as const,
      },
      version: 3 as const,
    };

    const loaded = load(legacy);
    expect(loaded.version).toBe(4);
    expect(loaded.document.version).toBe(4);
    expect(loaded.document.stacks.raster[0]).toMatchObject({ source: textLayer.source });
  });

  it('refuses a custom-font reference whose content hash is not a lowercase SHA-256', () => {
    const current = createEmptyCanvasState();
    const textLayer = {
      ...createEmptyPaintLayer('Text', 'text'),
      source: {
        align: 'left',
        color: '#fff',
        content: 'hello',
        fontFamily: 'Example',
        fontRef: { contentHash: 'hash-a', family: 'Example', id: 'font-1', label: 'Example Regular' },
        fontSize: 24,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text' as const,
      },
    };
    const legacy = {
      ...current,
      document: {
        ...current.document,
        stacks: { ...current.document.stacks, raster: [textLayer] },
        version: 3 as const,
      },
      version: 3 as const,
    };

    expect(refusal(legacy)).toMatchObject({ scope: 'document', status: 'invalid' });
  });

  it('passes a valid state through normalized, keeping every persisted field', () => {
    const canvas = {
      ...withNodes([
        {
          ...createEmptyPaintLayer('Layer 1', 'layer-1'),
          blendMode: 'multiply',
          opacity: 0.5,
          source: { image: { height: 100, imageName: 'v3.png', width: 100 }, type: 'image' },
        },
      ]),
      documentRevision: 4,
      stagingArea: { ...createEmptyCanvasState().stagingArea, autoSwitchMode: 'latest' },
    };
    canvas.document.selectedLayerId = 'layer-1';

    expect(load(canvas)).toEqual(canvas);
  });

  it('normalizes persisted geometry to whole pixels in documents and snapshots', () => {
    const state = createEmptyCanvasState();
    const document = {
      ...state.document,
      bbox: { height: 12.6, width: 12.4, x: -3.4, y: -3.6 },
      height: 511.6,
      width: 512.4,
    };
    const loaded = load({ ...state, document, snapshots: [{ createdAt: 'now', document, id: 's', name: 'S' }] });
    const expected = { bbox: { height: 13, width: 12, x: -3, y: -4 }, height: 512, width: 512 };
    expect(loaded.document).toMatchObject(expected);
    expect(loaded.snapshots[0]!.document).toMatchObject(expected);
  });

  it('normalizes the removed oldest staging auto-switch mode to off and keeps progress', () => {
    const state = createEmptyCanvasState();
    expect(
      load({ ...state, stagingArea: { ...state.stagingArea, autoSwitchMode: 'oldest' } }).stagingArea.autoSwitchMode
    ).toBe('off');
    expect(
      load({ ...state, stagingArea: { ...state.stagingArea, autoSwitchMode: 'progress' } }).stagingArea.autoSwitchMode
    ).toBe('progress');
  });

  it('refuses garbage input as invalid state and preserves the raw payload', () => {
    for (const garbage of [[], 'canvas', 42]) {
      expect(refusal(garbage)).toMatchObject({ raw: garbage, scope: 'state', status: 'invalid' });
    }
  });
});

describe('normalizeCanvasDocumentContract', () => {
  it('re-validates an in-memory document and refuses one holding an invalid node', () => {
    const state = createEmptyCanvasState();
    const document = { ...state.document, stacks: stacksFrom([createEmptyPaintLayer('A', 'a')]) };
    expect(normalizeCanvasDocumentContract(document)).toEqual(document);
    expect(
      normalizeCanvasDocumentContract({ ...document, stacks: { ...document.stacks, raster: [{ id: 'x' } as never] } })
    ).toBeNull();
  });
});

describe('createEmptyCanvasState', () => {
  it('creates a well-formed empty canvas at the default document size', () => {
    expect(createEmptyCanvasState()).toEqual({
      document: {
        background: 'transparent',
        bbox: { height: DEFAULT_CANVAS_DOCUMENT_HEIGHT, width: DEFAULT_CANVAS_DOCUMENT_WIDTH, x: 0, y: 0 },
        height: DEFAULT_CANVAS_DOCUMENT_HEIGHT,
        selectedLayerId: null,
        stacks: { control: [], inpaint_mask: [], raster: [], regional_guidance: [] },
        version: 3,
        width: DEFAULT_CANVAS_DOCUMENT_WIDTH,
      },
      documentRevision: 0,
      snapshots: [],
      stagingArea: {
        areThumbnailsVisible: true,
        autoSwitchMode: 'off',
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 3,
    });
  });

  it('honors a custom document size', () => {
    const state = createEmptyCanvasState(800, 600);
    expect(state.document).toMatchObject({ bbox: { height: 600, width: 800, x: 0, y: 0 }, height: 600, width: 800 });
  });
});

describe('createNewCanvasState', () => {
  it('seeds exactly one empty inpaint mask, selected, matching the layers-panel factory', () => {
    const state = createNewCanvasState();
    const mask = state.document.stacks.inpaint_mask[0];

    expect(state.document.stacks.inpaint_mask).toHaveLength(1);
    expect(state.document.stacks.raster).toEqual([]);
    expect(mask).toEqual(createInpaintMaskLayer(nextInpaintMaskName([]), mask?.id ?? ''));
    expect(state.document.selectedLayerId).toBe(mask?.id);
  });

  it('honors a custom document size while staying otherwise identical to an empty canvas', () => {
    const state = createNewCanvasState(800, 600);
    expect({
      ...state,
      document: { ...state.document, selectedLayerId: null, stacks: createEmptyCanvasState().document.stacks },
    }).toEqual(createEmptyCanvasState(800, 600));
  });
});

describe('loadCanvasState ingress repair', () => {
  const stateWith = (stacks: CanvasStateContractV3['document']['stacks'], selectedLayerId: string | null = null) => {
    const state = createEmptyCanvasState();
    return { ...state, document: { ...state.document, selectedLayerId, stacks } };
  };

  it('repairs a dangling selection to a real node and reports it', () => {
    const result = loadCanvasState(stateWith(stacksFrom([createEmptyPaintLayer('A', 'a')]), 'ghost'));
    expect(result.status).toBe('loaded');
    if (result.status !== 'loaded') {
      return;
    }
    expect(result.value.document.selectedLayerId).toBe('a');
    expect(result.diagnostics).toMatchObject([{ path: 'document.selectedLayerId' }]);
    const clean = loadCanvasState(stateWith(stacksFrom([createEmptyPaintLayer('A', 'a')]), 'a'));
    expect(clean.status === 'loaded' && clean.diagnostics).toEqual([]);
  });

  it('refuses a stack it does not know', () => {
    const state = stateWith({ ...stacksFrom([]), extra: [] } as CanvasStateContractV3['document']['stacks']);
    expect(refusal(state)).toMatchObject({
      diagnostics: [{ path: 'document.stacks.extra' }],
      scope: 'document',
      status: 'invalid',
    });
  });

  it('drops display flags the raster stack cannot hold instead of persisting them', () => {
    const leaf = { ...createEmptyPaintLayer('A', 'a'), isHidden: true, children: [] };
    const group = groupContract('g', [], { isHidden: false });
    const loaded = load(stateWith({ ...stacksFrom([]), raster: [leaf, group] }));
    expect(loaded.document.stacks.raster[0]).toEqual(createEmptyPaintLayer('A', 'a'));
    expect(loaded.document.stacks.raster[1]).toEqual(groupContract('g'));
  });

  it('caps the node count', () => {
    const leaves = Array.from({ length: 10_001 }, (_, index) => createEmptyPaintLayer(`L${index}`, `l${index}`));
    expect(refusal(stateWith(stacksFrom(leaves)))).toMatchObject({
      diagnostics: [{ message: 'document exceeds 10000 nodes' }],
      status: 'invalid',
    });
  });

  it('fills the staging area from its defaults, keeping only well-typed fields', () => {
    const state = createEmptyCanvasState();
    const loaded = load({
      ...state,
      stagingArea: { autoSwitchMode: 'bogus', isVisible: false, pendingImages: 'nope' },
    });
    expect(loaded.stagingArea).toEqual({ ...state.stagingArea, isVisible: false });
  });
});
