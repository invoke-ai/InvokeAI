import type { CanvasStateContractV3 } from '@workbench/canvas-engine/api';

import { createControlLayer, createVectorLayer } from '@workbench/canvas-engine/api';
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
  writableCanvasSchemaVersion,
} from './canvasMigration';
import { MAX_SUPPORTED_CANVAS_SCHEMA_VERSION } from './canvasSchemaVersion';
import {
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

const FUTURE_CANVAS_SCHEMA_VERSION = MAX_SUPPORTED_CANVAS_SCHEMA_VERSION + 1;

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
});

describe('loadCanvasState', () => {
  it('round-trips z-image controls without rewriting persisted adapter kinds', () => {
    const adapters = [
      { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: 'sd-control', weight: 0.75 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 't2i_adapter', model: 't2i', weight: 1 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 'control_lora', model: 'flux-control', weight: 0.75 },
      { beginEndStepPct: [0.2, 0.9], controlMode: null, kind: 'z_image_control', model: 'z-control', weight: 0.7 },
      { beginEndStepPct: [0, 0.8], controlMode: null, kind: 'anima_lllite', model: 'anima-sketch', weight: 1.2 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 'qwen_image_control', model: 'qwen-union', weight: 0.85 },
      { beginEndStepPct: [0.1, 0.9], controlMode: null, kind: 'flux2_control', model: 'flux2-union', weight: 0 },
    ] as const;
    const layers = adapters.map((adapter, index) => ({
      ...createControlLayer({ id: `control-${index}`, name: `Control ${index}` }),
      adapter,
    }));

    const loaded = load(withNodes(layers, 'control'));

    expect(loaded.document.stacks.control.map((layer) => (layer.type === 'control' ? layer.adapter : null))).toEqual(
      adapters
    );
  });

  it('loads an Anima control layer saved before LLLite support exactly as it was saved', () => {
    // Earlier builds offered only ControlNet on Anima, listing LLLite models under it and warning that the kind was
    // unsupported. Load cannot tell an LLLite key from a ControlNet key without the catalog, so it keeps the kind; the
    // settings panel reports it and switching to LLLite keeps the model.
    const saved = {
      ...createControlLayer({ id: 'anima-control', name: 'Anima Control' }),
      adapter: {
        beginEndStepPct: [0, 0.75],
        controlMode: 'balanced',
        kind: 'controlnet',
        model: 'anima-sketch',
        weight: 0.75,
      },
    };

    const loaded = load(withNodes([saved], 'control'));

    expect(loaded.document.stacks.control).toEqual([saved]);
  });

  it('refuses a control layer whose adapter kind this build does not know, naming the layer', () => {
    const future = {
      ...createControlLayer({ id: 'future-control', name: 'Future Control' }),
      adapter: { beginEndStepPct: [0, 1], controlMode: null, kind: 'future_adapter', model: 'm', weight: 1 },
    };

    const refused = refusal(withNodes([createControlLayer({ id: 'kept', name: 'Kept' }), future], 'control'));

    expect(refused.status).toBe('invalid');
    expect(refused.status === 'invalid' ? refused.diagnostics : []).toEqual([
      expect.objectContaining({ path: 'document.stacks.control[1]' }),
    ]);
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
      {
        amount: 0.1,
        distribution: 'uniform',
        grain: 'chroma',
        id: 'a9',
        isEnabled: true,
        response: 'midtones',
        seed: 3_000_000_000,
        size: 2.5,
        type: 'noise',
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

  it('refuses a state whose snapshot entries are malformed instead of discarding them', () => {
    const state = createEmptyCanvasState();
    const validDocument = {
      ...state.document,
      stacks: stacksFrom([createControlLayer({ id: 'snapshot-control', name: 'Snapshot Control' })]),
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

  it('round-trips valid snapshots containing every layer type and nested groups', () => {
    const state = createEmptyCanvasState();
    const stacks = {
      control: [groupContract('cg', [createControlLayer({ id: 'control', name: 'Control' })], { isHidden: true })],
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

  it.each([
    [
      'a future outer version',
      { ...createEmptyCanvasState(), version: FUTURE_CANVAS_SCHEMA_VERSION },
      'state',
      FUTURE_CANVAS_SCHEMA_VERSION,
    ],
    ['an older outer version', { ...createEmptyCanvasState(), version: 2 }, 'state', 2],
    ['a legacy version', { ...createEmptyCanvasState(), version: 1 }, 'state', 1],
    [
      'a nested document of another version',
      { ...createEmptyCanvasState(), document: { ...createEmptyCanvasDocument(), version: 2 } },
      'document',
      2,
    ],
    [
      'a nested document of a future version',
      {
        ...createEmptyCanvasState(),
        document: { ...createEmptyCanvasDocument(), version: FUTURE_CANVAS_SCHEMA_VERSION },
      },
      'document',
      FUTURE_CANVAS_SCHEMA_VERSION,
    ],
    [
      'a snapshot of another version',
      {
        ...createEmptyCanvasState(),
        snapshots: [
          {
            createdAt: 'now',
            document: { ...createEmptyCanvasDocument(), version: FUTURE_CANVAS_SCHEMA_VERSION },
            id: 'f',
            name: 'F',
          },
        ],
      },
      'snapshot',
      FUTURE_CANVAS_SCHEMA_VERSION,
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

  describe('noise adjustments adopt schema 5', () => {
    const noise = (isEnabled: boolean) => ({
      amount: 0.2,
      distribution: 'gaussian',
      grain: 'color',
      id: `noise-${String(isEnabled)}`,
      isEnabled,
      response: 'flat',
      seed: 7,
      size: 1,
      type: 'noise',
    });
    const noisyLayer = (isEnabled = true) => ({
      ...createEmptyPaintLayer('Grain', 'grain'),
      adjustments: [noise(isEnabled)],
    });
    const declaring = (canvas: ReturnType<typeof withNodes>, version: number) => ({
      ...canvas,
      document: { ...canvas.document, version },
      version,
    });

    it.each([
      ['an enabled noise entry on a layer', [noisyLayer()]],
      ['a disabled noise entry on a layer', [noisyLayer(false)]],
      [
        'a noise entry on a layer nested in groups',
        [groupContract('outer', [groupContract('inner', [noisyLayer() as never])])],
      ],
      ['a noise entry on a group', [groupContract('graded', [], { adjustments: [noise(true) as never] })]],
    ])('writes a v3 or v4 canvas holding %s as version 5', (_label, nodes) => {
      for (const declared of [3, 4]) {
        const loaded = load(declaring(withNodes(nodes), declared));

        expect(loaded.version).toBe(5);
        expect(loaded.document.version).toBe(5);
      }
    });
  });

  describe('spatial adjustments adopt schema 5', () => {
    const spatialEntries = (isEnabled: boolean) => [
      { amount: 40, id: `clarity-${String(isEnabled)}`, isEnabled, size: 40, type: 'clarity' },
      { amount: -25, id: `texture-${String(isEnabled)}`, isEnabled, size: 8, type: 'texture' },
      { amount: 0, id: `dehaze-${String(isEnabled)}`, isEnabled, type: 'dehaze' },
    ];
    const cases = [true, false].flatMap((isEnabled) =>
      spatialEntries(isEnabled).map(
        (entry) => [`${entry.type} (${isEnabled ? 'enabled' : 'disabled'})`, entry] as const
      )
    );

    it.each(cases)('writes a v3 or v4 canvas holding %s on a layer or a group as version 5', (_label, entry) => {
      const onLayer = [{ ...createEmptyPaintLayer('Detail', 'detail'), adjustments: [entry] }];
      const onGroup = [groupContract('graded', [], { adjustments: [entry as never] })];
      for (const nodes of [onLayer, onGroup]) {
        for (const declared of [3, 4]) {
          const canvas = withNodes(nodes);
          const loaded = load({ ...canvas, document: { ...canvas.document, version: declared }, version: declared });

          expect(loaded.version).toBe(5);
          expect(loaded.document.version).toBe(5);
        }
      }
    });
  });

  describe('control layer derivation fields', () => {
    const image = (imageName: string) => ({ height: 64, imageName, width: 64 });
    const detection = { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' };
    const typed = {
      ...createControlLayer({ id: 'edges', name: 'Edges 1' }),
      controlType: 'edges',
      detection,
      filter: detection,
      original: { image: image('photo.png'), type: 'image' },
      originalTransform: { rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 20 },
      source: { bitmap: image('map.png'), offset: { x: 0, y: 0 }, type: 'paint' },
    };
    const legacy = {
      ...createControlLayer({ id: 'legacy', name: 'Control Layer 1' }),
      filter: detection,
      source: { bitmap: image('map.png'), type: 'paint' },
    };
    const declaring = (canvas: ReturnType<typeof withNodes>, version: number) => ({
      ...canvas,
      document: { ...canvas.document, version },
      version,
    });
    const expectExactly = (actual: unknown, expected: Record<string, unknown>) => {
      expect(actual).toEqual(expected);
      expect(Object.keys(actual as object).sort()).toEqual(Object.keys(expected).sort());
    };

    it('round-trips a typed, detected layer and writes a v3, v4 or v5 canvas as version 6', () => {
      for (const declared of [3, 4, 5]) {
        const loaded = load(declaring(withNodes([typed], 'control'), declared));
        expectExactly(loaded.document.stacks.control[0], typed);
        expect(loaded.version).toBe(6);
        expect(loaded.document.version).toBe(6);
      }
      expect(
        load(withNodes([groupContract('outer', [groupContract('inner', [typed as never])])], 'control')).version
      ).toBe(6);
    });

    it('keeps a legacy control layer unchanged and at its declared version', () => {
      const loaded = load(withNodes([legacy], 'control'));
      expectExactly(loaded.document.stacks.control[0], legacy);
      expect(loaded.version).toBe(3);
      expect(loaded.document.version).toBe(3);
    });

    it('never lowers a declared version 6 once the fields are gone', () => {
      const loaded = load(declaring(withNodes([legacy], 'control'), 6));
      expect(loaded.version).toBe(6);
      expect(loaded.document.version).toBe(6);
    });

    it('writes version 6 when only a snapshot holds a typed layer, leaving the live document at its own version', () => {
      const loaded = load({
        ...createEmptyCanvasState(),
        snapshots: [{ createdAt: 'now', document: withNodes([typed], 'control').document, id: 's', name: 'S' }],
      });
      expect(loaded.version).toBe(6);
      expect(loaded.document.version).toBe(3);
      expect(loaded.snapshots[0]?.document.version).toBe(6);
    });
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
});

describe('createNewCanvasState', () => {
  it('seeds exactly one empty inpaint mask, left unselected so a first stroke paints a new layer', () => {
    const state = createNewCanvasState();
    const mask = state.document.stacks.inpaint_mask[0];

    expect(state.document.stacks.inpaint_mask).toHaveLength(1);
    expect(state.document.stacks.raster).toEqual([]);
    expect(mask).toEqual(createInpaintMaskLayer(nextInpaintMaskName([]), mask?.id ?? ''));
    expect(state.document.selectedLayerId).toBeNull();
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

  it('caps the node count', () => {
    const leaves = Array.from({ length: 10_001 }, (_, index) => createEmptyPaintLayer(`L${index}`, `l${index}`));
    expect(refusal(stateWith(stacksFrom(leaves)))).toMatchObject({
      diagnostics: [{ message: 'document exceeds 10000 nodes' }],
      status: 'invalid',
    });
  });
});

describe('vector layer appearance persistence', () => {
  it('round-trips independent layer appearance in the live document and snapshots with schema 7', () => {
    const first = {
      ...createVectorLayer({ id: 'a', name: 'A' }),
      strokeColor: '#123456',
      strokeWidth: 17,
      fillColor: '#abcdef',
      fillEnabled: true,
      traceTaperEnds: true,
      traceTaper: 240,
    };
    const second = {
      ...createVectorLayer({ id: 'b', name: 'B' }),
      strokeColor: '#ff0000',
      strokeWidth: 3,
      fillColor: '#00ff00',
      strokeEnabled: false,
    };
    const state = createEmptyCanvasState();
    const document = { ...state.document, stacks: stacksFrom([first, second]) };
    const saved = load({ ...state, document, snapshots: [{ id: 'saved', name: 'Saved', createdAt: 'now', document }] });
    expect(saved.version).toBe(7);
    expect(saved.document.version).toBe(7);
    expect(saved.snapshots[0]?.document.version).toBe(7);
    const reopened = load(JSON.parse(JSON.stringify(saved)));
    expect(reopened).toEqual(saved);
    expect(reopened.document.stacks.vector).toEqual([first, second]);
    expect(reopened.snapshots[0]?.document.stacks.vector).toEqual([first, second]);
    // A snapshot alone raises the live canvas floor so an older writer cannot drop its appearance.
    expect(load({ ...state, snapshots: saved.snapshots }).version).toBe(7);
    expect(writableCanvasSchemaVersion([7], { stacks: stacksFrom([]) })).toBe(7);
    expect(writableCanvasSchemaVersion([7], { stacks: { ...stacksFrom([]), vector: [] } })).toBe(7);
  });

  it.each([
    { strokeColor: undefined },
    { strokeWidth: undefined },
    { strokeColor: 'not a color' },
    { strokeWidth: -3 },
    { strokeWidth: 601 },
    { strokeWidth: 1.5 },
  ])('refuses vector layers with missing or invalid stroke appearance: %j', (appearance) => {
    const vector = { ...createVectorLayer({ id: 'v', name: 'Vector' }), ...appearance };
    const state = createEmptyCanvasState();
    expect(
      refusal({ ...state, document: { ...state.document, stacks: { ...state.document.stacks, vector: [vector] } } })
        .status
    ).toBe('invalid');
  });
});
