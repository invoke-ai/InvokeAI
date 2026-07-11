import { describe, expect, it } from 'vitest';

import {
  createEmptyCanvasStateV2,
  createNewCanvasStateV2,
  DEFAULT_CANVAS_DOCUMENT_HEIGHT,
  DEFAULT_CANVAS_DOCUMENT_WIDTH,
  migrateCanvasStateToV2,
  placementToTransform,
} from './canvasMigration';
import { createInpaintMaskLayer, nextInpaintMaskName } from './widgets/layers/layerOps';

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

describe('migrateCanvasStateToV2', () => {
  it('round-trips z-image controls without rewriting persisted adapter kinds', () => {
    const adapters = [
      { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: 'sd-control', weight: 0.75 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 't2i_adapter', model: 't2i', weight: 1 },
      { beginEndStepPct: [0, 1], controlMode: null, kind: 'control_lora', model: 'flux-control', weight: 0.75 },
      { beginEndStepPct: [0.2, 0.9], controlMode: null, kind: 'z_image_control', model: 'z-control', weight: 0.7 },
    ] as const;
    const layers = adapters.map((adapter, index) => ({
      adapter,
      blendMode: 'normal',
      id: `control-${index}`,
      isEnabled: true,
      isLocked: false,
      name: `Control ${index}`,
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    }));
    const state = {
      ...createEmptyCanvasStateV2(),
      document: { ...createEmptyCanvasStateV2().document, layers },
    };

    const migrated = migrateCanvasStateToV2(state);

    expect(migrated.document.layers.map((layer) => (layer.type === 'control' ? layer.adapter : null))).toEqual(
      adapters
    );
  });

  it('normalizes an incomplete persisted Z-Image control with backend defaults', () => {
    const state = createEmptyCanvasStateV2();
    const migrated = migrateCanvasStateToV2({
      ...state,
      document: {
        ...state.document,
        layers: [
          {
            adapter: { kind: 'z_image_control', model: 'z-control' },
            blendMode: 'normal',
            id: 'z-control',
            isEnabled: true,
            isLocked: false,
            name: 'Z Control',
            opacity: 1,
            source: { bitmap: null, type: 'paint' },
            transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
            type: 'control',
            withTransparencyEffect: true,
          },
        ],
      },
    });
    const layer = migrated.document.layers[0];

    expect(layer?.type).toBe('control');
    expect(layer?.type === 'control' ? layer.adapter : null).toEqual({
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'z_image_control',
      model: 'z-control',
      weight: 0.75,
    });
  });

  it('normalizes control adapters in both the live document and saved snapshots', () => {
    const state = createEmptyCanvasStateV2();
    const invalidLayer = {
      adapter: { beginEndStepPct: [0.8, 0.2], controlMode: null, kind: 'z_image_control', model: null, weight: -1 },
      blendMode: 'normal',
      id: 'z-control',
      isEnabled: true,
      isLocked: false,
      name: 'Z Control',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const document = { ...state.document, layers: [invalidLayer] };
    const migrated = migrateCanvasStateToV2({
      ...state,
      document,
      snapshots: [{ createdAt: 'now', document, id: 'snapshot', name: 'Snapshot' }],
    });

    const getAdapter = (doc: (typeof migrated)['document']) => {
      const layer = doc.layers[0];
      return layer?.type === 'control' ? layer.adapter : null;
    };
    expect(getAdapter(migrated.document)).toMatchObject({ beginEndStepPct: [0, 1], weight: 0.75 });
    expect(getAdapter(migrated.snapshots[0]!.document)).toMatchObject({ beginEndStepPct: [0, 1], weight: 0.75 });
  });

  it('keeps valid snapshots and discards malformed snapshot entries without blocking the live canvas', () => {
    const state = createEmptyCanvasStateV2();
    const liveLayer = {
      blendMode: 'normal',
      id: 'live',
      isEnabled: true,
      isLocked: false,
      name: 'Live',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const snapshotLayer = {
      adapter: { kind: 'z_image_control', model: 'z-control' },
      ...liveLayer,
      id: 'snapshot-control',
      name: 'Snapshot Control',
      type: 'control',
      withTransparencyEffect: true,
    };
    const validDocument = { ...state.document, layers: [snapshotLayer] };

    const migrated = migrateCanvasStateToV2({
      ...state,
      document: { ...state.document, layers: [liveLayer] },
      snapshots: [
        { createdAt: 'now', document: validDocument, id: 'valid', name: 'Valid' },
        null,
        'malformed',
        {},
        { createdAt: 'now', id: 'missing-document', name: 'Missing document' },
        { createdAt: 'now', document: null, id: 'null-document', name: 'Null document' },
        {
          createdAt: 'now',
          document: { ...state.document, layers: { bad: true } },
          id: 'malformed-layers',
          name: 'Malformed layers',
        },
      ],
    });

    expect(migrated.document.layers.map((layer) => layer.id)).toEqual(['live']);
    expect(migrated.snapshots).toHaveLength(1);
    expect(migrated.snapshots[0]).toMatchObject({ createdAt: 'now', id: 'valid', name: 'Valid' });
    const validLayer = migrated.snapshots[0]!.document.layers[0];
    expect(validLayer?.type === 'control' ? validLayer.adapter : null).toEqual({
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'z_image_control',
      model: 'z-control',
      weight: 0.75,
    });
  });

  it('maps a v1 raster layer to a v2 raster layer positioned by transform', () => {
    const v1Canvas = {
      document: {
        height: 768,
        layers: [
          {
            acceptedAt: '2026-06-09T00:00:00.000Z',
            height: 512,
            id: 'layer-1',
            imageName: 'candidate.png',
            imageUrl: '/api/v1/images/i/candidate.png/full',
            label: 'Layer 1',
            placement: { height: 300, opacity: 0.8, width: 600, x: 12, y: 24 },
            queuedAt: '2026-06-09T00:00:00.000Z',
            sourceQueueItemId: 'queue-1',
            thumbnailUrl: '/api/v1/images/i/candidate.png/thumbnail',
            width: 1024,
          },
        ],
        version: 1,
        width: 1024,
      },
      stagingArea: {
        areThumbnailsVisible: true,
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 1,
    };

    const migrated = migrateCanvasStateToV2(v1Canvas);

    expect(migrated.version).toBe(2);
    expect(migrated.document.layers).toHaveLength(1);

    const layer = migrated.document.layers[0];

    expect(layer).toEqual({
      blendMode: 'normal',
      id: 'layer-1',
      isEnabled: true,
      isLocked: false,
      name: 'Layer 1',
      opacity: 0.8,
      source: { image: { height: 512, imageName: 'candidate.png', width: 1024 }, type: 'image' },
      transform: { rotation: 0, scaleX: 600 / 1024, scaleY: 300 / 512, x: 12, y: 24 },
      type: 'raster',
    });
  });

  it('generates an id and positional name for a layer missing them, and defaults a missing placement', () => {
    const migrated = migrateCanvasStateToV2({
      document: {
        height: 512,
        layers: [{ height: 256, imageName: 'no-id.png', width: 256 }],
        width: 512,
      },
      version: 1,
    });

    const layer = migrated.document.layers[0];

    expect(layer?.id).toBeTruthy();
    expect(layer?.name).toBe('Layer 1');
    expect(layer?.type).toBe('raster');

    if (layer?.type === 'raster' && layer.source.type === 'image') {
      expect(layer.source.image).toEqual({ height: 256, imageName: 'no-id.png', width: 256 });
      expect(layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
      expect(layer.opacity).toBe(1);
    }
  });

  it('migrates a bare imageName-string legacy layer', () => {
    const migrated = migrateCanvasStateToV2({
      document: { height: 512, layers: ['legacy-image.png'], width: 512 },
      version: 1,
    });

    const layer = migrated.document.layers[0];

    expect(layer?.name).toBe('legacy-image.png');

    if (layer?.type === 'raster' && layer.source.type === 'image') {
      expect(layer.source.image.imageName).toBe('legacy-image.png');
    }
  });

  it('supports the pre-`document` legacy shape with layers directly on the canvas', () => {
    const migrated = migrateCanvasStateToV2({
      layers: ['ancient.png'],
      version: 1,
    });

    expect(migrated.document.width).toBe(DEFAULT_CANVAS_DOCUMENT_WIDTH);
    expect(migrated.document.height).toBe(DEFAULT_CANVAS_DOCUMENT_HEIGHT);
    expect(migrated.document.layers).toHaveLength(1);
  });

  it('defaults bbox to the full document rect', () => {
    const migrated = migrateCanvasStateToV2({ document: { height: 600, layers: [], width: 800 }, version: 1 });

    expect(migrated.document.bbox).toEqual({ height: 600, width: 800, x: 0, y: 0 });
  });

  it('defaults selectedLayerId to null (v1 had no selection concept)', () => {
    const migrated = migrateCanvasStateToV2({ document: { height: 512, layers: [], width: 512 }, version: 1 });

    expect(migrated.document.selectedLayerId).toBeNull();
  });

  it('carries the staging area through unchanged and defaults autoSwitchMode to off', () => {
    const pendingImages = [
      {
        height: 512,
        imageName: 'staged.png',
        imageUrl: '/api/v1/images/i/staged.png/full',
        placement: { height: 512, opacity: 1, width: 512, x: 0, y: 0 },
        queuedAt: '2026-06-09T00:00:00.000Z',
        sourceQueueItemId: 'queue-1',
        thumbnailUrl: '/api/v1/images/i/staged.png/thumbnail',
        width: 512,
      },
    ];

    const migrated = migrateCanvasStateToV2({
      document: { height: 512, layers: [], width: 512 },
      stagingArea: {
        areThumbnailsVisible: false,
        isVisible: true,
        pendingImageIds: ['staged.png'],
        pendingImages,
        selectedImageIndex: 0,
        selectedLayerId: 'layer-x',
        sourceQueueItemId: 'queue-1',
      },
      version: 1,
    });

    expect(migrated.stagingArea).toEqual({
      areThumbnailsVisible: false,
      autoSwitchMode: 'off',
      isVisible: true,
      pendingImageIds: ['staged.png'],
      pendingImages,
      selectedImageIndex: 0,
      selectedLayerId: 'layer-x',
      sourceQueueItemId: 'queue-1',
    });
    // The candidate objects themselves are carried through by reference, not reshaped.
    expect(migrated.stagingArea.pendingImages[0]).toBe(pendingImages[0]);
  });

  it('starts with no snapshots', () => {
    const migrated = migrateCanvasStateToV2({ document: { height: 512, layers: [], width: 512 }, version: 1 });

    expect(migrated.snapshots).toEqual([]);
  });

  it('passes already-v2 input through normalized rather than re-migrating it', () => {
    const v2Canvas = {
      document: {
        background: 'transparent',
        bbox: { height: 512, width: 512, x: 0, y: 0 },
        height: 512,
        layers: [
          {
            blendMode: 'multiply',
            id: 'layer-1',
            isEnabled: true,
            isLocked: false,
            name: 'Layer 1',
            opacity: 0.5,
            source: { image: { height: 100, imageName: 'v2.png', width: 100 }, type: 'image' },
            transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
            type: 'raster',
          },
        ],
        selectedLayerId: 'layer-1',
        version: 2,
        width: 512,
      },
      documentRevision: 0,
      snapshots: [],
      stagingArea: {
        areThumbnailsVisible: true,
        autoSwitchMode: 'latest',
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 2,
    };

    const migrated = migrateCanvasStateToV2(v2Canvas);

    expect(migrated).toEqual(v2Canvas);
    // Not double-migrated: blendMode/opacity/selectedLayerId survive untouched, which a v1->v2
    // re-derivation from a (nonexistent) placement would not preserve.
    expect(migrated.document.layers[0]).toEqual(v2Canvas.document.layers[0]);
    expect(migrated.stagingArea.autoSwitchMode).toBe('latest');
  });

  it('preserves the progress canvas staging auto-switch mode', () => {
    const migrated = migrateCanvasStateToV2({
      document: { height: 512, layers: [], width: 512 },
      stagingArea: { autoSwitchMode: 'progress' },
      version: 2,
    });

    expect(migrated.stagingArea.autoSwitchMode).toBe('progress');
  });

  it('normalizes the removed oldest canvas staging auto-switch mode to off', () => {
    const migrated = migrateCanvasStateToV2({
      document: { height: 512, layers: [], width: 512 },
      stagingArea: { autoSwitchMode: 'oldest' },
      version: 2,
    });

    expect(migrated.stagingArea.autoSwitchMode).toBe('off');
  });

  it('round-trips content-sized fields (paint offset, gradient width/height) on an already-v2 doc unchanged', () => {
    const paintOffset = { x: -30, y: 45 };
    const gradientExtent = { height: 220, width: 180 };
    const v2Canvas = {
      document: {
        background: 'transparent',
        bbox: { height: 512, width: 512, x: 0, y: 0 },
        height: 512,
        layers: [
          {
            blendMode: 'normal',
            id: 'paint-1',
            isEnabled: true,
            isLocked: false,
            name: 'Paint 1',
            opacity: 1,
            // Content-sized paint bitmap placed off-origin.
            source: { bitmap: { height: 40, imageName: 'paint.png', width: 40 }, offset: paintOffset, type: 'paint' },
            transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
            type: 'raster',
          },
          {
            blendMode: 'normal',
            id: 'gradient-1',
            isEnabled: true,
            isLocked: false,
            name: 'Gradient 1',
            opacity: 1,
            // Gradient carrying an explicit content extent (not document-sized).
            source: {
              angle: 45,
              height: gradientExtent.height,
              kind: 'linear',
              stops: [
                { color: '#000000', offset: 0 },
                { color: '#ffffff', offset: 1 },
              ],
              type: 'gradient',
              width: gradientExtent.width,
            },
            transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
            type: 'raster',
          },
        ],
        selectedLayerId: 'paint-1',
        version: 2,
        width: 512,
      },
      documentRevision: 3,
      snapshots: [],
      stagingArea: {
        areThumbnailsVisible: true,
        autoSwitchMode: 'off',
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 2,
    };

    const migrated = migrateCanvasStateToV2(v2Canvas);

    // Whole-state round-trip: normalization must not drop or rewrite the
    // content-sizing fields the current schema carries.
    expect(migrated).toEqual(v2Canvas);
    const paintLayer = migrated.document.layers[0];
    const gradientLayer = migrated.document.layers[1];
    if (paintLayer?.type !== 'raster' || gradientLayer?.type !== 'raster') {
      throw new Error('expected two raster layers to survive normalization');
    }
    expect(paintLayer.source).toEqual({
      bitmap: { height: 40, imageName: 'paint.png', width: 40 },
      offset: paintOffset,
      type: 'paint',
    });
    expect(gradientLayer.source).toMatchObject({
      height: gradientExtent.height,
      type: 'gradient',
      width: gradientExtent.width,
    });
  });

  it.each([undefined, null, 'garbage', 42, [], {}])(
    'falls back to a fresh empty v2 state for garbage input (%j)',
    (garbage) => {
      const migrated = migrateCanvasStateToV2(garbage);

      expect(migrated).toEqual(createEmptyCanvasStateV2());
      expect(migrated.document.width).toBe(DEFAULT_CANVAS_DOCUMENT_WIDTH);
      expect(migrated.document.height).toBe(DEFAULT_CANVAS_DOCUMENT_HEIGHT);
      expect(migrated.document.layers).toEqual([]);
      expect(migrated.stagingArea.autoSwitchMode).toBe('off');
    }
  );
});

describe('createEmptyCanvasStateV2', () => {
  it('creates a well-formed empty v2 canvas at the default document size', () => {
    const state = createEmptyCanvasStateV2();

    expect(state).toEqual({
      document: {
        background: 'transparent',
        bbox: { height: DEFAULT_CANVAS_DOCUMENT_HEIGHT, width: DEFAULT_CANVAS_DOCUMENT_WIDTH, x: 0, y: 0 },
        height: DEFAULT_CANVAS_DOCUMENT_HEIGHT,
        layers: [],
        selectedLayerId: null,
        version: 2,
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
      version: 2,
    });
  });

  it('honors a custom document size', () => {
    const state = createEmptyCanvasStateV2(800, 600);

    expect(state.document.width).toBe(800);
    expect(state.document.height).toBe(600);
    expect(state.document.bbox).toEqual({ height: 600, width: 800, x: 0, y: 0 });
  });
});

describe('createNewCanvasStateV2', () => {
  it('seeds exactly one empty inpaint mask, selected, with legacy-default fill', () => {
    const state = createNewCanvasStateV2();
    const { layers, selectedLayerId } = state.document;

    expect(layers).toHaveLength(1);
    const mask = layers[0];
    expect(mask?.type).toBe('inpaint_mask');
    expect(mask?.name).toBe('Inpaint Mask 1');
    expect(mask?.isEnabled).toBe(true);
    expect(mask?.isLocked).toBe(false);
    expect(mask?.opacity).toBe(1);
    // Empty = no bitmap (no strokes); the diagonal-hatch fill is the legacy default.
    expect(mask && 'mask' in mask ? mask.mask : null).toEqual({
      bitmap: null,
      fill: { color: '#e07575', style: 'diagonal' },
    });
    // The seeded mask is the initially selected layer.
    expect(selectedLayerId).toBe(mask?.id);
  });

  it('matches the layers-panel inpaint-mask factory shape (kept in lockstep)', () => {
    const state = createNewCanvasStateV2();
    const mask = state.document.layers[0];
    // Rebuild the expected layer via the canonical factory, pinning the minted id
    // and name so only the contract shape is compared.
    const expected = createInpaintMaskLayer(nextInpaintMaskName([]), mask?.id ?? '');

    expect(mask).toEqual(expected);
  });

  it('honors a custom document size while staying otherwise identical to an empty canvas', () => {
    const state = createNewCanvasStateV2(800, 600);

    expect(state.document.width).toBe(800);
    expect(state.document.height).toBe(600);
    expect(state.document.bbox).toEqual({ height: 600, width: 800, x: 0, y: 0 });
    // Only the document's layers/selection differ from the empty canvas.
    expect({ ...state, document: { ...state.document, layers: [], selectedLayerId: null } }).toEqual(
      createEmptyCanvasStateV2(800, 600)
    );
  });

  it('does not disturb the migration/empty path (garbage still migrates to an empty, mask-free canvas)', () => {
    // The new-canvas seed is scoped to fresh projects only; migrating unknown
    // input keeps producing an empty, mask-free document.
    expect(migrateCanvasStateToV2('garbage').document.layers).toEqual([]);
    expect(createEmptyCanvasStateV2().document.layers).toEqual([]);
  });
});
