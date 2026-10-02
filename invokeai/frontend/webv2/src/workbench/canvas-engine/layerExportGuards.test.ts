import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  createLayerExportGuards,
  isSupportedExportSource,
  type CreateLayerExportGuardsDeps,
  type LayerExportGuards,
} from './layerExportGuards';

const imageSource = (imageName = 'a.png'): CanvasLayerSourceContract =>
  ({ image: { height: 4, imageName, width: 4 }, type: 'image' }) as CanvasLayerSourceContract;

const layerOf = (source: CanvasLayerSourceContract, id = 'layer-1'): CanvasLayerContract =>
  ({ id, isEnabled: true, isLocked: false, name: id, source, type: 'raster' }) as CanvasLayerContract;

const entryOf = (overrides: Partial<LayerCacheEntry> = {}): LayerCacheEntry =>
  ({
    layerId: 'layer-1',
    rect: { height: 4, width: 4, x: 0, y: 0 },
    stale: false,
    version: 7,
    ...overrides,
  }) as LayerCacheEntry;

/** A cache store backed by one entry, which is all any guard reads. */
const cacheOf = (entry: LayerCacheEntry | undefined): LayerCacheStore =>
  ({
    get: (id: string) => (id === entry?.layerId ? entry : undefined),
    version: (id: string) => (id === entry?.layerId ? entry.version : 0),
  }) as unknown as LayerCacheStore;

let layer: CanvasLayerContract;
let entry: LayerCacheEntry;
let generation: number;
let job: RasterizationJob | undefined;
let disposed: boolean;
let canvasState: boolean;

const build = (overrides: Partial<CreateLayerExportGuardsDeps> = {}): LayerExportGuards =>
  createLayerExportGuards({
    getDocument: () => ({ height: 64, stacks: stacksFrom([layer]), width: 64 }) as CanvasDocumentContractV3,
    getDocumentGeneration: () => generation,
    getRasterizationJob: () => job,
    hasCanvasState: () => canvasState,
    isDisposed: () => disposed,
    layerCache: cacheOf(entry),
    projectId: 'project-1',
    ...overrides,
  });

beforeEach(() => {
  layer = layerOf(imageSource());
  entry = entryOf();
  generation = 3;
  job = undefined;
  disposed = false;
  canvasState = true;
});

describe('isSupportedExportSource', () => {
  it.each([
    ['image', imageSource(), true],
    ['rect shape', { height: 4, kind: 'rect', type: 'shape', width: 4 } as CanvasLayerSourceContract, true],
    ['polygon shape', { kind: 'polygon', points: [], type: 'shape' } as unknown as CanvasLayerSourceContract, false],
  ])('reports %s as %s', (_label, source, expected) => {
    expect(isSupportedExportSource(source)).toBe(expected);
  });
});

describe('isCurrentRasterizationJob', () => {
  const jobFor = (overrides: Partial<RasterizationJob> = {}): RasterizationJob =>
    ({ documentGeneration: 3, source: imageSource(), version: 7, ...overrides }) as RasterizationJob;

  it('is false with no job in flight', () => {
    expect(build().isCurrentRasterizationJob(layer)).toBe(false);
  });

  it('is true when the job matches the layers source, cache version and generation', () => {
    job = jobFor();
    expect(build().isCurrentRasterizationJob(layer)).toBe(true);
  });

  it.each([
    ['the cache version has moved on', () => (job = jobFor({ version: 6 }))],
    ['the document generation has moved on', () => (job = jobFor({ documentGeneration: 2 }))],
    ['the source has been edited since the job started', () => (job = jobFor({ source: imageSource('other.png') }))],
  ])('is false once %s', (_label, arrange) => {
    arrange();
    expect(build().isCurrentRasterizationJob(layer)).toBe(false);
  });
});

describe('isLayerExportGuardCurrent', () => {
  const capture = (guards: LayerExportGuards = build()) => guards.captureLayerExportGuard(layer, entry);

  it('accepts a guard nothing has invalidated', () => {
    const guards = build();
    expect(guards.isLayerExportGuardCurrent(capture(guards))).toBe(true);
  });

  it('stamps the guard from the caller-supplied entry and the live generation', () => {
    expect(capture()).toMatchObject({
      cacheVersion: 7,
      documentGeneration: 3,
      layerId: 'layer-1',
      projectId: 'project-1',
    });
  });

  it.each([
    ['the engine has been disposed', () => (disposed = true)],
    ['the canvas state has been torn down', () => (canvasState = false)],
    ['the document generation has advanced', () => (generation = 4)],
    ['the cache entry has been re-versioned', () => (entry = entryOf({ version: 8 }))],
    ['the cache entry is gone', () => (entry = entryOf({ layerId: 'other' }))],
  ])('rejects a guard once %s', (_label, invalidate) => {
    const guard = capture();
    invalidate();
    expect(build().isLayerExportGuardCurrent(guard)).toBe(false);
  });

  it('rejects a guard whose layer object was replaced, even with identical fields', () => {
    const guard = capture();
    layer = layerOf(imageSource());
    expect(build().isLayerExportGuardCurrent(guard)).toBe(false);
  });

  it('rejects a guard stamped by a different project', () => {
    const guard = capture();
    expect(build({ projectId: 'project-2' }).isLayerExportGuardCurrent(guard)).toBe(false);
  });
});

describe('captureCurrentLayerExportGuard', () => {
  it('stamps a guard for a layer whose cached pixels are trustworthy', () => {
    expect(build().captureCurrentLayerExportGuard('layer-1')).toMatchObject({ cacheVersion: 7, layerId: 'layer-1' });
  });

  it.each([
    ['the layer is not in the document', () => 'missing-layer'],
    [
      'the cache entry is stale',
      () => {
        entry = entryOf({ stale: true });
        return 'layer-1';
      },
    ],
    [
      'the cached content rect is empty',
      () => {
        entry = entryOf({ rect: { height: 0, width: 0, x: 0, y: 0 } });
        return 'layer-1';
      },
    ],
    [
      'a rasterization for the current source is still running',
      () => {
        job = { documentGeneration: 3, source: imageSource(), version: 7 } as RasterizationJob;
        return 'layer-1';
      },
    ],
    [
      'the source has no rasterizer',
      () => {
        layer = layerOf({ kind: 'polygon', type: 'shape' } as unknown as CanvasLayerSourceContract);
        return 'layer-1';
      },
    ],
  ])('refuses when %s', (_label, arrange) => {
    const layerId = arrange();
    expect(build().captureCurrentLayerExportGuard(layerId)).toBeNull();
  });

  it('refuses when the entry exists but the guard it would stamp is already invalid', () => {
    // A torn-down canvas state fails the currency re-check the capture ends with.
    canvasState = false;
    expect(build().captureCurrentLayerExportGuard('layer-1')).toBeNull();
  });
});

describe('hasExportableLayerContent', () => {
  it('is true for a source whose declared content rect is non-empty', () => {
    expect(build().hasExportableLayerContent('layer-1')).toBe(true);
  });

  it('is false for a layer the document does not hold', () => {
    expect(build().hasExportableLayerContent('missing-layer')).toBe(false);
  });

  it('is false for a source with no rasterizer, however much it is cached', () => {
    layer = layerOf({ kind: 'polygon', type: 'shape' } as unknown as CanvasLayerSourceContract);
    expect(build().hasExportableLayerContent('layer-1')).toBe(false);
  });

  describe('a paint layer with nothing persisted', () => {
    beforeEach(() => {
      layer = layerOf({ bitmap: null, type: 'paint' } as unknown as CanvasLayerSourceContract);
    });

    it('is true when the live cache holds unflushed strokes', () => {
      expect(build().hasExportableLayerContent('layer-1')).toBe(true);
    });

    it.each([
      ['the cache is stale', () => (entry = entryOf({ stale: true }))],
      ['the cache is empty', () => (entry = entryOf({ rect: { height: 0, width: 0, x: 0, y: 0 } }))],
      [
        'those pixels are still being rasterized',
        () =>
          (job = { documentGeneration: 3, source: { bitmap: null, type: 'paint' }, version: 7 } as RasterizationJob),
      ],
    ])('is false when %s', (_label, arrange) => {
      arrange();
      expect(build().hasExportableLayerContent('layer-1')).toBe(false);
    });
  });

  it('is false for a non-paint source with an empty content rect, cache notwithstanding', () => {
    // Only paint can hold pixels beyond its persisted rect, so the live cache is
    // not consulted here — this layer has a fresh, non-empty one and still fails.
    layer = layerOf({ image: { height: 0, imageName: 'a.png', width: 0 }, type: 'image' } as CanvasLayerSourceContract);
    expect(entry.stale).toBe(false);
    expect(build().hasExportableLayerContent('layer-1')).toBe(false);
  });
});
