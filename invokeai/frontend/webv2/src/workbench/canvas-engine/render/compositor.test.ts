import type {
  CanvasBlendMode,
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import type { RasterCallLogEntry, StubRasterSurface } from './raster.testStub';

import { compositeDocument, createCheckerboardTile, prepareComposite, shouldSmoothAtZoom } from './compositor';
import { createDerivedSurfaceCache } from './derivedSurfaceCache';
import { createLayerCacheStore } from './layerCache';
import { createTestStubRasterBackend } from './raster.testStub';

const VIEW: Mat2d = identity();

const imageRef = (): CanvasImageRef => ({ height: 10, imageName: 'x', width: 10 });

const rasterLayer = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const maskLayer = (id: string): CanvasLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

const controlLayer = (id: string): CanvasLayerContract => ({
  adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: false,
});

const inpaintMaskLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#00ff00', style: 'solid' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const makeDoc = (
  layers: CanvasLayerContract[],
  overrides: Partial<CanvasDocumentContractV3> = {}
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: 100,
  ...overrides,
});

const findSet = (log: RasterCallLogEntry[], prop: string): unknown[] =>
  log.filter((e) => e.op === 'set' && e.args[0] === prop).map((e) => e.args[1]);

describe('compositeDocument', () => {
  it('clears then draws layer caches bottom-to-top (array index 0 is top-most)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const top = rasterLayer('top');
    const bottom = rasterLayer('bottom');
    const topCache = caches.getOrCreate('top', 10, 10);
    const bottomCache = caches.getOrCreate('bottom', 10, 10);

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([top, bottom]), caches, VIEW, { backend });

    const log = target.callLog;
    expect(log.some((e) => e.op === 'clearRect')).toBe(true);

    const drawImages = log.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(2);
    expect(drawImages[0]!.args[0]).toBe(bottomCache.surface.canvas);
    expect(drawImages[1]!.args[0]).toBe(topCache.surface.canvas);
  });

  it('isolates the named layer and suppresses staged content and filter previews', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const isolated = caches.getOrCreate('isolated', 10, 10);
    caches.getOrCreate('other', 10, 10);
    const filterPreview = backend.createSurface(10, 10);
    const staged = backend.createSurface(10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([rasterLayer('isolated'), rasterLayer('other')]), caches, VIEW, {
      backend,
      layerPreviews: new Map([['isolated', { rect: { height: 10, width: 10, x: 0, y: 0 }, surface: filterPreview }]]),
      isolationLayerId: 'isolated',
      stagedPreview: { rect: { height: 10, width: 10, x: 0, y: 0 }, surface: staged },
    });

    const drawImages = target.callLog.filter((entry) => entry.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args[0]).toBe(isolated.surface.canvas);
  });

  it('applies per-layer opacity and maps blend mode to a composite operation', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    const target = backend.createSurface(200, 200);

    const blend: CanvasBlendMode = 'multiply';
    compositeDocument(target, makeDoc([rasterLayer('a', { blendMode: blend, opacity: 0.4 })]), caches, VIEW, {
      backend,
    });

    expect(findSet(target.callLog, 'globalAlpha')).toContain(0.4);
    expect(findSet(target.callLog, 'globalCompositeOperation')).toContain('multiply');
  });

  it('fills the ENTIRE viewport with the checkerboard pattern (unbounded plane)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const tile = createCheckerboardTile(backend);

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([], { background: 'transparent' }), caches, VIEW, {
      backend,
      checkerboardTile: tile,
    });

    const patternCalls = target.callLog.filter((e) => e.op === 'createPattern');
    expect(patternCalls).toHaveLength(1);
    expect(patternCalls[0]!.args[0]).toBe(tile.canvas);
    expect(patternCalls[0]!.args[1]).toBe('repeat');

    const fills = target.callLog.filter((e) => e.op === 'fillRect');
    expect(fills).toHaveLength(1);
    expect(fills[0]!.args).toEqual([0, 0, 200, 200]);
    const styles = findSet(target.callLog, 'fillStyle');
    expect(styles).toHaveLength(1);
    expect(typeof styles[0]).toBe('object');
  });

  it('colorizes the mask alpha via source-in on an intermediate surface', () => {
    const base = createTestStubRasterBackend();
    const created: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        created.push(surface);
        return surface;
      },
    };
    const caches = createLayerCacheStore(backend);
    const maskEntry = caches.getOrCreate('rg', 10, 10) as unknown as { surface: StubRasterSurface };
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([maskLayer('rg')]), caches, VIEW, { backend });

    const colorized = created.find(
      (s) =>
        s !== target &&
        s !== maskEntry.surface &&
        s.callLog.some((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation' && e.args[1] === 'source-in')
    );
    expect(colorized).toBeDefined();
    expect(findSet(colorized!.callLog, 'fillStyle')).toContain('#ff0000');
    expect(target.callLog.some((e) => e.op === 'drawImage')).toBe(true);
  });

  it('performs no effect allocations or pixel readbacks on a warmed unchanged composite', () => {
    const base = createTestStubRasterBackend();
    const created: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        created.push(surface);
        return surface;
      },
    };
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('control', 10, 10);
    caches.getOrCreate('mask', 10, 10);
    const target = backend.createSurface(100, 100);
    const derivedSurfaces = createDerivedSurfaceCache();
    const control = controlLayer('control');
    if (control.type !== 'control') {
      throw new Error('Expected control fixture');
    }
    const doc = makeDoc([{ ...control, withTransparencyEffect: true }, maskLayer('mask')]);

    compositeDocument(target, doc, caches, VIEW, { backend, derivedSurfaces });
    const allocationsAfterWarmup = created.length;
    const readbacksAfterWarmup = created.reduce(
      (count, surface) => count + surface.callLog.filter((entry) => entry.op === 'getImageData').length,
      0
    );

    compositeDocument(target, doc, caches, VIEW, { backend, derivedSurfaces });
    expect(created).toHaveLength(allocationsAfterWarmup);
    expect(
      created.reduce(
        (count, surface) => count + surface.callLog.filter((entry) => entry.op === 'getImageData').length,
        0
      )
    ).toBe(readbacksAfterWarmup);
  });

  it('composites in strict group order: raster < control < regional < inpaint mask, ignoring global index', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('raster', 10, 10);
    caches.getOrCreate('control', 11, 11);
    caches.getOrCreate('regional', 12, 12);
    caches.getOrCreate('inpaint', 13, 13);
    const widthToId: Record<number, string> = { 10: 'raster', 11: 'control', 12: 'regional', 13: 'inpaint' };
    const target = backend.createSurface(200, 200) as StubRasterSurface;

    const doc = makeDoc([
      rasterLayer('raster'),
      inpaintMaskLayer('inpaint'),
      controlLayer('control'),
      maskLayer('regional'),
    ]);
    compositeDocument(target, doc, caches, VIEW, { backend });

    const order = target.callLog
      .filter((e) => e.op === 'drawImage')
      .map((e) => widthToId[(e.args[0] as { width: number }).width])
      .filter((id): id is string => id !== undefined);

    expect(order.indexOf('raster')).toBeLessThan(order.indexOf('control'));
    expect(order.indexOf('control')).toBeLessThan(order.indexOf('regional'));
    expect(order.indexOf('regional')).toBeLessThan(order.indexOf('inpaint'));
  });

  it('draws a staged preview over its bbox when provided', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const target = backend.createSurface(200, 200);
    const staged = backend.createSurface(50, 50);

    compositeDocument(target, makeDoc([]), caches, VIEW, {
      backend,
      stagedPreview: { rect: { height: 40, width: 40, x: 5, y: 5 }, surface: staged },
    });

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args).toEqual([staged.canvas, 5, 5, 40, 40]);
  });

  it('lands the staged preview above raster layers and below control layers, masks and regions', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('raster', 10, 10);
    caches.getOrCreate('control', 11, 11);
    caches.getOrCreate('regional', 12, 12);
    caches.getOrCreate('inpaint', 13, 13);
    const widthToId: Record<number, string> = {
      10: 'raster',
      11: 'control',
      12: 'regional',
      13: 'inpaint',
      14: 'staged',
    };
    const target = backend.createSurface(200, 200) as StubRasterSurface;
    const doc = makeDoc([
      rasterLayer('raster'),
      inpaintMaskLayer('inpaint'),
      controlLayer('control'),
      maskLayer('regional'),
    ]);

    compositeDocument(target, doc, caches, VIEW, {
      backend,
      stagedPreview: { rect: { height: 14, width: 14, x: 0, y: 0 }, surface: backend.createSurface(14, 14) },
    });

    const order = target.callLog
      .filter((e) => e.op === 'drawImage')
      .map((e) => widthToId[(e.args[0] as { width: number }).width])
      .filter((id): id is string => id !== undefined);
    expect(order.indexOf('raster')).toBeLessThan(order.indexOf('staged'));
    expect(order.indexOf('staged')).toBeLessThan(order.indexOf('control'));
    expect(order.indexOf('staged')).toBeLessThan(order.indexOf('regional'));
    expect(order.indexOf('staged')).toBeLessThan(order.indexOf('inpaint'));
  });
});

describe('compositeDocument — raster adjustments', () => {
  it('draws the provided adjusted surface instead of the raw cache for a raster layer', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const layer = rasterLayer('a', {
      adjustments: [
        { brightness: 0.5, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      ],
    });
    const cache = caches.getOrCreate('a', 10, 10);
    const adjusted = backend.createSurface(10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([layer]), caches, VIEW, {
      backend,
      adjustedSurface: (l) => (l.id === 'a' ? adjusted : null),
    });

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args[0]).toBe(adjusted.canvas);
    expect(drawImages[0]!.args[0]).not.toBe(cache.surface.canvas);
  });
});

describe('shouldSmoothAtZoom', () => {
  it('smooths only when the document is down-scaled (zoom < 1)', () => {
    expect(shouldSmoothAtZoom(0.1)).toBe(true);
    expect(shouldSmoothAtZoom(0.5)).toBe(true);
    expect(shouldSmoothAtZoom(0.99)).toBe(true);
    expect(shouldSmoothAtZoom(1)).toBe(false);
    expect(shouldSmoothAtZoom(4)).toBe(false);
    expect(shouldSmoothAtZoom(20)).toBe(false);
  });
});

describe('compositeDocument — floating selection', () => {
  const floatOf = (backend: ReturnType<typeof createTestStubRasterBackend>, layerId: string) => ({
    layerId,
    matrix: identity(),
    rect: { height: 8, width: 8, x: 0, y: 0 },
    surface: backend.createSurface(8, 8),
  });

  it('draws the float immediately above its own layer and below the next one up', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const top = rasterLayer('top');
    const bottom = rasterLayer('bottom');
    const topCache = caches.getOrCreate('top', 10, 10);
    const bottomCache = caches.getOrCreate('bottom', 10, 10);
    const float = floatOf(backend, 'bottom');

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([top, bottom]), caches, VIEW, { backend, floatingSelection: float });

    const drawn = target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    expect(drawn).toEqual([bottomCache.surface.canvas, float.surface.canvas, topCache.surface.canvas]);
  });
});

describe('compositeDocument — hidden layers', () => {
  const hiddenMask = (id: string): CanvasLayerContract => ({
    ...(maskLayer(id) as CanvasRegionalGuidanceLayerContract),
    isHidden: true,
  });

  it('does not draw a hidden overlay layer', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('m', 10, 10);
    const rasterCache = caches.getOrCreate('r', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([hiddenMask('m'), rasterLayer('r')]), caches, VIEW, { backend });

    const drawn = target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    expect(drawn).toEqual([rasterCache.surface.canvas]);
  });
});

describe('compositeDocument — prepared plans', () => {
  const scene = () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const top = caches.getOrCreate('top', 10, 10);
    const bottom = caches.getOrCreate('bottom', 10, 10);
    const target = backend.createSurface(50, 50);
    const drawn = (): unknown[] => target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    return { backend, bottom, caches, drawn, target, top };
  };

  it('ignores a plan prepared for another document or isolation', () => {
    const { backend, bottom, caches, drawn, target, top } = scene();
    const doc = makeDoc([rasterLayer('top'), rasterLayer('bottom')]);
    const isolatedTop = prepareComposite(doc, { isolationLayerId: 'top' });

    compositeDocument(target, doc, caches, VIEW, { backend, preparation: isolatedTop });
    compositeDocument(target, { ...doc, stacks: stacksFrom([rasterLayer('bottom')]) }, caches, VIEW, {
      backend,
      preparation: prepareComposite(doc),
    });

    expect(drawn()).toEqual([bottom.surface.canvas, top.surface.canvas, bottom.surface.canvas]);
  });
});

describe('compositeDocument — damage', () => {
  const scene = () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const a = caches.getOrCreateRect('a', { height: 10, width: 10, x: 0, y: 0 });
    const b = caches.getOrCreateRect('b', { height: 10, width: 10, x: 0, y: 0 });
    const doc = makeDoc([
      rasterLayer('a'),
      rasterLayer('b', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 50, y: 50 } }),
    ]);
    const target = backend.createSurface(100, 100);
    const ops = (op: string) => target.callLog.filter((entry) => entry.op === op);
    return { a, b, backend, caches, doc, ops, target };
  };

  it('clears only the damaged screen region and skips layers outside it', () => {
    const { a, backend, caches, doc, ops, target } = scene();

    compositeDocument(target, doc, caches, VIEW, {
      backend,
      damage: { kind: 'regions', regions: [{ layerId: 'a', rect: { height: 4, width: 4, x: 2, y: 2 } }] },
    });

    expect(ops('clearRect').map((entry) => entry.args)).toEqual([[1, 1, 6, 6]]);
    expect(ops('drawImage').map((entry) => entry.args[0])).toEqual([a.surface.canvas]);
  });

  it('repaints everything when damage names a layer the document no longer has', () => {
    const { backend, caches, doc, ops, target } = scene();

    compositeDocument(target, doc, caches, VIEW, {
      backend,
      damage: { kind: 'regions', regions: [{ layerId: 'gone', rect: { height: 4, width: 4, x: 2, y: 2 } }] },
    });

    expect(ops('clearRect').map((entry) => entry.args)).toEqual([[0, 0, 100, 100]]);
    expect(ops('drawImage')).toHaveLength(2);
  });
});

describe('compositeDocument — control layer view', () => {
  // The original sits 4px right of the map's frame and is drawn at half size.
  const ORIGINAL_MATRIX: Mat2d = { a: 0.5, b: 0, c: 0, d: 0.5, e: 4, f: 0 };
  const LAYER_AT = { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 };

  const draws = (log: RasterCallLogEntry[]) => {
    let alpha: unknown = 1;
    let blend: unknown = 'source-over';
    let transform: unknown[] = [];
    const result: { source: unknown; alpha: unknown; blend: unknown; transform: unknown[] }[] = [];
    for (const entry of log) {
      if (entry.op === 'set' && entry.args[0] === 'globalAlpha') {
        alpha = entry.args[1];
      } else if (entry.op === 'set' && entry.args[0] === 'globalCompositeOperation') {
        blend = entry.args[1];
      } else if (entry.op === 'setTransform') {
        transform = entry.args;
      } else if (entry.op === 'drawImage') {
        result.push({ alpha, blend, source: entry.args[0], transform });
      }
    }
    return result;
  };

  const scene = (view: 'original' | 'overlay', layerOverrides: Record<string, unknown> = {}) => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const map = caches.getOrCreate('control', 10, 10);
    const original = backend.createSurface(40, 40);
    const layer = {
      ...controlLayer('control'),
      blendMode: 'multiply',
      opacity: 0.8,
      transform: LAYER_AT,
      withTransparencyEffect: true,
      ...layerOverrides,
    } as CanvasLayerContract;
    const target = backend.createSurface(200, 200);
    const controlView = {
      layerId: 'control',
      matrix: ORIGINAL_MATRIX,
      rect: { height: 40, width: 40, x: 0, y: 0 },
      surface: original,
      view,
    };
    return { backend, caches, controlView, layer, map, original, target };
  };

  it('draws the original through the layer and original transforms in place of the map, without the effect', () => {
    const { backend, caches, controlView, layer, original, target } = scene('original');
    const derivedSurfaces = createDerivedSurfaceCache();

    compositeDocument(target, makeDoc([layer]), caches, VIEW, { backend, controlView, derivedSurfaces });

    expect(draws(target.callLog)).toEqual([
      { alpha: 0.8, blend: 'multiply', source: original.canvas, transform: [0.5, 0, 0, 0.5, 14, 20] },
    ]);
    expect(derivedSurfaces.byteSize()).toBe(0);
  });

  it('draws the map over the original at half the layer opacity in the overlay view', () => {
    const { backend, caches, controlView, layer, map, original, target } = scene('overlay', {
      withTransparencyEffect: false,
    });

    compositeDocument(target, makeDoc([layer]), caches, VIEW, { backend, controlView });

    expect(draws(target.callLog)).toEqual([
      { alpha: 0.8, blend: 'multiply', source: original.canvas, transform: [0.5, 0, 0, 0.5, 14, 20] },
      { alpha: 0.4, blend: 'multiply', source: map.surface.canvas, transform: [1, 0, 0, 1, 10, 20] },
    ]);
  });
});
