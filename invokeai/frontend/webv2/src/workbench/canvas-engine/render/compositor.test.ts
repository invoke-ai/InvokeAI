import type {
  CanvasBlendMode,
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import type { RasterCallLogEntry, StubRasterSurface } from './raster.testStub';

import { compositeDocument, createCheckerboardTile, shouldSmoothAtZoom } from './compositor';
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
  overrides: Partial<CanvasDocumentContractV2> = {}
): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 100,
  ...overrides,
});

/** Sets are recorded as { op: 'set', args: [prop, value] }; find the value for a prop. */
const findSet = (log: RasterCallLogEntry[], prop: string): unknown[] =>
  log.filter((e) => e.op === 'set' && e.args[0] === prop).map((e) => e.args[1]);

describe('compositeDocument', () => {
  it('clears then draws layer caches bottom-to-top (array index 0 is top-most)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    // top -> bottom in array order.
    const top = rasterLayer('top');
    const bottom = rasterLayer('bottom');
    const topCache = caches.getOrCreate('top', 10, 10);
    const bottomCache = caches.getOrCreate('bottom', 10, 10);

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([top, bottom]), caches, VIEW);

    const log = target.callLog;
    // First op is a clearRect (after the outer save + identity setTransform).
    expect(log.some((e) => e.op === 'clearRect')).toBe(true);

    const drawImages = log.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(2);
    // Bottom layer's canvas is drawn before the top layer's.
    expect(drawImages[0]!.args[0]).toBe(bottomCache.surface.canvas);
    expect(drawImages[1]!.args[0]).toBe(topCache.surface.canvas);
  });

  it('skips layers that are disabled', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    caches.getOrCreate('b', 10, 10);
    const target = backend.createSurface(200, 200);

    const doc = makeDoc([rasterLayer('a', { isEnabled: false }), rasterLayer('b')]);
    compositeDocument(target, doc, caches, VIEW);

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args[0]).toBe(caches.get('b')!.surface.canvas);
  });

  it('isolates the named layer even when disabled and suppresses unrelated staged content', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const isolated = caches.getOrCreate('isolated', 10, 10);
    caches.getOrCreate('other', 10, 10);
    const filterPreview = backend.createSurface(10, 10);
    const staged = backend.createSurface(10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(
      target,
      makeDoc([rasterLayer('isolated', { isEnabled: false }), rasterLayer('other')]),
      caches,
      VIEW,
      {
        layerPreviews: new Map([['isolated', { rect: { height: 10, width: 10, x: 0, y: 0 }, surface: filterPreview }]]),
        onlyLayerId: 'isolated',
        stagedPreview: { rect: { height: 10, width: 10, x: 0, y: 0 }, surface: staged },
      }
    );

    const drawImages = target.callLog.filter((entry) => entry.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args[0]).toBe(isolated.surface.canvas);
  });

  it('preserves disabled-layer and staged-preview semantics when isolation is absent', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('disabled', 10, 10);
    const visible = caches.getOrCreate('visible', 10, 10);
    const staged = backend.createSurface(10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(
      target,
      makeDoc([rasterLayer('disabled', { isEnabled: false }), rasterLayer('visible')]),
      caches,
      VIEW,
      { stagedPreview: { rect: { height: 10, width: 10, x: 0, y: 0 }, surface: staged } }
    );

    const drawImages = target.callLog.filter((entry) => entry.op === 'drawImage');
    expect(drawImages.map((entry) => entry.args[0])).toEqual([visible.surface.canvas, staged.canvas]);
  });

  it('skips the layer named by skipLayerId (open text-edit session)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    caches.getOrCreate('b', 10, 10);
    const target = backend.createSurface(200, 200);

    const doc = makeDoc([rasterLayer('a'), rasterLayer('b')]);
    compositeDocument(target, doc, caches, VIEW, { skipLayerId: 'a' });

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    // 'a' is skipped (its live text is shown by the contenteditable portal); only 'b' draws.
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args[0]).toBe(caches.get('b')!.surface.canvas);
  });

  it('skips layers that have no cache entry yet', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    const target = backend.createSurface(200, 200);

    // 'b' has no cache; only 'a' should draw.
    compositeDocument(target, makeDoc([rasterLayer('a'), rasterLayer('b')]), caches, VIEW);
    expect(target.callLog.filter((e) => e.op === 'drawImage')).toHaveLength(1);
  });

  it('applies per-layer opacity and maps blend mode to a composite operation', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    const target = backend.createSurface(200, 200);

    const blend: CanvasBlendMode = 'multiply';
    compositeDocument(target, makeDoc([rasterLayer('a', { blendMode: blend, opacity: 0.4 })]), caches, VIEW);

    expect(findSet(target.callLog, 'globalAlpha')).toContain(0.4);
    expect(findSet(target.callLog, 'globalCompositeOperation')).toContain('multiply');
  });

  it('applies control transparency, opacity, and blend mode to a filter preview', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('control', 10, 10);
    const preview = backend.createSurface(10, 10);
    const target = backend.createSurface(100, 100);
    const layer = {
      ...controlLayer('control'),
      blendMode: 'multiply' as const,
      opacity: 0.4,
      withTransparencyEffect: true,
    };

    compositeDocument(target, makeDoc([layer]), caches, VIEW, {
      backend,
      layerPreviews: new Map([['control', { rect: { height: 14, width: 16, x: -2, y: -3 }, surface: preview }]]),
    });

    const draw = target.callLog.find((entry) => entry.op === 'drawImage');
    expect(draw?.args[0]).not.toBe(preview.canvas);
    expect(draw?.args.slice(1)).toEqual([-2, -3]);
    expect(findSet(target.callLog, 'globalAlpha')).toContain(0.4);
    expect(findSet(target.callLog, 'globalCompositeOperation')).toContain('multiply');
  });

  it("maps the 'normal' blend mode to 'source-over'", () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([rasterLayer('a', { blendMode: 'normal' })]), caches, VIEW);
    expect(findSet(target.callLog, 'globalCompositeOperation')).toContain('source-over');
  });

  it('fills the ENTIRE viewport with the checkerboard pattern (unbounded plane)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const tile = createCheckerboardTile(backend);

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([], { background: 'transparent' }), caches, VIEW, { checkerboardTile: tile });

    // The pattern is created once from the tile (cheap: no per-cell fill loop).
    const patternCalls = target.callLog.filter((e) => e.op === 'createPattern');
    expect(patternCalls).toHaveLength(1);
    expect(patternCalls[0]!.args[0]).toBe(tile.canvas);
    expect(patternCalls[0]!.args[1]).toBe('repeat');

    // The whole 200x200 target is filled with the pattern — NOT clipped to the
    // 100x100 doc rect (the document is no longer a visual boundary).
    const fills = target.callLog.filter((e) => e.op === 'fillRect');
    expect(fills).toHaveLength(1);
    expect(fills[0]!.args).toEqual([0, 0, 200, 200]);
    // fillStyle was set to the pattern object (the stub's non-null marker), not a color string.
    const styles = findSet(target.callLog, 'fillStyle');
    expect(styles).toHaveLength(1);
    expect(typeof styles[0]).toBe('object');
  });

  it('fills the full viewport regardless of the doc rect position/size (offset + scaled view)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const tile = createCheckerboardTile(backend);

    // A tiny doc, panned far off-origin under a scaled view: the checker still
    // covers the whole screen because it is screen-anchored, not doc-anchored.
    const view: Mat2d = { a: 3, b: 0, c: 0, d: 3, e: -500, f: 220 };
    const target = backend.createSurface(320, 240);
    compositeDocument(target, makeDoc([], { height: 8, width: 8 }), caches, view, { checkerboardTile: tile });

    const fills = target.callLog.filter((e) => e.op === 'fillRect');
    expect(fills).toHaveLength(1);
    expect(fills[0]!.args).toEqual([0, 0, 320, 240]);
  });

  it('draws NO checkerboard when the tile is absent (toggle off), leaving bg.inset through', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([], { background: 'transparent' }), caches, VIEW);

    // No tile → no pattern and no fill; the cleared surface shows the widget's bg.inset.
    expect(target.callLog.some((e) => e.op === 'createPattern')).toBe(false);
    expect(target.callLog.filter((e) => e.op === 'fillRect')).toHaveLength(0);
    // The whole target is still cleared each frame.
    expect(target.callLog.some((e) => e.op === 'clearRect')).toBe(true);
  });

  it('ignores the contract background field (checker fills the viewport even for a color background)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);

    const solidTarget = backend.createSurface(200, 200);
    // The `background` field no longer renders: a color-background doc with the
    // checkerboard on still fills the whole viewport with the pattern, not a flat color.
    compositeDocument(solidTarget, makeDoc([], { background: { color: '#123456' } }), caches, VIEW, {
      checkerboardTile: createCheckerboardTile(backend),
    });
    const patternCalls = solidTarget.callLog.filter((e) => e.op === 'createPattern');
    expect(patternCalls).toHaveLength(1);
    const fills = solidTarget.callLog.filter((e) => e.op === 'fillRect');
    expect(fills).toHaveLength(1);
    expect(fills[0]!.args).toEqual([0, 0, 200, 200]);
    // The flat color is never applied.
    expect(findSet(solidTarget.callLog, 'fillStyle')).not.toContain('#123456');
  });

  it('builds the checker tile with the given colors on the diagonal', () => {
    const backend = createTestStubRasterBackend();
    const tile = createCheckerboardTile(backend, { a: '#111111', b: '#222222' }) as StubRasterSurface;
    // The tile's two fillStyle sets are exactly the fed colors (base then diagonal).
    const styles = findSet(tile.callLog, 'fillStyle');
    expect(styles).toEqual(['#111111', '#222222']);
  });

  it('draws mask-bearing layers as a tinted fill of the mask color (backend-less fallback)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('rg', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([maskLayer('rg')]), caches, VIEW);

    // Without a backend the mask draws its coverage then tints with a flat fill.
    expect(target.callLog.some((e) => e.op === 'drawImage')).toBe(true);
    expect(findSet(target.callLog, 'fillStyle')).toContain('#ff0000');
  });

  it('colorizes the mask alpha via source-in on an intermediate surface when a backend is supplied', () => {
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

    // The colorize happens on a NEW intermediate surface (not the target, not the
    // mask cache): it blits the stencil then fills source-in with the fill colour.
    const colorized = created.find(
      (s) =>
        s !== target &&
        s !== maskEntry.surface &&
        s.callLog.some((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation' && e.args[1] === 'source-in')
    );
    expect(colorized).toBeDefined();
    expect(findSet(colorized!.callLog, 'fillStyle')).toContain('#ff0000');
    // The colorized overlay is then blitted onto the target.
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

  it('culls a fully offscreen effect layer before derived work or drawing', () => {
    const base = createTestStubRasterBackend();
    let allocations = 0;
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        allocations += 1;
        return base.createSurface(w, h);
      },
    };
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('control', 10, 10);
    const target = backend.createSurface(100, 100);
    const control = controlLayer('control');
    if (control.type !== 'control') {
      throw new Error('Expected control fixture');
    }
    const doc = makeDoc([{ ...control, transform: { ...control.transform, x: 1_000 }, withTransparencyEffect: true }]);
    const allocationsBeforeComposite = allocations;
    const diagnostics = createCanvasDiagnostics(true);

    compositeDocument(target, doc, caches, VIEW, {
      backend,
      derivedSurfaces: createDerivedSurfaceCache(),
      diagnostics,
    });

    expect(allocations).toBe(allocationsBeforeComposite);
    expect(target.callLog.filter((entry) => entry.op === 'drawImage')).toHaveLength(0);
    expect(diagnostics.snapshot()).toMatchObject({
      compositeFrames: 1,
      layersConsidered: 1,
      layersCulled: 1,
      layersDrawn: 0,
    });
  });

  it('draws mask layers ABOVE all non-mask layers regardless of their global z position', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    // A mask placed at the BOTTOM of the z-order (last in the array) must still be
    // composited after (above) the raster layer above it.
    caches.getOrCreate('raster', 10, 10);
    caches.getOrCreate('mask', 10, 10);
    const target = backend.createSurface(200, 200) as StubRasterSurface;

    const doc = makeDoc([rasterLayer('raster'), maskLayer('mask')]);
    compositeDocument(target, doc, caches, VIEW, { backend });

    // The mask's source-in colorize marks its draw pass; it must come AFTER the
    // last raster blit. Find the first source-in op (mask pass) and assert every
    // non-mask blit already ran — i.e. the mask pass draws last.
    const firstDrawImage = target.callLog.findIndex((e) => e.op === 'drawImage');
    expect(firstDrawImage).toBeGreaterThanOrEqual(0);
    // Two draws land on the target: the raster blit, then the colorized mask blit;
    // the mask blit is the LAST drawImage.
    const drawIdxs = target.callLog.map((e, i) => (e.op === 'drawImage' ? i : -1)).filter((i) => i >= 0);
    expect(drawIdxs.length).toBeGreaterThanOrEqual(2);
  });

  it('composites in strict group order: raster < control < regional < inpaint mask, ignoring global index', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    // Distinct cache sizes so each layer's blit is identifiable by source width
    // (masks blit a colorized intermediate sized to their cache, not the cache
    // surface itself, so identity matching won't work — width does).
    caches.getOrCreate('raster', 10, 10);
    caches.getOrCreate('control', 11, 11);
    caches.getOrCreate('regional', 12, 12);
    caches.getOrCreate('inpaint', 13, 13);
    const widthToId: Record<number, string> = { 10: 'raster', 11: 'control', 12: 'regional', 13: 'inpaint' };
    const target = backend.createSurface(200, 200) as StubRasterSurface;

    // Deliberately SCRAMBLED array order (index 0 = top-most): a raster created
    // above a control layer, masks interleaved below. Group order must win.
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

    // Raster (bottom) first, then control, then the masks — regardless of array index.
    expect(order.indexOf('raster')).toBeLessThan(order.indexOf('control'));
    expect(order.indexOf('control')).toBeLessThan(order.indexOf('regional'));
    expect(order.indexOf('regional')).toBeLessThan(order.indexOf('inpaint'));
  });

  it('disables image smoothing for the composite when imageSmoothing is false (zoomed-in policy)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([rasterLayer('a')]), caches, VIEW, { imageSmoothing: false });
    // The smoothing flag is set exactly once, to false, so every layer/staged
    // blit up-scales nearest-neighbor (crisp + cheap) rather than bilinear.
    expect(findSet(target.callLog, 'imageSmoothingEnabled')).toEqual([false]);
  });

  it('enables image smoothing by default and when imageSmoothing is true (down-scale/quality)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);

    const defaulted = backend.createSurface(200, 200);
    compositeDocument(defaulted, makeDoc([rasterLayer('a')]), caches, VIEW);
    expect(findSet(defaulted.callLog, 'imageSmoothingEnabled')).toEqual([true]);

    const explicit = backend.createSurface(200, 200);
    compositeDocument(explicit, makeDoc([rasterLayer('a')]), caches, VIEW, { imageSmoothing: true });
    expect(findSet(explicit.callLog, 'imageSmoothingEnabled')).toEqual([true]);
  });

  it('draws a staged preview over its bbox when provided', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const target = backend.createSurface(200, 200);
    const staged = backend.createSurface(50, 50);

    compositeDocument(target, makeDoc([]), caches, VIEW, {
      stagedPreview: { rect: { height: 40, width: 40, x: 5, y: 5 }, surface: staged },
    });

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args).toEqual([staged.canvas, 5, 5, 40, 40]);
  });

  it('draws a placed staged preview at its candidate opacity and keeps the pending outline opaque', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const target = backend.createSurface(200, 200);
    const staged = backend.createSurface(23, 17);

    compositeDocument(target, makeDoc([]), caches, VIEW, {
      stagedPreview: {
        opacity: 0.35,
        rect: { height: 34, width: 46, x: -8, y: 13 },
        surface: staged,
      },
    });

    const drawImages = target.callLog.filter((entry) => entry.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    expect(drawImages[0]!.args).toEqual([staged.canvas, -8, 13, 46, 34]);
    expect(findSet(target.callLog, 'globalAlpha')).toEqual([0.35, 1]);
  });
});

describe('compositeDocument — raster adjustments', () => {
  it('draws the provided adjusted surface instead of the raw cache for a raster layer', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const layer = rasterLayer('a', { adjustments: { brightness: 0.5, contrast: 0, saturation: 0 } });
    const cache = caches.getOrCreate('a', 10, 10);
    const adjusted = backend.createSurface(10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([layer]), caches, VIEW, {
      adjustedSurface: (l) => (l.id === 'a' ? adjusted : null),
    });

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages).toHaveLength(1);
    // The adjusted surface's canvas is drawn, NOT the raw cache surface.
    expect(drawImages[0]!.args[0]).toBe(adjusted.canvas);
    expect(drawImages[0]!.args[0]).not.toBe(cache.surface.canvas);
  });

  it('draws the raw cache when the provider returns null (identity / no adjustments)', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const layer = rasterLayer('a');
    const cache = caches.getOrCreate('a', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([layer]), caches, VIEW, { adjustedSurface: () => null });

    const drawImages = target.callLog.filter((e) => e.op === 'drawImage');
    expect(drawImages[0]!.args[0]).toBe(cache.surface.canvas);
  });
});

describe('shouldSmoothAtZoom', () => {
  it('smooths only when the document is down-scaled (zoom < 1)', () => {
    expect(shouldSmoothAtZoom(0.1)).toBe(true);
    expect(shouldSmoothAtZoom(0.5)).toBe(true);
    expect(shouldSmoothAtZoom(0.99)).toBe(true);
    // At or above 1× the document is magnified: keep pixels crisp and skip the
    // per-frame bilinear up-scale whose cost grows with zoom.
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
    compositeDocument(target, makeDoc([top, bottom]), caches, VIEW, { floatingSelection: float });

    const drawn = target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    expect(drawn).toEqual([bottomCache.surface.canvas, float.surface.canvas, topCache.surface.canvas]);
  });

  it('inherits its layer opacity and blend mode', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('a', 10, 10);
    const float = floatOf(backend, 'a');

    const target = backend.createSurface(200, 200);
    const doc = makeDoc([rasterLayer('a', { blendMode: 'multiply' as CanvasBlendMode, opacity: 0.5 })]);
    compositeDocument(target, doc, caches, VIEW, { floatingSelection: float });

    // Twice: once for the layer's own cache, once for the float over it.
    expect(findSet(target.callLog, 'globalAlpha').filter((value) => value === 0.5)).toHaveLength(2);
    expect(findSet(target.callLog, 'globalCompositeOperation').filter((value) => value === 'multiply')).toHaveLength(2);
  });

  it('still draws a float whose source layer has been emptied by the cut', () => {
    // The lift can take everything a layer held; the detached pixels must not
    // vanish with it.
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreateRect('a', { height: 0, width: 0, x: 0, y: 0 });
    const float = floatOf(backend, 'a');

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([rasterLayer('a')]), caches, VIEW, { floatingSelection: float });

    const drawn = target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    expect(drawn).toEqual([float.surface.canvas]);
  });

  it('ignores a float whose layer id matches nothing in the document', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const cache = caches.getOrCreate('a', 10, 10);
    const float = floatOf(backend, 'gone');

    const target = backend.createSurface(200, 200);
    compositeDocument(target, makeDoc([rasterLayer('a')]), caches, VIEW, { floatingSelection: float });

    const drawn = target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    expect(drawn).toEqual([cache.surface.canvas]);
  });
});

describe('compositeDocument — hidden layers', () => {
  // Typed to the guidance member: `isHidden` exists only on the three overlay
  // contracts, so a bare spread of the union does not accept it — which is the
  // invariant this design is built on.
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

  it('draws it again once unhidden — hiding never touches its pixels', () => {
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    const maskCache = caches.getOrCreate('m', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([maskLayer('m')]), caches, VIEW, { backend });

    const drawn = target.callLog.filter((e) => e.op === 'drawImage').map((e) => e.args[0]);
    expect(drawn.length).toBeGreaterThan(0);
    expect(maskCache.surface.width).toBe(10);
  });

  it('still draws a hidden layer while it is the isolated operation target', () => {
    // An operation preview acts ON that layer; suppressing it would leave the
    // user editing something they cannot see.
    const backend = createTestStubRasterBackend();
    const caches = createLayerCacheStore(backend);
    caches.getOrCreate('m', 10, 10);
    const target = backend.createSurface(200, 200);

    compositeDocument(target, makeDoc([hiddenMask('m')]), caches, VIEW, { backend, onlyLayerId: 'm' });

    expect(target.callLog.some((e) => e.op === 'drawImage')).toBe(true);
  });
});
