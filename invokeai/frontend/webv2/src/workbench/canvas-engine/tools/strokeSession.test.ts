import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { StubRasterBackend, StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createStrokeSession } from '@workbench/canvas-engine/tools/strokeSession';
import { describe, expect, it, vi } from 'vitest';

const pointer = (x: number, y: number): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

/** A capturing backend so the stroke scratch surface can be identified. */
const createCapturingBackend = (): { backend: StubRasterBackend; created: StubRasterSurface[] } => {
  const inner = createTestStubRasterBackend();
  const created: StubRasterSurface[] = [];
  return {
    backend: {
      ...inner,
      createSurface: (w, h) => {
        const s = inner.createSurface(w, h);
        created.push(s);
        return s;
      },
    },
    created,
  };
};

const runStroke = (opts: { withMask: boolean }) => {
  const { backend, created } = createCapturingBackend();
  const layers = createLayerCacheStore(backend);
  const entry: LayerCacheEntry = layers.getOrCreate('L', 100, 100);
  const mask = opts.withMask ? backend.createSurface(100, 100) : null;
  const clipMask = mask ? { rect: { height: 100, width: 100, x: 0, y: 0 }, surface: mask } : null;
  const emitStrokeCommitted = vi.fn();
  const notifyLayerPainted = vi.fn();

  const ctx = {
    backend,
    createPath2D: (d?: string) => ({ d }) as unknown as Path2D,
    emitStrokeCommitted,
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted,
  } as unknown as ToolContext;

  // Only the scratch is created after this point.
  created.length = 0;
  const session = createStrokeSession({
    clipMask,
    color: '#ff0000',
    composite: 'source-over',
    ctx,
    layerId: 'L',
    opacity: 1,
    size: 20,
    thinning: 0,
    tool: 'brush',
  });
  session.addPoints([pointer(10, 10)]);
  session.addPoints([pointer(40, 10), pointer(40, 40)]);
  const event = session.commit();

  const scratch = created[0]!;
  return { cache: entry.surface as StubRasterSurface, emitStrokeCommitted, event, notifyLayerPainted, scratch };
};

const compositeOps = (surface: StubRasterSurface): unknown[] =>
  surface.callLog.filter((e) => e.op === 'set' && e.args[0] === 'globalCompositeOperation').map((e) => e.args[1]);

describe('strokeSession: selection-constrained painting', () => {
  it('with a clip mask, intersects the scratch stroke with the mask (destination-in) before compositing', () => {
    const { scratch } = runStroke({ withMask: true });
    expect(compositeOps(scratch)).toContain('destination-in');
    // The mask is drawn into the scratch to clip it.
    expect(scratch.callLog.some((e) => e.op === 'drawImage')).toBe(true);
  });

  it('without a clip mask, the scratch stays a plain filled stroke (no extra clip ops)', () => {
    const { scratch } = runStroke({ withMask: false });
    expect(compositeOps(scratch)).not.toContain('destination-in');
    expect(scratch.callLog.some((e) => e.op === 'drawImage')).toBe(false);
    expect(scratch.callLog.filter((e) => e.op === 'fill')).not.toHaveLength(0);
  });

  it('applies the selection clip on the scratch, not by changing the cache composite ops', () => {
    const withMask = runStroke({ withMask: true });
    const noMask = runStroke({ withMask: false });
    // The clip is a `destination-in` on the SCRATCH; the layer-cache composite op
    // sequence (source-over blit of the clipped stroke) is unchanged — the clip
    // never adds a mask op to the cache itself. (The cache's content EXTENT can
    // differ: without a selection the cache grows to the stroke's true bounds,
    // while a selection bounds the growth to the mask — content-sized behavior.)
    expect(compositeOps(withMask.cache)).toEqual(compositeOps(noMask.cache));
    expect(compositeOps(withMask.cache)).not.toContain('destination-in');
  });

  it('returns a commit with a dirty rect the mask does not shift', () => {
    const { event } = runStroke({ withMask: true });
    expect(event!.dirtyRect.width).toBeGreaterThan(0);
    expect(event!.dirtyRect.height).toBeGreaterThan(0);
  });
});

describe('strokeSession: content-sized cache growth', () => {
  const makeSession = (initialRect: { x: number; y: number; width: number; height: number }) => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreateRect('L', initialRect);
    entry.stale = false;
    const emitStrokeCommitted = vi.fn();
    const notifyLayerPainted = vi.fn();
    const ctx = {
      backend,
      createPath2D: (d?: string) => ({ d }) as unknown as Path2D,
      emitStrokeCommitted,
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted,
    } as unknown as ToolContext;
    const session = createStrokeSession({
      clipMask: null,
      color: '#ff0000',
      composite: 'source-over',
      ctx,
      layerId: 'L',
      opacity: 1,
      size: 20,
      thinning: 0,
      tool: 'brush',
    });
    return { emitStrokeCommitted, entry, notifyLayerPainted, session };
  };

  it('grows an EMPTY (brand-new) paint cache to the stroke bounds on the first stroke', () => {
    const { entry, session } = makeSession({ height: 0, width: 0, x: 0, y: 0 });
    session.addPoints([pointer(50, 60)]);
    const event = session.commit();

    // The cache adopted the stroke's content bounds, snapped OUTWARD to the 64px
    // growth-chunk grid — still content-sized (a couple of chunks), NOT an
    // origin-anchored document-sized surface. A size-20 dab at (50,60) sits roughly
    // at [40,60]×[50,70], which chunk-pads to a small chunk-aligned rect.
    expect(entry.rect.width).toBeGreaterThan(0);
    expect(entry.rect.height).toBeGreaterThan(0);
    // Chunk-aligned extent (origin and size are multiples of the 64px chunk).
    expect(entry.rect.x % 64).toBe(0);
    expect(entry.rect.y % 64).toBe(0);
    expect(entry.rect.width % 64).toBe(0);
    expect(entry.rect.height % 64).toBe(0);
    // Content-sized: a few chunks around the dab, not a huge (document) surface.
    expect(entry.rect.width).toBeLessThanOrEqual(128);
    expect(entry.rect.height).toBeLessThanOrEqual(128);
    // The padded extent still fully contains the painted dab center (50,60).
    expect(entry.rect.x).toBeLessThanOrEqual(50);
    expect(entry.rect.x + entry.rect.width).toBeGreaterThan(50);
    expect(entry.rect.y).toBeLessThanOrEqual(60);
    expect(entry.rect.y + entry.rect.height).toBeGreaterThan(60);
    expect(entry.surface.width).toBe(entry.rect.width);
    expect(entry.surface.height).toBe(entry.rect.height);
    // The committed dirty rect is the same (chunk-padded) layer-local region.
    expect(event!.dirtyRect).toEqual(entry.rect);
  });

  it('returns the completed event without publishing engine side effects', () => {
    const { emitStrokeCommitted, notifyLayerPainted, session } = makeSession({ height: 0, width: 0, x: 0, y: 0 });
    session.addPoints([pointer(10, 10)]);
    const event = session.commit();

    expect(event).toMatchObject({ layerId: 'L', tool: 'brush' });
    expect(event!.dirtyRect.width).toBeGreaterThan(0);
    expect(event!.dirtyRect.height).toBeGreaterThan(0);
    expect(emitStrokeCommitted).not.toHaveBeenCalled();
    expect(notifyLayerPainted).not.toHaveBeenCalled();
  });

  it('returns null when the gesture produced no dirty pixels', () => {
    const { session } = makeSession({ height: 0, width: 0, x: 0, y: 0 });
    expect(session.commit()).toBeNull();
  });

  it('grows an existing cache to the UNION of its extent and an out-of-extent stroke (negative coords included)', () => {
    const { entry, session } = makeSession({ height: 20, width: 20, x: 0, y: 0 });
    session.addPoints([pointer(-40, -40)]);
    session.commit();

    // Union of the pre-stroke [0,20)² extent and the stroke bounds around
    // (-40,-40): the origin moved into negative layer-local space and the old
    // extent's far edge is still covered.
    expect(entry.rect.x).toBeLessThan(-30);
    expect(entry.rect.y).toBeLessThan(-30);
    expect(entry.rect.x + entry.rect.width).toBeGreaterThanOrEqual(20);
    expect(entry.rect.y + entry.rect.height).toBeGreaterThanOrEqual(20);
    expect(entry.surface.width).toBe(entry.rect.width);
    expect(entry.surface.height).toBe(entry.rect.height);
  });

  it('reallocates the cache surface O(stroke / chunk) times — NOT once per batch — across an extending drag', () => {
    // Pre-size the cache to a chunk-aligned rect already covering the first dab, so
    // only genuine growth (not the empty-cache adoption) counts as a reallocation.
    const { entry, session } = makeSession({ height: 64, width: 64, x: 64, y: 64 });
    const surface = entry.surface as StubRasterSurface;
    const resizeCount = (): number => surface.callLog.filter((e) => e.op === 'resize').length;

    // Ten 10px batches extending the stroke 100px rightward within a single chunk
    // row. Without chunk-padding, growToRect grows to the EXACT union every batch,
    // so each batch reallocates + full-copies the cache (≈10 resizes). With the
    // 64px chunk grid, successive small extensions land inside the padded extent,
    // so the surface reallocates only when the stroke crosses a chunk boundary
    // (~100 / 64 ≈ 2 times).
    const batches = 10;
    for (let i = 0; i < batches; i++) {
      session.addPoints([pointer(100 + i * 10, 100)]);
    }
    session.commit();

    const resizes = resizeCount();
    expect(resizes).toBeLessThanOrEqual(2);
    // Sanity: far fewer reallocations than batches — the unpadded behavior this
    // regression guards would resize on nearly every batch.
    expect(resizes).toBeLessThan(batches);
  });
});

describe('strokeSession: cache version bump (live adjusted-surface invalidation)', () => {
  const makeVersionSession = () => {
    const { backend } = createCapturingBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('L', 100, 100);
    entry.stale = false;
    const ctx = {
      backend,
      createPath2D: (d?: string) => ({ d }) as unknown as Path2D,
      emitStrokeCommitted: vi.fn(),
      invalidate: vi.fn(),
      layers,
      notifyLayerPainted: vi.fn(),
    } as unknown as ToolContext;
    const session = createStrokeSession({
      clipMask: null,
      color: '#ff0000',
      composite: 'source-over',
      ctx,
      layerId: 'L',
      opacity: 1,
      size: 20,
      thinning: 0,
      tool: 'brush',
    });
    return { entry, session };
  };

  it('bumps the cache version on every mid-stroke frame (so the adjusted-surface memo recomputes live)', () => {
    const { entry, session } = makeVersionSession();
    const v0 = entry.version;
    session.addPoints([pointer(10, 10)]);
    const v1 = entry.version;
    session.addPoints([pointer(40, 40)]);
    const v2 = entry.version;
    // Each painted frame advances the version — a version-keyed adjusted surface
    // would otherwise serve stale (pre-stroke) adjusted pixels mid-stroke.
    expect(v1).toBeGreaterThan(v0);
    expect(v2).toBeGreaterThan(v1);
  });

  it('bumps the version on cancel so the restored pixels re-derive the adjusted surface', () => {
    const { entry, session } = makeVersionSession();
    session.addPoints([pointer(10, 10)]);
    const vBeforeCancel = entry.version;
    session.cancel();
    expect(entry.version).toBeGreaterThan(vBeforeCancel);
  });
});
