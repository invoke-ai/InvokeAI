import type { CanvasDocumentSnapshot, LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasStateContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ExportLayerPixelsResult } from '@workbench/canvas-engine/controllers/rasterExportController';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createRasterSnapshotCapture, type CreateRasterSnapshotCaptureDeps } from './rasterSnapshotCapture';

const surface = (width = 4, height = 4): RasterSurface =>
  ({
    canvas: {},
    ctx: { clearRect: vi.fn(), drawImage: vi.fn(), setTransform: vi.fn() },
    height,
    width,
  }) as unknown as RasterSurface;

const imageLayer = (id: string, size = 4) => ({
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  source: { image: { height: size, imageName: `${id}.png`, width: size }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const canvasState = (layerIds: readonly string[] = ['a']): CanvasStateContractV3 =>
  ({
    document: { stacks: stacksFrom(layerIds.map((id) => imageLayer(id) as unknown as CanvasLayerContract)) },
  }) as unknown as CanvasStateContractV3;

const guard = (layerId: string): LayerExportGuard => ({ layerId }) as LayerExportGuard;

const okPixels = (layerId: string, size = 4): ExportLayerPixelsResult => ({
  guard: guard(layerId),
  rect: { height: size, width: size, x: 0, y: 0 },
  release: vi.fn(),
  status: 'ok',
  surface: surface(size, size),
});

/** Mutable so a test can advance engine state mid-capture, the way an edit does. */
interface Harness {
  deps: CreateRasterSnapshotCaptureDeps;
  memory: RasterMemoryBudgetController;
  state: {
    canvas: CanvasStateContractV3 | null;
    directPixelEpoch: number;
    disposed: boolean;
    documentGeneration: number;
    lifecycleGeneration: number;
  };
  currentGuards: Set<LayerExportGuard>;
  rasterize: ReturnType<typeof vi.fn>;
}

let harness: Harness;

const makeHarness = (
  overrides: Partial<Omit<CreateRasterSnapshotCaptureDeps, 'memory'>> = {},
  memory = new RasterMemoryBudgetController({ budgetBytes: 1_000_000 })
): Harness => {
  const state = {
    canvas: canvasState() as CanvasStateContractV3 | null,
    directPixelEpoch: 0,
    disposed: false,
    documentGeneration: 1,
    lifecycleGeneration: 0,
  };
  const currentGuards = new Set<LayerExportGuard>();
  const rasterize = vi.fn((layerId: string) => {
    const result = okPixels(layerId);
    currentGuards.add(result.status === 'ok' ? result.guard : guard(layerId));
    return Promise.resolve(result);
  });
  const deps: CreateRasterSnapshotCaptureDeps = {
    createSurface: (width, height) => surface(width, height),
    getCanvasState: () => state.canvas,
    getDirectPixelEpoch: () => state.directPixelEpoch,
    getDocumentGeneration: () => state.documentGeneration,
    getLifecycleGeneration: () => state.lifecycleGeneration,
    isDisposed: () => state.disposed,
    isGuardCurrent: (candidate) => currentGuards.has(candidate),
    memory,
    rasterizeLayerPixels: rasterize as unknown as CreateRasterSnapshotCaptureDeps['rasterizeLayerPixels'],
    ...overrides,
  };
  return { currentGuards, deps, memory, rasterize, state };
};

beforeEach(() => {
  harness = makeHarness();
});

describe('captureDocumentSnapshot', () => {
  it('clones the canvas rather than aliasing it', () => {
    const snapshot = createRasterSnapshotCapture(harness.deps).captureDocumentSnapshot();
    expect(snapshot?.canvas).not.toBe(harness.state.canvas);
    expect(getDocumentLeaves(snapshot?.canvas.document)[0]?.id).toBe('a');
  });

  it('refuses once disposed, and when there is no canvas', () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    harness.state.disposed = true;
    expect(capture.captureDocumentSnapshot()).toBeNull();
    harness.state.disposed = false;
    harness.state.canvas = null;
    expect(capture.captureDocumentSnapshot()).toBeNull();
  });
});

describe('isDocumentSnapshotCurrent', () => {
  it('holds while nothing has moved', () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect(capture.isDocumentSnapshotCurrent(snapshot)).toBe(true);
  });

  it.each([
    ['the canvas object is replaced', (h: Harness) => (h.state.canvas = canvasState())],
    ['the raster content epoch advances', (h: Harness) => (h.state.directPixelEpoch += 1)],
    ['the lifecycle generation advances', (h: Harness) => (h.state.lifecycleGeneration += 1)],
    ['the document generation advances', (h: Harness) => (h.state.documentGeneration += 1)],
    ['the engine is disposed', (h: Harness) => (h.state.disposed = true)],
  ])('breaks once %s', (_label, mutate) => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    mutate(harness);
    expect(capture.isDocumentSnapshotCurrent(snapshot)).toBe(false);
  });

  it('rejects a snapshot it never issued', () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const foreign = { canvas: canvasState(), documentGeneration: 1 } as CanvasDocumentSnapshot;
    expect(capture.isDocumentSnapshotCurrent(foreign)).toBe(false);
  });

  it('rejects a replacement canvas that is structurally identical', () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    // Same layers, same generation — only the object identity differs, which is
    // exactly what a reducer pass produces.
    harness.state.canvas = canvasState();
    expect(capture.isDocumentSnapshotCurrent(snapshot)).toBe(false);
  });
});

describe('captureRasterSnapshot', () => {
  it('pins every captured layer until the capture settles', async () => {
    let resolveRasterize!: (result: ExportLayerPixelsResult) => void;
    const rasterizeLayerPixels = vi.fn(
      () =>
        new Promise<ExportLayerPixelsResult>((resolve) => {
          resolveRasterize = resolve;
        })
    );
    const guarded = makeHarness({ rasterizeLayerPixels });
    const capture = createRasterSnapshotCapture(guarded.deps);
    const snapshot = capture.captureDocumentSnapshot()!;

    const pending = capture.captureRasterSnapshot(snapshot, ['a']);
    expect(guarded.memory.isPinned('a')).toBe(true);

    const pixels = okPixels('a');
    if (pixels.status === 'ok') {
      guarded.currentGuards.add(pixels.guard);
    }
    resolveRasterize(pixels);
    await expect(pending).resolves.toMatchObject({ status: 'ok' });
    expect(guarded.memory.isPinned('a')).toBe(false);
  });

  it('detaches a surface per layer and reports the empty ones separately', async () => {
    harness.state.canvas = canvasState(['a', 'b']);
    harness.rasterize.mockImplementation((layerId: string) => {
      if (layerId === 'b') {
        return Promise.resolve({ status: 'empty' });
      }
      const result = okPixels(layerId);
      harness.currentGuards.add(result.status === 'ok' ? result.guard : guard(layerId));
      return Promise.resolve(result);
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    const result = await capture.captureRasterSnapshot(snapshot, ['a', 'b']);
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    expect([...result.snapshot.layerSurfaces.keys()]).toEqual(['a']);
    expect([...result.snapshot.emptyLayerIds]).toEqual(['b']);
  });

  it('rasterizes a repeated layer id only once', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    await capture.captureRasterSnapshot(snapshot, ['a', 'a', 'a']);
    expect(harness.rasterize).toHaveBeenCalledTimes(1);
  });

  it('copies the rect rather than aliasing the live one', async () => {
    const live = okPixels('a');
    harness.rasterize.mockImplementation(() => {
      harness.currentGuards.add(live.status === 'ok' ? live.guard : guard('a'));
      return Promise.resolve(live);
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    const result = await capture.captureRasterSnapshot(snapshot, ['a']);
    expect(result.status).toBe('ok');
    if (result.status !== 'ok' || live.status !== 'ok') {
      return;
    }
    expect(result.snapshot.layerSurfaces.get('a')?.rect).not.toBe(live.rect);
  });

  it('releases the live surface even though it detached a copy', async () => {
    const live = okPixels('a');
    harness.rasterize.mockImplementation(() => {
      harness.currentGuards.add(live.status === 'ok' ? live.guard : guard('a'));
      return Promise.resolve(live);
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    await capture.captureRasterSnapshot(snapshot, ['a']);
    expect(live.status === 'ok' && live.release).toHaveBeenCalled();
  });

  it('refuses a snapshot it never issued as not-ready, not stale', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const foreign = { canvas: canvasState(), documentGeneration: 1 } as CanvasDocumentSnapshot;
    expect((await capture.captureRasterSnapshot(foreign, ['a'])).status).toBe('not-ready');
  });

  it('refuses a layer that is not in the snapshot', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['missing'])).status).toBe('not-ready');
    expect(harness.rasterize).not.toHaveBeenCalled();
  });

  it('refuses a layer whose source cannot be exported', async () => {
    harness.state.canvas = {
      document: {
        stacks: stacksFrom([
          { ...imageLayer('a'), source: { kind: 'polygon', type: 'shape' } } as unknown as CanvasLayerContract,
        ]),
      },
    } as unknown as CanvasStateContractV3;
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe('not-ready');
  });

  it('reports over-budget without rasterizing when the estimate does not fit', async () => {
    const harnessTiny = makeHarness({}, new RasterMemoryBudgetController({ budgetBytes: 8 }));
    const capture = createRasterSnapshotCapture(harnessTiny.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe('over-budget');
    expect(harnessTiny.rasterize).not.toHaveBeenCalled();
  });

  it('reports over-budget when a surface rasterizes larger than its estimate', async () => {
    // The 4x4 image estimates at 64 B; a 64x64 rasterized surface needs 16 KB more.
    const tight = makeHarness({}, new RasterMemoryBudgetController({ budgetBytes: 128 }));
    tight.rasterize.mockImplementation(() => {
      const result = { ...okPixels('a'), surface: surface(64, 64) } as ExportLayerPixelsResult;
      tight.currentGuards.add(result.status === 'ok' ? result.guard : guard('a'));
      return Promise.resolve(result);
    });
    const capture = createRasterSnapshotCapture(tight.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe('over-budget');
  });

  it.each([
    ['aborted', 'aborted'],
    ['over-budget', 'over-budget'],
    ['missing', 'not-ready'],
    ['disabled', 'not-ready'],
    ['unsupported', 'not-ready'],
  ])('maps a %s rasterize to %s', async (rasterizeStatus, expected) => {
    harness.rasterize.mockResolvedValue({ status: rasterizeStatus });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe(expected);
  });

  it('aborts before doing any work when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'], { signal: controller.signal })).status).toBe(
      'aborted'
    );
    expect(harness.rasterize).not.toHaveBeenCalled();
  });

  it('aborts when the signal fires during a rasterize', async () => {
    const controller = new AbortController();
    harness.rasterize.mockImplementation(() => {
      controller.abort();
      return Promise.resolve(okPixels('a'));
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'], { signal: controller.signal })).status).toBe(
      'aborted'
    );
  });

  it('goes stale when the document changes mid-capture, before any surface is kept', async () => {
    harness.state.canvas = canvasState(['a', 'b']);
    harness.rasterize.mockImplementation((layerId: string) => {
      if (layerId === 'b') {
        harness.state.directPixelEpoch += 1;
      }
      const result = okPixels(layerId);
      harness.currentGuards.add(result.status === 'ok' ? result.guard : guard(layerId));
      return Promise.resolve(result);
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a', 'b'])).status).toBe('stale');
  });

  it('detaches nothing once it notices the document moved, rather than copying and discarding', async () => {
    const createSurface = vi.fn((width: number, height: number) => surface(width, height));
    const detaching = makeHarness({ createSurface });
    detaching.state.canvas = canvasState(['a', 'b']);
    detaching.rasterize.mockImplementation((layerId: string) => {
      // The edit lands while the FIRST layer is still rasterizing.
      detaching.state.directPixelEpoch += 1;
      const result = okPixels(layerId);
      detaching.currentGuards.add(result.status === 'ok' ? result.guard : guard(layerId));
      return Promise.resolve(result);
    });
    const capture = createRasterSnapshotCapture(detaching.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a', 'b'])).status).toBe('stale');
    expect(createSurface).not.toHaveBeenCalled();
  });

  it('goes stale when a layer rasterizes against an already-superseded guard', async () => {
    harness.rasterize.mockResolvedValue(okPixels('a')); // never registered as current
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe('stale');
  });

  it('goes stale when an early guard is invalidated by a later layer', async () => {
    harness.state.canvas = canvasState(['a', 'b']);
    const guards: LayerExportGuard[] = [];
    harness.rasterize.mockImplementation((layerId: string) => {
      const result = okPixels(layerId);
      if (result.status === 'ok') {
        guards.push(result.guard);
        harness.currentGuards.add(result.guard);
      }
      // Editing `a` while `b` rasterizes retires a's guard but not the document.
      if (layerId === 'b' && guards[0]) {
        harness.currentGuards.delete(guards[0]);
      }
      return Promise.resolve(result);
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a', 'b'])).status).toBe('stale');
  });
});

describe('snapshot lifetime', () => {
  it('releases idempotently and empties what it held', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const documentSnapshot = capture.captureDocumentSnapshot()!;
    const result = await capture.captureRasterSnapshot(documentSnapshot, ['a']);
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    result.snapshot.release();
    result.snapshot.release();
    expect(result.snapshot.layerSurfaces.size).toBe(0);
  });

  it('releases every outstanding snapshot on teardown', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const documentSnapshot = capture.captureDocumentSnapshot()!;
    const first = await capture.captureRasterSnapshot(documentSnapshot, ['a']);
    const second = await capture.captureRasterSnapshot(documentSnapshot, ['a']);
    capture.releaseActiveSnapshots();
    expect(first.status === 'ok' && first.snapshot.layerSurfaces.size).toBe(0);
    expect(second.status === 'ok' && second.snapshot.layerSurfaces.size).toBe(0);
  });

  it('stops tracking a snapshot the caller already released', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const documentSnapshot = capture.captureDocumentSnapshot()!;
    const result = await capture.captureRasterSnapshot(documentSnapshot, ['a']);
    if (result.status !== 'ok') {
      throw new Error('expected a captured snapshot');
    }
    result.snapshot.release();
    // Teardown must not re-release, which would double-release the byte lease.
    expect(() => capture.releaseActiveSnapshots()).not.toThrow();
  });

  it('frees the detached bytes it reserved once released', async () => {
    const capture = createRasterSnapshotCapture(harness.deps);
    const documentSnapshot = capture.captureDocumentSnapshot()!;
    const before = harness.memory.getAvailableBytes();
    const result = await capture.captureRasterSnapshot(documentSnapshot, ['a']);
    expect(harness.memory.getAvailableBytes()).toBeLessThan(before);
    if (result.status === 'ok') {
      result.snapshot.release();
    }
    expect(harness.memory.getAvailableBytes()).toBe(before);
  });
});

describe('region capture accounting and cleanup', () => {
  it.each(['stale', 'aborted'] as const)('releases a read lease returned after capture becomes %s', async (status) => {
    const controller = new AbortController();
    const live = okPixels('a');
    const pin = harness.memory.pin('a');
    if (live.status !== 'ok') {
      throw new Error('expected live pixels');
    }
    live.release = vi.fn(() => pin.release());
    harness.rasterize.mockImplementation(() => {
      if (status === 'stale') {
        harness.state.directPixelEpoch++;
      } else {
        controller.abort();
      }
      return Promise.resolve(live);
    });
    const capture = createRasterSnapshotCapture(harness.deps);
    const result = await capture.captureRasterSnapshot(capture.captureDocumentSnapshot()!, ['a'], {
      region: { x: 0, y: 0, width: 1, height: 1 },
      signal: controller.signal,
    });
    expect(result.status).toBe(status);
    expect(live.release).toHaveBeenCalledOnce();
    expect(harness.memory.isPinned('a')).toBe(false);
    expect(harness.memory.snapshot()).toMatchObject({ reservedBytes: 0, detachedBytes: 0 });
  });

  it.each([100, 1000])(
    'tops up an underestimated source before allocating a cropped surface with a %i byte budget',
    async (budget) => {
      const memory = new RasterMemoryBudgetController({ budgetBytes: budget });
      const createSurface = vi.fn((width: number, height: number) => surface(width, height));
      const h = makeHarness({ createSurface }, memory);
      const live = okPixels('a', 20);
      if (live.status !== 'ok') {
        throw new Error('expected live pixels');
      }
      h.currentGuards.add(live.guard);
      h.rasterize.mockResolvedValue(live);
      const capture = createRasterSnapshotCapture(h.deps);
      const result = await capture.captureRasterSnapshot(capture.captureDocumentSnapshot()!, ['a', 'a'], {
        region: { x: 10, y: 10, width: 8, height: 8 },
      });
      expect(h.rasterize).toHaveBeenCalledOnce();
      if (budget === 100) {
        expect(result.status).toBe('over-budget');
        expect(createSurface).not.toHaveBeenCalled();
      } else {
        expect(result.status).toBe('ok');
        if (result.status !== 'ok') {
          throw new Error('expected captured pixels');
        }
        expect(result.snapshot.layerSurfaces.get('a')?.rect).toEqual({ x: 9, y: 9, width: 10, height: 10 });
        expect(memory.snapshot().detachedBytes).toBe(400);
        result.snapshot.release();
      }
      expect(live.release).toHaveBeenCalledOnce();
      expect(memory.isPinned('a')).toBe(false);
      expect(memory.snapshot()).toMatchObject({ reservedBytes: 0, detachedBytes: 0 });
    }
  );

  it('represents a disjoint layer without copying its pixels or losing its original bounds', async () => {
    const createSurface = vi.fn((width: number, height: number) => surface(width, height));
    const h = makeHarness({ createSurface });
    const capture = createRasterSnapshotCapture(h.deps);
    const result = await capture.captureRasterSnapshot(capture.captureDocumentSnapshot()!, ['a'], {
      region: { x: 100, y: 100, width: 10, height: 10 },
    });
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('expected captured pixels');
    }
    const layer = result.snapshot.layerSurfaces.get('a')!;
    expect(layer.rect).toEqual({ x: 0, y: 0, width: 0, height: 0 });
    expect(layer.sourceRect).toEqual({ x: 0, y: 0, width: 4, height: 4 });
    expect(createSurface).toHaveBeenCalledWith(1, 1);
    expect(layer.surface.ctx.drawImage).not.toHaveBeenCalled();
    expect(h.memory.snapshot().detachedBytes).toBe(4);
    result.snapshot.release();
  });
});
