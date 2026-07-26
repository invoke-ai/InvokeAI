import type { CanvasDocumentSnapshot, LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasStateContractV2 } from '@workbench/canvas-engine/contracts';
import type { ExportLayerPixelsResult } from '@workbench/canvas-engine/controllers/rasterExportController';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
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

const canvasState = (layerIds: readonly string[] = ['a']): CanvasStateContractV2 =>
  ({ document: { layers: layerIds.map((id) => imageLayer(id)) } }) as unknown as CanvasStateContractV2;

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
  state: {
    canvas: CanvasStateContractV2 | null;
    contentEpoch: number;
    disposed: boolean;
    documentGeneration: number;
    lifecycleGeneration: number;
  };
  currentGuards: Set<LayerExportGuard>;
  rasterize: ReturnType<typeof vi.fn>;
}

let harness: Harness;

const makeHarness = (overrides: Partial<CreateRasterSnapshotCaptureDeps> = {}): Harness => {
  const state = {
    canvas: canvasState() as CanvasStateContractV2 | null,
    contentEpoch: 0,
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
    getContentEpoch: () => state.contentEpoch,
    getDocumentGeneration: () => state.documentGeneration,
    getLifecycleGeneration: () => state.lifecycleGeneration,
    isDisposed: () => state.disposed,
    isGuardCurrent: (candidate) => currentGuards.has(candidate),
    memory: new RasterMemoryBudgetController({ budgetBytes: 1_000_000 }),
    rasterizeLayerPixels: rasterize as unknown as CreateRasterSnapshotCaptureDeps['rasterizeLayerPixels'],
    syncMemoryBaselines: vi.fn(),
    ...overrides,
  };
  return { currentGuards, deps, rasterize, state };
};

beforeEach(() => {
  harness = makeHarness();
});

describe('captureDocumentSnapshot', () => {
  it('clones the canvas rather than aliasing it', () => {
    const snapshot = createRasterSnapshotCapture(harness.deps).captureDocumentSnapshot();
    expect(snapshot?.canvas).not.toBe(harness.state.canvas);
    expect(snapshot?.canvas.document.layers[0]?.id).toBe('a');
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
    ['the raster content epoch advances', (h: Harness) => (h.state.contentEpoch += 1)],
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
      document: { layers: [{ ...imageLayer('a'), source: { kind: 'polygon', type: 'shape' } }] },
    } as unknown as CanvasStateContractV2;
    const capture = createRasterSnapshotCapture(harness.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe('not-ready');
  });

  it('reports over-budget without rasterizing when the estimate does not fit', async () => {
    const harnessTiny = makeHarness({ memory: new RasterMemoryBudgetController({ budgetBytes: 8 }) });
    const capture = createRasterSnapshotCapture(harnessTiny.deps);
    const snapshot = capture.captureDocumentSnapshot()!;
    expect((await capture.captureRasterSnapshot(snapshot, ['a'])).status).toBe('over-budget');
    expect(harnessTiny.rasterize).not.toHaveBeenCalled();
  });

  it('reports over-budget when a surface rasterizes larger than its estimate', async () => {
    // The 4x4 image estimates at 64 B; a 64x64 rasterized surface needs 16 KB more.
    const tight = makeHarness({ memory: new RasterMemoryBudgetController({ budgetBytes: 128 }) });
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
        harness.state.contentEpoch += 1;
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
      detaching.state.contentEpoch += 1;
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
    const before = harness.deps.memory.getAvailableBytes();
    const result = await capture.captureRasterSnapshot(documentSnapshot, ['a']);
    expect(harness.deps.memory.getAvailableBytes()).toBeLessThan(before);
    if (result.status === 'ok') {
      result.snapshot.release();
    }
    expect(harness.deps.memory.getAvailableBytes()).toBe(before);
  });
});
