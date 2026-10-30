import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type {
  SpatialAdjustmentExecutor,
  SpatialExecutorJob,
  SpatialExecutorResult,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';

import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { createInlineSpatialExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { LayerCacheEntry } from './layerCache';
import type { StubRasterSurface } from './raster.testStub';

import { createAdjustedSurfaceCache } from './adjustedSurfaceCache';
import { createDerivedSurfaceCache } from './derivedSurfaceCache';
import { createLayerCacheStore } from './layerCache';
import { createTestStubRasterBackend } from './raster.testStub';
import { type AdjustedSurfaceStatus, SPATIAL_CONTENT_DEBOUNCE_MS } from './spatialDisplay';

const BRIGHTEN: CanvasAdjustmentsContract = [
  { brightness: 0.5, contrast: 0, id: 'b', isEnabled: true, type: 'brightness-contrast' },
];
const CONTRAST: CanvasAdjustmentsContract = [
  { brightness: 0, contrast: 0.5, id: 'c', isEnabled: true, type: 'brightness-contrast' },
];

const makeEntry = (layerId: string, surface: StubRasterSurface, version = 0): LayerCacheEntry => ({
  hasPublishedPixels: true,
  lastUsed: 0,
  layerId,
  rect: { height: surface.height, width: surface.width, x: 0, y: 0 },
  stale: false,
  surface,
  version,
});

const drawImageCount = (surface: StubRasterSurface): number =>
  surface.callLog.filter((entry) => entry.op === 'drawImage').length;

describe('createAdjustedSurfaceCache', () => {
  it('returns null (and caches nothing) for identity adjustments', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));
    expect(cache.get('a', entry, undefined)).toBeNull();
    expect(
      cache.get('a', entry, [{ brightness: 0, contrast: 0, id: 'i', isEnabled: true, type: 'brightness-contrast' }])
    ).toBeNull();
    expect(cache.size()).toBe(0);
  });

  it('builds and reuses the adjusted surface while version + adjustments are unchanged', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));

    const first = cache.get('a', entry, BRIGHTEN);
    expect(first).not.toBeNull();
    const stub = first as StubRasterSurface;
    expect(drawImageCount(stub)).toBe(1);

    // Same version + key: reuse the same surface, no rebuild.
    const second = cache.get('a', entry, BRIGHTEN);
    expect(second).toBe(first);
    expect(drawImageCount(stub)).toBe(1);
    expect(cache.size()).toBe(1);
    expect(cache.byteSize()).toBe(10 * 10 * 4);
  });

  it('rebuilds when the adjustments value changes', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));

    const first = cache.get('a', entry, BRIGHTEN) as StubRasterSurface;
    expect(drawImageCount(first)).toBe(1);
    // Different adjustments → rebuild (draws again into the reused surface).
    cache.get('a', entry, CONTRAST);
    expect(drawImageCount(first)).toBe(2);
  });

  it('invalidates only the edited layer when its cache version bumps', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entryA = makeEntry('a', backend.createSurface(10, 10), 0);
    const entryB = makeEntry('b', backend.createSurface(10, 10), 0);

    const surfaceA = cache.get('a', entryA, BRIGHTEN) as StubRasterSurface;
    const surfaceB = cache.get('b', entryB, BRIGHTEN) as StubRasterSurface;
    expect(drawImageCount(surfaceA)).toBe(1);
    expect(drawImageCount(surfaceB)).toBe(1);

    // Layer 'a' is edited: its cache version bumps → its adjusted surface rebuilds.
    const editedA = makeEntry('a', surfaceA, 1);
    const rebuiltA = cache.get('a', editedA, BRIGHTEN) as StubRasterSurface;
    expect(rebuiltA).not.toBe(surfaceA);
    expect(drawImageCount(rebuiltA)).toBe(1);

    // Layer 'b' is untouched (same version) → reused, no rebuild.
    const reusedB = cache.get('b', entryB, BRIGHTEN);
    expect(reusedB).toBe(surfaceB);
    expect(drawImageCount(surfaceB)).toBe(1);
  });

  it('drops a layer slot on delete and reverting to identity', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));
    cache.get('a', entry, BRIGHTEN);
    expect(cache.size()).toBe(1);
    // Reverting to identity clears the slot.
    expect(cache.get('a', entry, undefined)).toBeNull();
    expect(cache.size()).toBe(0);
    // Re-cache then delete.
    cache.get('a', entry, BRIGHTEN);
    cache.delete('a');
    expect(cache.size()).toBe(0);
    expect(cache.byteSize()).toBe(0);
  });
});

describe('spatial stacks', () => {
  const W = 8;
  const H = 6;
  const clarity = (amount: number): CanvasAdjustmentsContract => [
    { amount, id: 'cl', isEnabled: true, size: 40, type: 'clarity' },
  ];

  /** An executor whose jobs settle when the test says; an `ok` job's pixels all become its 1-based job number. */
  const createManualExecutor = () => {
    const jobs: { steps: SpatialExecutorJob['steps']; settle: (status: SpatialExecutorResult['status']) => void }[] =
      [];
    const executor: SpatialAdjustmentExecutor = {
      dispose: () => undefined,
      run: (job) =>
        new Promise((resolve) => {
          const tag = jobs.length + 1;
          jobs.push({
            settle: (status) => {
              new Uint8ClampedArray(job.buffer).fill(status === 'ok' ? tag : 0);
              resolve({ buffer: job.buffer, status });
            },
            steps: job.steps,
          });
        }),
    };
    return { executor, jobs };
  };

  const flush = async () => {
    for (let i = 0; i < 10; i++) {
      await Promise.resolve();
    }
  };

  const setup = (options: { executor?: SpatialAdjustmentExecutor; overBudget?: boolean } = {}) => {
    const manual = createManualExecutor();
    const backend = createTestStubRasterBackend({ readbackAlpha: 255 });
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', W, H);
    layers.publishPixels('L');
    const derived = createDerivedSurfaceCache();
    const service = createSpatialAdjustmentService({
      createExecutor: () => options.executor ?? manual.executor,
      reserve: (bytes) =>
        options.overBudget
          ? { availableBytes: 0, requestedBytes: bytes, status: 'over-budget' }
          : { lease: { release: () => undefined }, status: 'ok' },
    });
    const settled: string[] = [];
    const statuses: (AdjustedSurfaceStatus | null)[] = [];
    const cache = createAdjustedSurfaceCache(backend, derived, {
      damageSince: (layerId, version) => layers.damageSince(layerId, version),
      spatial: {
        onSettle: (layerId) => settled.push(layerId),
        onStatus: (_layerId, status) => statuses.push(status),
        run: (request) => service.run(request),
      },
    });
    const entry = () => layers.get('L')!;
    const show = (adjustments: CanvasAdjustmentsContract) => cache.get('L', entry(), adjustments) as StubRasterSurface;
    return { cache, derived, entry, jobs: manual.jobs, layers, service, settled, show, statuses };
  };

  /** The first byte of the last whole-surface write: 0 for a per-pixel interim, a job number for a result. */
  const shownTag = (surface: StubRasterSurface): number => {
    const writes = surface.callLog.filter(
      (call) => call.op === 'putImageData' && (call.args[0] as ImageData).width === W && call.args[1] === 0
    );
    return (writes[writes.length - 1]!.args[0] as ImageData).data[0]!;
  };

  const sourceReads = (surface: StubRasterSurface) =>
    surface.callLog.filter((call) => call.op === 'drawImage').map((call) => call.args.slice(1, 5));

  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows a per-pixel interim at once, then the landed result, exactly and without further work', async () => {
    const h = setup();
    const surface = h.show(clarity(50));

    expect(shownTag(surface)).toBe(0);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBeNull();
    expect(h.cache.peekDisplayed('L', h.entry(), clarity(50))).toBe(surface);
    expect(h.statuses).toEqual(['computing']);
    expect(h.jobs.map((job) => job.steps)).toEqual([[{ amount: 50, kind: 'clarity', size: 40 }]]);

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['L']);
    expect(h.statuses).toEqual(['computing', null]);
    expect(shownTag(surface)).toBe(1);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(surface);

    const calls = surface.callLog.length;
    expect(h.show(clarity(50))).toBe(surface);
    expect(surface.callLog).toHaveLength(calls);
    expect(h.jobs).toHaveLength(1);
  });

  it('keeps the shown pixels through a parameter change and starts its job at once', async () => {
    const h = setup();
    const surface = h.show(clarity(50));
    h.jobs[0]!.settle('ok');
    await flush();
    const reads = sourceReads(surface).length;

    expect(h.show(clarity(80))).toBe(surface);
    expect(sourceReads(surface)).toHaveLength(reads);
    expect(shownTag(surface)).toBe(1);
    expect(h.cache.peek('L', h.entry(), clarity(80))).toBeNull();
    expect(h.jobs).toHaveLength(2);

    h.jobs[1]!.settle('ok');
    await flush();
    expect(shownTag(surface)).toBe(2);
    expect(h.cache.peek('L', h.entry(), clarity(80))).toBe(surface);
  });

  it('re-patches known damage per-pixel and starts a job only after 120 ms without edits', async () => {
    vi.useFakeTimers();
    const h = setup();
    const surface = h.show(clarity(50));
    h.jobs[0]!.settle('ok');
    await flush();

    h.layers.publishPixels('L', { height: 2, width: 3, x: 1, y: 1 });
    h.show(clarity(50));
    expect(sourceReads(surface).at(-1)).toEqual([1, 1, 3, 2]);
    expect(shownTag(surface)).toBe(1);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBeNull();
    expect(h.statuses.at(-1)).toBe('computing');

    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS - 1);
    h.layers.publishPixels('L', { height: 1, width: 1, x: 5, y: 4 });
    h.show(clarity(50));
    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS - 1);
    expect(h.jobs).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(h.jobs).toHaveLength(2);

    h.jobs[1]!.settle('ok');
    await flush();
    expect(shownTag(surface)).toBe(2);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(surface);
  });

  it('falls back to a whole per-pixel interim after a write of unknown extent', async () => {
    vi.useFakeTimers();
    const h = setup();
    const surface = h.show(clarity(50));
    h.jobs[0]!.settle('ok');
    await flush();

    h.layers.publishPixels('L');
    h.show(clarity(50));
    expect(sourceReads(surface).at(-1)).toEqual([0, 0, W, H]);
    expect(shownTag(surface)).toBe(0);
    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
    expect(h.jobs).toHaveLength(2);
  });

  it('runs one job per layer while scrubbing: the running result lands as an interim, then the latest', async () => {
    const h = setup();
    const surface = h.show(clarity(10));
    h.show(clarity(20));
    h.show(clarity(30));
    expect(h.jobs).toHaveLength(1);

    h.jobs[0]!.settle('ok');
    await flush();
    expect(shownTag(surface)).toBe(1);
    expect(h.settled).toEqual(['L']);
    expect(h.cache.peek('L', h.entry(), clarity(30))).toBeNull();
    expect(h.jobs.map((job) => job.steps[0]!.amount)).toEqual([10, 30]);

    h.jobs[1]!.settle('ok');
    await flush();
    expect(shownTag(surface)).toBe(2);
    expect(h.cache.peek('L', h.entry(), clarity(30))).toBe(surface);
  });

  it('lands a result from before a known edit, re-patching that edit, and stays inexact until the next job', async () => {
    vi.useFakeTimers();
    const h = setup();
    const surface = h.show(clarity(50));
    h.layers.publishPixels('L', { height: 2, width: 2, x: 4, y: 2 });
    h.show(clarity(50));

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['L']);
    expect(shownTag(surface)).toBe(1);
    expect(sourceReads(surface).at(-1)).toEqual([4, 2, 2, 2]);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBeNull();
    expect(h.statuses.at(-1)).toBe('computing');

    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
    h.jobs[1]!.settle('ok');
    await flush();
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(surface);
  });

  it('discards a result whose source changed in ways it cannot re-patch', async () => {
    vi.useFakeTimers();
    const h = setup();
    const surface = h.show(clarity(50));
    h.layers.publishPixels('L');
    h.show(clarity(50));

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual([]);
    expect(shownTag(surface)).toBe(0);
    expect(h.statuses.at(-1)).toBe('computing');

    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
    h.jobs[1]!.settle('ok');
    await flush();
    expect(shownTag(surface)).toBe(2);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(surface);
  });

  it('is exact again on returning to the shown parameters, abandoning the job for the others', async () => {
    const h = setup();
    const surface = h.show(clarity(50));
    h.jobs[0]!.settle('ok');
    await flush();
    h.show(clarity(80));
    expect(h.statuses.at(-1)).toBe('computing');

    h.show(clarity(50));
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(surface);
    expect(h.statuses.at(-1)).toBeNull();
    h.jobs[1]!.settle('ok');
    await flush();
    expect(shownTag(surface)).toBe(1);
    expect(h.settled).toEqual(['L']);
  });

  it('rebuilds an evicted surface as an interim and recomputes it; a result for an evicted surface is dropped', async () => {
    const h = setup();
    h.show(clarity(50));
    h.derived.evict(0, () => false);
    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual([]);
    expect(h.statuses.at(-1)).toBeNull();

    const rebuilt = h.show(clarity(50));
    expect(shownTag(rebuilt)).toBe(0);
    expect(h.statuses.at(-1)).toBe('computing');
    h.jobs[1]!.settle('ok');
    await flush();
    expect(shownTag(rebuilt)).toBe(2);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(rebuilt);
  });

  it('reports too-large, error and over-budget, keeps the interim and retries only after a change', async () => {
    vi.useFakeTimers();
    const h = setup();
    const surface = h.show(clarity(50));
    h.jobs[0]!.settle('too-large');
    await flush();
    expect(h.statuses.at(-1)).toBe('too-large');
    expect(shownTag(surface)).toBe(0);
    h.show(clarity(50));
    expect(h.jobs).toHaveLength(1);

    // Too large depends on the content alone, so scrubbing the stack never reads the source back for another job.
    const reads = sourceReads(surface).length;
    h.show(clarity(60));
    h.show(clarity(70));
    expect(h.jobs).toHaveLength(1);
    expect(sourceReads(surface)).toHaveLength(reads);
    expect(h.statuses.at(-1)).toBe('too-large');

    // New content may fit.
    h.layers.publishPixels('L');
    h.show(clarity(70));
    expect(h.statuses.at(-1)).toBe('computing');
    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
    expect(h.jobs).toHaveLength(2);
    h.jobs[1]!.settle('error');
    await flush();
    expect(h.statuses.at(-1)).toBe('error');
    expect(h.settled).toEqual([]);
    // An error belongs to its stack: a parameter change retries at once.
    h.show(clarity(80));
    expect(h.jobs).toHaveLength(3);

    const starved = setup({ overBudget: true });
    starved.show(clarity(50));
    await flush();
    expect(starved.statuses).toEqual(['computing', 'over-budget']);
    expect(starved.jobs).toHaveLength(0);
  });

  it('shows only the per-pixel part of a stack that cannot run, repainting it when that part changes', async () => {
    const h = setup();
    const stack = (brightness: number, amount = 50): CanvasAdjustmentsContract => [
      { brightness, contrast: 0, id: 'b', isEnabled: true, type: 'brightness-contrast' },
      ...clarity(amount),
    ];
    const surface = h.show(stack(0.2));
    // Edited while the job runs: the shown pixels wait for a result that will not come.
    h.show(stack(0.4));
    const reads = sourceReads(surface).length;
    h.jobs[0]!.settle('too-large');
    await flush();
    expect(h.jobs).toHaveLength(1);
    expect(sourceReads(surface).slice(reads)).toEqual([[0, 0, W, H]]);
    expect(h.settled).toEqual(['L']);

    // A per-pixel edit repaints at once; a spatial one changes nothing.
    h.show(stack(0.6));
    expect(sourceReads(surface).slice(reads)).toEqual([
      [0, 0, W, H],
      [0, 0, W, H],
    ]);
    h.show(stack(0.6, 90));
    expect(sourceReads(surface)).toHaveLength(reads + 2);
    expect(h.jobs).toHaveLength(1);
    expect(h.statuses.at(-1)).toBe('too-large');
  });

  it('keeps exact surfaces through a cancel and drops the interims of the jobs it abandons', async () => {
    const h = setup();
    h.layers.getOrCreate('M', W, H);
    h.layers.publishPixels('M');
    const exact = h.show(clarity(50));
    h.jobs[0]!.settle('ok');
    await flush();
    h.cache.get('M', h.layers.get('M')!, clarity(50));
    expect(h.statuses.at(-1)).toBe('computing');

    h.cache.cancelSpatial();
    expect(h.statuses.at(-1)).toBeNull();
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(exact);
    expect(h.cache.peekDisplayed('M', h.layers.get('M')!, clarity(50))).toBeNull();
    h.jobs[1]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['L']);

    expect(h.show(clarity(50))).toBe(exact);
    expect(h.jobs).toHaveLength(2);
  });

  it('drops the spatial state of a layer its owner no longer keeps, though nothing draws it', async () => {
    const h = setup();
    h.show(clarity(50));
    h.jobs[0]!.settle('too-large');
    await flush();

    h.cache.retainSpatial(() => true);
    expect(h.statuses.at(-1)).toBe('too-large');
    h.cache.retainSpatial((layerId) => layerId !== 'L');
    expect(h.statuses.at(-1)).toBeNull();
    expect(h.cache.size()).toBe(0);
  });

  it('abandons spatial work when the layer is dropped or its stack loses its spatial entries', async () => {
    const h = setup();
    h.show(clarity(50));
    h.cache.delete('L');
    expect(h.statuses).toEqual(['computing', null]);
    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual([]);
    expect(h.cache.size()).toBe(0);

    h.show(clarity(50));
    const perPixel = h.show([{ brightness: 0.3, contrast: 0, id: 'b', isEnabled: true, type: 'brightness-contrast' }]);
    expect(h.statuses.at(-1)).toBeNull();
    expect(
      h.cache.peek('L', h.entry(), [
        { brightness: 0.3, contrast: 0, id: 'b', isEnabled: true, type: 'brightness-contrast' },
      ])
    ).toBe(perPixel);
  });

  it('starts no job for a placeholder cache, and one at once for its first published pixels', () => {
    const h = setup();
    h.layers.delete('L');
    h.layers.getOrCreate('L', W, H);
    h.show(clarity(50));
    expect(h.jobs).toHaveLength(0);
    expect(h.statuses).toEqual([]);

    h.layers.publishPixels('L');
    h.show(clarity(50));
    expect(h.jobs).toHaveLength(1);
    expect(h.statuses).toEqual(['computing']);
  });

  it('settles to the kernel result through the real service and inline executor', async () => {
    const h = setup({ executor: createInlineSpatialExecutor() });
    const surface = h.show(clarity(50));
    await flush();
    expect(h.settled).toEqual(['L']);
    expect(h.cache.peek('L', h.entry(), clarity(50))).toBe(surface);
  });
});
