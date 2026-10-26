import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type {
  SpatialAdjustmentExecutor,
  SpatialExecutorJob,
  SpatialExecutorResult,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { GroupCompositeScope, GroupSurfaceContent } from './groupCompositeScopes';
import type { StubRasterSurface } from './raster.testStub';
import type { AdjustedSurfaceStatus, DisplayedPixelsState } from './spatialDisplay';

import { createGroupSurfaceCache } from './groupSurfaceCache';
import { createLayerCacheStore } from './layerCache';
import { createTestStubRasterBackend } from './raster.testStub';
import { SPATIAL_CONTENT_DEBOUNCE_MS } from './spatialDisplay';

const W = 8;
const H = 6;
const NO_CONTENT: GroupSurfaceContent = { excludeIds: new Set(), float: null, previews: null };

const clarity = (amount: number): CanvasAdjustmentsContract => [
  { amount, id: 'cl', isEnabled: true, size: 40, type: 'clarity' },
];

const leaf = (id: string, layer: Partial<CanvasRasterLayerContractV2> = {}) =>
  ({
    id,
    layer: {
      adjustments: [],
      blendMode: 'normal',
      id,
      opacity: 1,
      type: 'raster',
      ...layer,
    } as unknown as CanvasRasterLayerContractV2,
  }) as unknown as SemanticLeaf;

const scope = (
  id: string,
  adjustments: CanvasAdjustmentsContract,
  children: readonly GroupCompositeScope[] = []
): GroupCompositeScope => ({ adjustments, blendMode: 'normal', children, end: 1, id, opacity: 1, start: 0 });

/** An executor whose jobs settle when the test says; an `ok` job's pixels all become its 1-based job number. */
const createManualExecutor = () => {
  const jobs: { settle: (status: SpatialExecutorResult['status']) => void; steps: SpatialExecutorJob['steps'] }[] = [];
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

/** Member `b`'s cache: inside `a`'s, so editing or nudging it leaves the group bounds alone. */
const B_RECT = { height: 2, width: 3, x: 2, y: 1 };

/** Member `a` in group `g`, plus `b` for `showPair`; nested setups put `inner` around `a` inside `outer`. */
const setup = (ancestors: Readonly<Record<string, readonly string[]>> = {}) => {
  const { executor, jobs } = createManualExecutor();
  const backend = createTestStubRasterBackend({ readbackAlpha: 255 });
  const layers = createLayerCacheStore(backend);
  layers.getOrCreateRect('a', { height: H, width: W, x: 0, y: 0 });
  layers.publishPixels('a');
  layers.getOrCreateRect('b', B_RECT);
  layers.publishPixels('b');
  const service = createSpatialAdjustmentService({
    createExecutor: () => executor,
    reserve: () => ({ lease: { release: () => undefined }, status: 'ok' }),
  });
  const diagnostics = createCanvasDiagnostics(true);
  const memberStates = new Map<string, DisplayedPixelsState>();
  /** The stack each member was last read with. */
  const memberStacks = new Map<string, CanvasAdjustmentsContract | undefined>();
  const settled: string[] = [];
  const statuses: [string, AdjustedSurfaceStatus | null][] = [];
  const surfaces: StubRasterSurface[] = [];
  const groups = createGroupSurfaceCache({
    createSurface: (width, height) => {
      const surface = backend.createSurface(width, height) as StubRasterSurface;
      surfaces.push(surface);
      return surface;
    },
    damageSince: (layerId, version) => layers.damageSince(layerId, version),
    diagnostics,
    getAdjustedSurface: (layer) => {
      memberStacks.set(layer.id, layer.adjustments);
      return { state: memberStates.get(layer.id) ?? 'exact', surface: null };
    },
    getCacheEntry: (layerId) => layers.get(layerId),
    spatial: {
      // Mirrors the engine: a landing redraws the groups holding the owner, never the owner itself.
      onSettle: (groupId) => {
        settled.push(groupId);
        groups.invalidate(ancestors[groupId] ?? []);
      },
      onStatus: (groupId, status) => statuses.push([groupId, status]),
      run: (request) => service.run(request),
    },
  });
  const show = (target: GroupCompositeScope) => groups.get(target, [leaf('a')], [identity()], NO_CONTENT)!;
  /** Draws `a` under `b` in a two-member group. */
  const showPair = (stack: CanvasAdjustmentsContract, b: SemanticLeaf, bMatrix: Mat2d = identity()) =>
    groups.get({ ...scope('g', stack), end: 2 }, [leaf('a'), b], [identity(), bMatrix], NO_CONTENT)!;
  const rawPaints = () => {
    const { groupSurfaceRebuilds, groupSurfaceRefreshes } = diagnostics.snapshot();
    return groupSurfaceRebuilds + groupSurfaceRefreshes;
  };
  return {
    groups,
    jobs,
    layers,
    memberStacks,
    memberStates,
    rawPaints,
    settled,
    show,
    showPair,
    statuses,
    surfaces,
  };
};

/** The first byte of the last whole-surface write: 0 for a per-pixel interim, a job number for a result. */
const shownTag = (surface: StubRasterSurface): number => {
  const writes = surface.callLog.filter(
    (call) => call.op === 'putImageData' && (call.args[0] as ImageData).width === W && call.args[1] === 0
  );
  return (writes[writes.length - 1]!.args[0] as ImageData).data[0]!;
};

/** Source regions drawn into a surface, as [x, y, width, height]. */
const sourceReads = (surface: StubRasterSurface) =>
  surface.callLog
    .filter((call) => call.op === 'drawImage' && call.args.length === 9)
    .map((call) => call.args.slice(1, 5));

afterEach(() => {
  vi.useRealTimers();
});

describe('spatial group surfaces', () => {
  it('keeps the raw composite and shows the evaluated stack on a separate surface', async () => {
    const h = setup();
    const first = h.show(scope('g', clarity(50)));
    const [raw, adjusted] = h.surfaces;

    expect(first.surface).toBe(adjusted);
    expect(first.surface).not.toBe(raw);
    expect(first.state).toBe('pending');
    expect(shownTag(adjusted!)).toBe(0);
    expect(h.statuses).toEqual([['g', 'computing']]);
    expect(h.jobs.map((job) => job.steps)).toEqual([[{ amount: 50, kind: 'clarity', size: 40 }]]);
    expect(h.groups.byteSize()).toBe(2 * W * H * 4);

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['g']);
    expect(h.statuses.at(-1)).toEqual(['g', null]);
    const paints = h.rawPaints();
    const again = h.show(scope('g', clarity(50)));
    expect(again.state).toBe('exact');
    expect(again.surface).toBe(adjusted);
    expect(shownTag(adjusted!)).toBe(1);
    expect(h.rawPaints()).toBe(paints);
    expect(h.jobs).toHaveLength(1);
  });

  it('keeps the raw composite and the shown pixels through a parameter edit, starting its job at once', async () => {
    const h = setup();
    h.show(scope('g', clarity(50)));
    h.jobs[0]!.settle('ok');
    await flush();
    const adjusted = h.surfaces[1]!;
    const paints = h.rawPaints();
    const reads = sourceReads(adjusted).length;

    expect(h.show(scope('g', clarity(80))).state).toBe('pending');
    expect(h.rawPaints()).toBe(paints);
    expect(h.surfaces).toHaveLength(2);
    expect(sourceReads(adjusted)).toHaveLength(reads);
    expect(shownTag(adjusted)).toBe(1);
    expect(h.jobs).toHaveLength(2);

    h.jobs[1]!.settle('ok');
    await flush();
    expect(h.show(scope('g', clarity(80))).state).toBe('exact');
    expect(shownTag(adjusted)).toBe(2);
  });

  it('repaints damaged raw pixels, re-patches them per-pixel and re-runs the job after the content debounce', async () => {
    vi.useFakeTimers();
    const h = setup();
    h.show(scope('g', clarity(50)));
    h.jobs[0]!.settle('ok');
    await flush();
    const adjusted = h.surfaces[1]!;
    const paints = h.rawPaints();

    h.layers.publishPixels('a', { height: 1, width: 1, x: 3, y: 2 });
    expect(h.show(scope('g', clarity(50))).state).toBe('pending');
    expect(h.rawPaints()).toBe(paints + 1);
    // The member's one-pixel damage reaches one source pixel further, plus one pixel of antialiasing.
    expect(sourceReads(adjusted).at(-1)).toEqual([1, 0, 5, 5]);
    expect(shownTag(adjusted)).toBe(1);

    await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS - 1);
    expect(h.jobs).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(h.jobs).toHaveLength(2);
    h.jobs[1]!.settle('ok');
    await flush();
    expect(h.show(scope('g', clarity(50))).state).toBe('exact');
    expect(shownTag(adjusted)).toBe(2);
  });

  it('waits for members still computing, then starts at once and keeps its pixels while their results land', async () => {
    const h = setup();
    h.memberStates.set('a', 'pending');
    expect(h.show(scope('g', clarity(50))).state).toBe('pending');
    expect(h.statuses).toEqual([['g', 'computing']]);
    expect(h.jobs).toHaveLength(0);
    const adjusted = h.surfaces[1]!;
    const reads = sourceReads(adjusted).length;

    // The member's result landed: the engine redraws the group without any member version change.
    h.memberStates.set('a', 'exact');
    h.groups.invalidate(['g']);
    expect(h.show(scope('g', clarity(50))).state).toBe('pending');
    expect(sourceReads(adjusted)).toHaveLength(reads);
    expect(h.jobs).toHaveLength(1);

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.show(scope('g', clarity(50))).state).toBe('exact');
  });

  it('runs on a member that cannot finish, reporting the composite as inexact', async () => {
    const h = setup();
    h.memberStates.set('a', 'inexact');
    h.show(scope('g', clarity(50)));
    expect(h.jobs).toHaveLength(1);
    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.show(scope('g', clarity(50))).state).toBe('inexact');
    expect(h.statuses.at(-1)).toEqual(['g', null]);
  });

  it('converges nested spatial groups: the outer job waits for the inner result', async () => {
    const h = setup({ inner: ['outer'] });
    const outer = scope('outer', clarity(30), [scope('inner', clarity(60))]);

    expect(h.show(outer).state).toBe('pending');
    expect(h.jobs.map((job) => job.steps[0]!.amount)).toEqual([60]);
    expect(h.statuses).toContainEqual(['outer', 'computing']);

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['inner']);
    expect(h.show(outer).state).toBe('pending');
    expect(h.jobs.map((job) => job.steps[0]!.amount)).toEqual([60, 30]);

    h.jobs[1]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['inner', 'outer']);
    expect(h.show(outer).state).toBe('exact');
    expect(h.jobs).toHaveLength(2);
    expect(h.statuses.filter(([, status]) => status === null).map(([id]) => id)).toEqual(['inner', 'outer']);
  });

  it('evicts the adjusted surface with its slot and abandons the slot job', async () => {
    const h = setup();
    h.show(scope('g', clarity(50)));
    h.groups.evict(0, h.groups.tick());

    expect(h.groups.byteSize()).toBe(0);
    expect(h.statuses.at(-1)).toEqual(['g', null]);
    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual([]);
  });

  it('clears the status and releases the adjusted surface once the group stops drawing a spatial stack', async () => {
    const h = setup();
    h.show(scope('g', clarity(50)));
    h.jobs[0]!.settle('too-large');
    await flush();
    expect(h.statuses.at(-1)).toEqual(['g', 'too-large']);
    expect(h.show(scope('g', clarity(50))).state).toBe('inexact');
    expect(h.jobs).toHaveLength(1);

    // Edited to a per-pixel stack: the group reports the slot it now draws, which has no adjusted surface.
    const perPixel = h.show(scope('g', [{ id: 'inv', isEnabled: true, type: 'invert' }]));
    expect(perPixel.state).toBe('exact');
    expect(h.statuses.at(-1)).toEqual(['g', null]);
    expect(h.groups.byteSize()).toBe(3 * W * H * 4);

    // Undone: the warm spatial slot still knows its target cannot run.
    h.show(scope('g', clarity(50)));
    expect(h.statuses.at(-1)).toEqual(['g', 'too-large']);
    // No longer drawn at all (disabled, or its stack reset): only `retainSpatial` can tell.
    h.groups.retainSpatial(() => false);
    expect(h.statuses.at(-1)).toEqual(['g', null]);
    expect(h.groups.byteSize()).toBe(W * H * 4);
  });

  it("reports the frame's slot, so another consumer's key never flips the group's status", async () => {
    const h = setup();
    const preview = {
      rect: { height: H, width: W, x: 0, y: 0 },
      surface: createTestStubRasterBackend().createSurface(W, H),
    };
    const previewed: GroupSurfaceContent = { ...NO_CONTENT, previews: new Map([['a', preview]]) };
    const target = scope('g', clarity(50));
    const frame = () => h.groups.get(target, [leaf('a')], [identity()], previewed, { frame: true })!;

    // The frame shows a filter preview on `a`; the overview composites the committed pixels in another slot.
    frame();
    h.show(target);
    expect(h.statuses).toEqual([['g', 'computing']]);
    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.statuses.at(-1)).toEqual(['g', null]);

    h.show(target);
    frame();
    h.show(target);
    expect(h.show(target).state).toBe('pending');
    h.jobs[1]!.settle('ok');
    await flush();
    expect(h.statuses).toEqual([
      ['g', 'computing'],
      ['g', null],
    ]);
  });

  it('re-patches a result that lands after more raw repaints than the trail records apart', async () => {
    vi.useFakeTimers();
    const h = setup();
    h.show(scope('g', clarity(50)));
    const adjusted = h.surfaces[1]!;
    for (let i = 0; i < 20; i++) {
      h.layers.publishPixels('a', { height: 1, width: 1, x: 3 + (i % 2), y: 2 });
      h.show(scope('g', clarity(50)));
    }

    h.jobs[0]!.settle('ok');
    await flush();
    expect(h.settled).toEqual(['g']);
    expect(shownTag(adjusted)).toBe(1);
  });

  it('keeps an exact group through a cancel; one still computing shows a whole interim and starts again', async () => {
    const h = setup();
    h.show(scope('g', clarity(50)));
    h.jobs[0]!.settle('ok');
    await flush();
    h.groups.cancelSpatial();
    expect(h.show(scope('g', clarity(50))).state).toBe('exact');
    expect(h.jobs).toHaveLength(1);

    h.show(scope('g', clarity(60)));
    h.groups.cancelSpatial();
    expect(h.statuses.at(-1)).toEqual(['g', null]);
    const adjusted = h.surfaces[1]!;
    expect(h.show(scope('g', clarity(60))).state).toBe('pending');
    expect(shownTag(adjusted)).toBe(0);
    h.jobs[1]!.settle('ok');
    await flush();
    h.jobs[2]!.settle('ok');
    await flush();
    expect(h.show(scope('g', clarity(60))).state).toBe('exact');
    expect(shownTag(adjusted)).toBe(3);
  });

  describe('member looks under a spatial group', () => {
    const brighten = (brightness: number): CanvasAdjustmentsContract => [
      { brightness, contrast: 0, id: 'bc', isEnabled: true, type: 'brightness-contrast' },
    ];
    /** `b`'s bounds plus one pixel of antialiasing, clamped to the group: what its edits repaint. */
    const B_REGION = [1, 0, 5, 4];

    const settledPair = async (b: SemanticLeaf = leaf('b')) => {
      const h = setup();
      h.showPair(clarity(50), b);
      h.jobs[0]!.settle('ok');
      await flush();
      expect(h.showPair(clarity(50), b).state).toBe('exact');
      return { ...h, adjusted: h.surfaces[1]!, paints: h.rawPaints() };
    };

    it('repaints only an edited member, keeps the adjusted pixels elsewhere and is exact only after a later job', async () => {
      vi.useFakeTimers();
      const h = await settledPair();

      expect(h.showPair(clarity(50), leaf('b', { adjustments: brighten(0.2) })).state).toBe('pending');
      expect(h.surfaces).toHaveLength(2);
      expect(h.rawPaints()).toBe(h.paints + 1);
      expect(sourceReads(h.adjusted).at(-1)).toEqual(B_REGION);
      expect(shownTag(h.adjusted)).toBe(1);
      expect(h.jobs).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
      expect(h.jobs).toHaveLength(2);

      // Changed again while that job runs: its result lands as an interim, never as exact.
      const faded = leaf('b', { adjustments: brighten(0.2), opacity: 0.5 });
      h.showPair(clarity(50), faded);
      expect(sourceReads(h.adjusted).at(-1)).toEqual(B_REGION);
      h.jobs[1]!.settle('ok');
      await flush();
      expect(h.showPair(clarity(50), faded).state).toBe('pending');
      expect(shownTag(h.adjusted)).toBe(2);

      await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
      h.jobs[2]!.settle('ok');
      await flush();
      expect(h.showPair(clarity(50), faded).state).toBe('exact');
      expect(h.surfaces).toHaveLength(2);
    });

    it('refreshes a member moved within the group bounds by its old and new bounds', async () => {
      vi.useFakeTimers();
      const h = await settledPair();
      const nudged: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: 1, f: 0 };

      expect(h.showPair(clarity(50), leaf('b'), nudged).state).toBe('pending');
      expect(h.surfaces).toHaveLength(2);
      expect(sourceReads(h.adjusted).at(-1)).toEqual([1, 0, 6, 4]);
      expect(shownTag(h.adjusted)).toBe(1);

      await vi.advanceTimersByTimeAsync(SPATIAL_CONTENT_DEBOUNCE_MS);
      h.jobs[1]!.settle('ok');
      await flush();
      expect(h.showPair(clarity(50), leaf('b'), nudged).state).toBe('exact');
    });

    it("keeps the shown pixels through a spatial member's edit, rereading the member until its result lands", async () => {
      const h = await settledPair(leaf('b', { adjustments: clarity(10) }));
      const reads = sourceReads(h.adjusted).length;

      h.memberStates.set('b', 'pending');
      expect(h.showPair(clarity(50), leaf('b', { adjustments: clarity(20) })).state).toBe('pending');
      expect(h.memberStacks.get('b')).toEqual(clarity(20));
      expect(h.rawPaints()).toBe(h.paints + 1);
      expect(sourceReads(h.adjusted)).toHaveLength(reads);
      expect(shownTag(h.adjusted)).toBe(1);
      expect(h.jobs).toHaveLength(1);

      h.memberStates.set('b', 'exact');
      h.groups.invalidate(['g']);
      h.showPair(clarity(50), leaf('b', { adjustments: clarity(20) }));
      expect(sourceReads(h.adjusted)).toHaveLength(reads);
      expect(h.jobs).toHaveLength(2);
      h.jobs[1]!.settle('ok');
      await flush();
      expect(h.showPair(clarity(50), leaf('b', { adjustments: clarity(20) })).state).toBe('exact');
    });

    it("keeps the outer group's pixels through a nested spatial group's edit", async () => {
      const h = setup({ inner: ['outer'] });
      const nested = (amount: number) => scope('outer', clarity(30), [scope('inner', clarity(amount))]);
      h.show(nested(60));
      h.jobs[0]!.settle('ok');
      await flush();
      h.show(nested(60));
      h.jobs[1]!.settle('ok');
      await flush();
      expect(h.show(nested(60)).state).toBe('exact');
      const outerAdjusted = h.surfaces[3]!;
      const reads = sourceReads(outerAdjusted).length;

      expect(h.show(nested(70)).state).toBe('pending');
      expect(h.surfaces).toHaveLength(4);
      expect(sourceReads(outerAdjusted)).toHaveLength(reads);
      expect(shownTag(outerAdjusted)).toBe(2);
      expect(h.jobs.map((job) => job.steps[0]!.amount)).toEqual([60, 30, 70]);

      h.jobs[2]!.settle('ok');
      await flush();
      h.show(nested(70));
      h.jobs[3]!.settle('ok');
      await flush();
      expect(h.show(nested(70)).state).toBe('exact');
      expect(sourceReads(outerAdjusted)).toHaveLength(reads);
    });
  });
});
