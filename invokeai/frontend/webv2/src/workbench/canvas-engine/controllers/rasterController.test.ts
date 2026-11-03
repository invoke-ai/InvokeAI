import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { GroupCompositeScope, GroupSurfaceContent } from '@workbench/canvas-engine/render/groupCompositeScopes';

import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { createBitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { adjustExactly } from '@workbench/canvas-engine/render/exactAdjustments';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createInlineSpatialExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService.testing';
import { describe, expect, it, vi } from 'vitest';

import { RasterController } from './rasterController';

const NO_GROUP_CONTENT: GroupSurfaceContent = { excludeIds: new Set(), float: null, previews: null };

const RECT = { height: 10, width: 10, x: 0, y: 0 };
const SURFACE_BYTES = 400;

const adjustedLayer = (id: string) =>
  ({
    adjustments: [
      { brightness: 0.5, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
    ],
    blendMode: 'normal',
    id,
    opacity: 1,
    type: 'raster',
  }) as unknown as CanvasRasterLayerContractV2;

const groupScope: GroupCompositeScope = {
  adjustments: [],
  blendMode: 'normal',
  children: [],
  end: 1,
  id: 'group',
  opacity: 0.5,
  start: 0,
};

const createController = (budgetBytes: number, held: ReadonlySet<string> = new Set()) =>
  new RasterController({
    backend: createTestStubRasterBackend(),
    budgetBytes,
    diagnostics: createCanvasDiagnostics(true),
    isLayerHeld: (layerId) => held.has(layerId),
  });

const publish = (controller: RasterController, layerId: string) => {
  controller.layers.getOrCreateRect(layerId, RECT);
  return controller.layers.publishPixels(layerId)!;
};

describe('RasterController', () => {
  it("shares a live cache's adjusted copy with export, never a detached copy's or baked pixels'", () => {
    let baked = false;
    const controller = new RasterController({
      backend: createTestStubRasterBackend(),
      diagnostics: createCanvasDiagnostics(true),
      isAdjustmentBaked: () => baked,
    });
    const entry = publish(controller, 'layer');
    const { adjustments } = adjustedLayer('layer');

    // Export only reuses the display's copy; it never builds one outside its reservation.
    expect(controller.peekAdjustedCacheSurface('layer', entry.surface, adjustments!)).toBeNull();
    expect(controller.memory.snapshot().derivedBytes).toBe(0);
    const displayed = controller.getAdjustedSurface(adjustedLayer('layer'), entry);
    expect(displayed).not.toBeNull();
    expect(controller.peekAdjustedCacheSurface('layer', entry.surface, adjustments!)).toBe(displayed);
    expect(
      controller.peekAdjustedCacheSurface('layer', createTestStubRasterBackend().createSurface(1, 1), adjustments!)
    ).toBeNull();
    // A copy made for other adjustment parameters is not this stack's.
    expect(
      controller.peekAdjustedCacheSurface('layer', entry.surface, [
        { brightness: 0.25, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' },
      ])
    ).toBeNull();
    baked = true;
    expect(controller.peekAdjustedCacheSurface('layer', entry.surface, adjustments!)).toBeNull();
    baked = false;
    expect(controller.peekAdjustedCacheSurface('layer', entry.surface, adjustments!)).toBe(displayed);
    // New source pixels make the copy stale; export adjusts its own sampled region instead of drawing it.
    controller.layers.publishPixels('layer');
    expect(controller.peekAdjustedCacheSurface('layer', entry.surface, adjustments!)).toBeNull();
  });

  it('accounts base, derived and group surfaces as they are allocated and released', () => {
    const controller = createController(10_000);
    const entry = publish(controller, 'a');
    controller.getAdjustedSurface(adjustedLayer('a'), entry);
    controller.groups.get(
      groupScope,
      [{ id: 'a', layer: adjustedLayer('a') } as unknown as SemanticLeaf],
      [identity()],
      NO_GROUP_CONTENT
    );

    expect(controller.memory.snapshot()).toMatchObject({
      baseBytes: SURFACE_BYTES,
      derivedBytes: SURFACE_BYTES,
      groupBytes: SURFACE_BYTES,
      totalBytes: SURFACE_BYTES * 3,
    });

    controller.deleteDerivedSurfaces('a');
    controller.layers.delete('a');
    expect(controller.memory.snapshot()).toMatchObject({ baseBytes: 0, derivedBytes: 0 });
    controller.dispose();
    controller.dispose();
    expect(controller.memory.snapshot().groupBytes).toBe(0);
  });

  it('reclaims unused derived variants and hidden caches before the working set', () => {
    const controller = createController(SURFACE_BYTES * 2);
    const visible = publish(controller, 'visible');
    publish(controller, 'hidden');
    controller.getAdjustedSurface(adjustedLayer('hidden'), controller.layers.peek('hidden')!);
    const usage = controller.beginFrame();
    controller.getAdjustedSurface(adjustedLayer('visible'), visible);

    const result = controller.enforceBudget(new Set(['visible']), usage);

    expect(result).toEqual({ evictedBaseLayerIds: ['hidden'], overageBytes: 0 });
    expect(controller.layers.peek('visible')).toBe(visible);
    expect(controller.memory.snapshot()).toMatchObject({ baseBytes: SURFACE_BYTES, derivedBytes: SURFACE_BYTES });
  });

  it('keeps visible derived surfaces across frames instead of rebuilding them under pressure', () => {
    const controller = createController(SURFACE_BYTES);
    const create = vi.fn();
    const entry = publish(controller, 'visible');
    for (let frame = 0; frame < 5; frame += 1) {
      const usage = controller.beginFrame();
      controller.derived.get({
        create: () => {
          create();
          return createTestStubRasterBackend().createSurface(10, 10);
        },
        kind: 'adjustments',
        layerId: 'visible',
        paramsKey: 'same',
        source: entry.surface,
        sourceVersion: entry.version,
      });
      expect(controller.enforceBudget(new Set(['visible']), usage).overageBytes).toBe(SURFACE_BYTES);
    }

    expect(create).toHaveBeenCalledOnce();
  });

  it('never evicts held or pinned pixels and accounts them as overage', () => {
    const controller = createController(SURFACE_BYTES, new Set(['dirty']));
    publish(controller, 'dirty');
    publish(controller, 'pinned');
    publish(controller, 'hidden');
    const pin = controller.memory.pin('pinned');

    const result = controller.enforceBudget(new Set(), controller.beginFrame());

    expect(result).toEqual({ evictedBaseLayerIds: ['hidden'], overageBytes: SURFACE_BYTES });
    expect(controller.layers.peek('dirty')).toBeDefined();
    expect(controller.layers.peek('pinned')).toBeDefined();
    pin.release();
    expect(controller.enforceBudget(new Set(), controller.beginFrame()).evictedBaseLayerIds).toEqual(['pinned']);
  });

  it('keeps a drawn group composite across frames instead of rebuilding it under pressure', () => {
    const backend = createTestStubRasterBackend();
    const createSurface = vi.fn((width: number, height: number) => backend.createSurface(width, height));
    const controller = new RasterController({
      backend: { ...backend, createSurface },
      budgetBytes: SURFACE_BYTES,
      diagnostics: createCanvasDiagnostics(true),
    });
    publish(controller, 'member');
    const members = [
      { id: 'member', layer: { ...adjustedLayer('member'), adjustments: [] } } as unknown as SemanticLeaf,
    ];
    const builds = (): number => createSurface.mock.calls.filter(([width]) => width === 10).length;
    const before = builds();

    for (let frame = 0; frame < 5; frame += 1) {
      const usage = controller.beginFrame();
      controller.groups.get(groupScope, members, [identity()], NO_GROUP_CONTENT);
      expect(controller.enforceBudget(new Set(['member']), usage).overageBytes).toBe(SURFACE_BYTES);
    }

    expect(builds() - before).toBe(1);
  });

  it('counts reserved and detached bytes when deciding what to reclaim', () => {
    const controller = createController(SURFACE_BYTES * 2);
    publish(controller, 'hidden');
    const reservation = controller.memory.reserveOperation(1, { purpose: 'thumbnail' });
    const detached = controller.memory.trackDetached(SURFACE_BYTES);

    expect(controller.enforceBudget(new Set(), controller.beginFrame()).evictedBaseLayerIds).toEqual(['hidden']);
    detached.release();
    if (reservation.status === 'ok') {
      reservation.lease.release();
    }
  });

  it('evicts inactive group composites and releases them with the reconstructible caches', () => {
    const controller = createController(SURFACE_BYTES);
    publish(controller, 'member');
    const members = [{ id: 'member', layer: adjustedLayer('member') } as unknown as SemanticLeaf];
    controller.groups.get(groupScope, members, [identity()], NO_GROUP_CONTENT);

    controller.enforceBudget(new Set(['member']), controller.beginFrame());
    expect(controller.memory.snapshot().groupBytes).toBe(0);

    controller.groups.get(groupScope, members, [identity()], NO_GROUP_CONTENT);
    controller.releaseReconstructible();
    expect(controller.memory.snapshot()).toMatchObject({ baseBytes: 0, derivedBytes: 0, groupBytes: 0 });
  });

  it('keeps unpersisted paint resident through encoding, upload failure and later strokes', async () => {
    let failUpload = true;
    const uploadImage = vi.fn(() =>
      failUpload ? Promise.reject(new Error('offline')) : Promise.resolve({ height: 10, imageName: 'saved', width: 10 })
    );
    let controller!: RasterController;
    const store = createBitmapStore({
      dispatch: () => true,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'])),
      getLayerSource: () => ({ bitmap: null, type: 'paint' }),
      getLayerSurface: (layerId) => {
        const entry = controller.layers.peek(layerId);
        return entry ? { offset: { x: entry.rect.x, y: entry.rect.y }, surface: entry.surface } : null;
      },
      hashBlob: () => Promise.resolve(`hash-${uploadImage.mock.calls.length}`),
      maxUploadAttempts: 1,
      sleep: () => Promise.resolve(),
      timers: { clearTimeout: () => undefined, setTimeout: () => 0 },
      uploadImage,
    });
    controller = new RasterController({
      backend: createTestStubRasterBackend(),
      budgetBytes: 0,
      diagnostics: createCanvasDiagnostics(true),
      isLayerHeld: (layerId) => store.hasPendingWork(layerId),
    });
    publish(controller, 'dirty');
    const evictOffscreen = () => controller.enforceBudget(new Set(), controller.beginFrame()).evictedBaseLayerIds;

    store.markLayerDirty('dirty');
    const failedFlush = store.flushPendingUploads();
    expect(evictOffscreen()).toEqual([]);
    await expect(failedFlush).rejects.toThrow('Canvas pixel persistence failed');
    expect(evictOffscreen()).toEqual([]);

    store.markLayerDirty('dirty');
    failUpload = false;
    expect(evictOffscreen()).toEqual([]);
    await store.flushPendingUploads();
    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(evictOffscreen()).toEqual(['dirty']);
    store.dispose();
  });
});

describe('RasterController spatial adjustments', () => {
  const spatialLayer = (id: string) =>
    ({
      adjustments: [{ amount: 50, id: 'adj-cl', isEnabled: true, size: 40, type: 'clarity' as const }],
      blendMode: 'normal',
      id,
      opacity: 1,
      type: 'raster',
    }) as unknown as CanvasRasterLayerContractV2;

  const flush = async () => {
    for (let i = 0; i < 10; i++) {
      await Promise.resolve();
    }
  };

  const createSpatialController = (budgetBytes = 1 << 20) => {
    const settled: string[] = [];
    const statuses: [string, string | null][] = [];
    const diagnostics = createCanvasDiagnostics(true);
    const controller = new RasterController({
      backend: createTestStubRasterBackend({ readbackAlpha: 255 }),
      budgetBytes,
      createSpatialExecutor: createInlineSpatialExecutor,
      diagnostics,
      onAdjustedSurfaceSettled: (layerId) => settled.push(layerId),
      onAdjustmentStatus: (layerId, status) => statuses.push([layerId, status]),
    });
    return { controller, diagnostics, settled, statuses };
  };

  it('redraws an invalidated group in full, since a landed result changes no member cache version', async () => {
    const { controller, diagnostics } = createSpatialController();
    const entry = publish(controller, 'a');
    const drawGroup = () =>
      controller.groups.get(
        groupScope,
        [{ id: 'a', layer: spatialLayer('a') } as unknown as SemanticLeaf],
        [identity()],
        NO_GROUP_CONTENT
      );
    const rebuilds = () => diagnostics.snapshot().groupSurfaceRebuilds;
    const allocations = () => diagnostics.snapshot().groupSurfaceAllocations;

    drawGroup();
    await flush();
    drawGroup();
    expect([allocations(), rebuilds()]).toEqual([1, 0]);

    controller.groups.invalidate(['group']);
    drawGroup();
    expect([allocations(), rebuilds()]).toEqual([1, 1]);
    expect(controller.peekAdjustedSurface(spatialLayer('a'), entry)).not.toBeNull();
  });

  it('reserves each job under the raster budget and settles to an exact copy export may reuse', async () => {
    const { controller, settled, statuses } = createSpatialController();
    const entry = publish(controller, 'layer');

    const shown = controller.getAdjustedSurface(spatialLayer('layer'), entry);
    expect(controller.memory.snapshot().reservedBytes).toBe(SURFACE_BYTES);
    expect(controller.peekAdjustedSurface(spatialLayer('layer'), entry)).toBeNull();
    expect(controller.peekDisplayedAdjustedSurface(spatialLayer('layer'), entry)).toBe(shown);

    await flush();
    expect(controller.memory.snapshot().reservedBytes).toBe(0);
    expect(settled).toEqual(['layer']);
    expect(statuses).toEqual([
      ['layer', 'computing'],
      ['layer', null],
    ]);
    expect(controller.peekAdjustedSurface(spatialLayer('layer'), entry)).toBe(shown);
  });

  it('reports over-budget when the job cannot reserve its pixels', async () => {
    const { controller, statuses } = createSpatialController(SURFACE_BYTES);
    const entry = publish(controller, 'layer');

    controller.getAdjustedSurface(spatialLayer('layer'), entry);
    await flush();
    expect(statuses.at(-1)).toEqual(['layer', 'over-budget']);
  });

  it("abandons only the display's work on cooldown: exact jobs and exact display copies survive", async () => {
    const { controller, settled, statuses } = createSpatialController();
    const exactEntry = publish(controller, 'exact');
    const exactShown = controller.getAdjustedSurface(spatialLayer('exact'), exactEntry);
    await flush();
    const pendingEntry = publish(controller, 'pending');
    controller.getAdjustedSurface(spatialLayer('pending'), pendingEntry);
    // A generation composite evaluating a detached snapshot, as a pixel-writing operation does.
    const composite = adjustExactly(
      { data: new Uint8ClampedArray(SURFACE_BYTES).fill(255), height: RECT.height, width: RECT.width },
      spatialLayer('detached').adjustments,
      RECT,
      { run: (request) => controller.spatial.run(request) }
    );

    controller.cancelSpatialAdjustments();
    expect(statuses.at(-1)).toEqual(['pending', null]);
    expect(controller.peekAdjustedSurface(spatialLayer('exact'), exactEntry)).toBe(exactShown);
    controller.releaseReconstructible();
    expect((await composite).status).toBe('ok');
    expect(settled).toEqual(['exact']);
  });

  it('abandons pending work on cooldown release and disposal, clearing each status', async () => {
    const { controller, settled, statuses } = createSpatialController();
    const entry = publish(controller, 'layer');

    controller.getAdjustedSurface(spatialLayer('layer'), entry);
    controller.releaseReconstructible();
    expect(statuses).toEqual([
      ['layer', 'computing'],
      ['layer', null],
    ]);
    await flush();
    expect(settled).toEqual([]);
    expect(controller.memory.snapshot().derivedBytes).toBe(0);

    const again = controller.layers.peek('layer') ?? publish(controller, 'layer');
    controller.getAdjustedSurface(spatialLayer('layer'), again);
    controller.dispose();
    expect(statuses.at(-1)).toEqual(['layer', null]);
    await flush();
    expect(settled).toEqual([]);
    await expect(
      controller.spatial.run({
        height: 1,
        pixels: new Uint8ClampedArray(4),
        priority: 'now',
        slotKey: 'late',
        steps: [],
        width: 1,
      })
    ).resolves.toEqual({ status: 'aborted' });
  });
});
