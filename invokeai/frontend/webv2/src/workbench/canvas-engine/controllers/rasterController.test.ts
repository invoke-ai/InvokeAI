import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { GroupCompositeScope, GroupSurfaceContent } from '@workbench/canvas-engine/render/groupCompositeScopes';

import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { createBitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
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

    const shared = controller.getAdjustedCacheSurface('layer', entry.surface, adjustments!);
    expect(shared).not.toBeNull();
    expect(shared).toBe(controller.getAdjustedSurface(adjustedLayer('layer'), entry));
    expect(
      controller.getAdjustedCacheSurface('layer', createTestStubRasterBackend().createSurface(1, 1), adjustments!)
    ).toBeNull();
    baked = true;
    expect(controller.getAdjustedCacheSurface('layer', entry.surface, adjustments!)).toBeNull();
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
