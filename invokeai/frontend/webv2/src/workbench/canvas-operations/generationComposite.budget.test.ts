import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasStateContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createRasterSnapshotCapture } from '@workbench/canvas-engine/rasterSnapshotCapture';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCompositeDedupeCache } from '@workbench/canvas-operations/compositeForGeneration';
import { composeForGeneration } from '@workbench/canvas-operations/generationComposite';
import { describe, it, expect, vi } from 'vitest';

const cases = [
  { width: 8192, height: 8192, bbox: 512 },
  { width: 8192, height: 8192, bbox: 1024 },
  { width: 12000, height: 8000, bbox: 512 },
  { width: 12000, height: 8000, bbox: 1024 },
];

describe('large map small bbox through real generation capture and memory budget', () => {
  it.each(cases)('$width x $height / bbox $bbox', async ({ width, height, bbox }) => {
    const backend = createTestStubRasterBackend();
    const memory = new RasterMemoryBudgetController();
    const layers = createLayerCacheStore(backend, { onBytesChange: (bytes) => memory.setCategoryBytes('base', bytes) });
    const shared = {
      blendMode: 'normal' as const,
      isEnabled: true,
      isLocked: false,
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    };
    const map: CanvasLayerContract = {
      ...shared,
      id: 'map',
      name: 'Map',
      source: { type: 'image', image: { width, height, imageName: 'map.png' } },
      type: 'raster',
    };
    const maskLayer: CanvasLayerContract = {
      ...shared,
      transform: { ...shared.transform, x: width / 2, y: height / 2 },
      id: 'mask',
      name: 'Mask',
      mask: {
        bitmap: { width: bbox, height: bbox, imageName: 'mask.png' },
        fill: { color: '#ff0000', style: 'solid' },
      },
      type: 'inpaint_mask',
    };
    const state = {
      document: {
        background: 'transparent',
        bbox: { x: width / 2, y: height / 2, width: bbox, height: bbox },
        width,
        height,
        stacks: stacksFrom([map, maskLayer]),
        selectedLayerId: 'map',
        version: 3,
      },
      documentRevision: 0,
      snapshots: [],
      stagingArea: {
        areThumbnailsVisible: true,
        autoSwitchMode: 'off',
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 3,
    } as CanvasStateContractV3;
    layers.getOrCreate('map', width, height);
    layers.getOrCreate('mask', bbox, bbox);
    // Steady-state caches: decoded images have already released their final lease.
    const initial = memory.snapshot();
    let peakBytes = initial.totalBytes;
    const allocations: number[] = [];
    const createSurface = (w: number, h: number) => {
      peakBytes = Math.max(peakBytes, memory.snapshot().totalBytes);
      allocations.push(w * h * 4);
      return backend.createSurface(w, h);
    };
    const capture = createRasterSnapshotCapture({
      createSurface,
      getCanvasState: () => state,
      getDirectPixelEpoch: () => 0,
      getDocumentGeneration: () => 0,
      getLifecycleGeneration: () => 0,
      isDisposed: () => false,
      isGuardCurrent: () => true,
      memory,
      rasterizeLayerPixels: (layerId) => {
        const entry = layers.get(layerId)!;
        return Promise.resolve({
          status: 'ok' as const,
          guard: { layerId } as LayerExportGuard,
          rect: entry.rect,
          surface: entry.surface,
          release() {},
        });
      },
    });
    const detectMode = vi.fn(() => 'inpaint' as const);
    const result = await composeForGeneration(
      {
        ...capture,
        dedupe: createCompositeDedupeCache(),
        getCompositeExecutorDeps: () => ({
          backend: { ...backend, createSurface },
          hashBlob: () => Promise.resolve('hash'),
          uploadImage: () => Promise.resolve({ imageName: 'composite.png', width: bbox, height: bbox }),
          reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'invocation-composite' }),
        }),
      },
      { signal: new AbortController().signal, detectMode }
    );
    expect(result.status).toBe('ok');
    expect(detectMode).toHaveBeenCalledWith(
      expect.objectContaining({ contentBounds: { x: 0, y: 0, width, height }, hasActiveInpaintMask: true })
    );
    expect(peakBytes).toBeLessThan(memory.budgetBytes);
    // Two detached inputs plus the mask compositor's output/readback and scratch/readback.
    expect(peakBytes - initial.totalBytes).toBeLessThanOrEqual(((bbox + 2) ** 2 + 5 * bbox ** 2) * 4);
    expect(Math.max(...allocations)).toBeLessThanOrEqual((bbox + 2) ** 2 * 4);
    expect(memory.snapshot().totalBytes).toBe(initial.totalBytes);
    expect(memory.isPinned('map')).toBe(false);
    expect(memory.isPinned('mask')).toBe(false);
    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(memory.snapshot().detachedBytes).toBe(0);
  });
});
