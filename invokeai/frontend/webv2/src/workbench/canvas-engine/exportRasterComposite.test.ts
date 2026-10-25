import type {
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { MEASURED_NOISY_TEXT } from '@workbench/canvas-engine/render/rasterComposite.testing';
import { describe, expect, it, vi } from 'vitest';

import type { ExportRasterCompositeDeps, RasterCompositeExportSnapshot } from './exportRasterComposite';

import { exportRasterComposite } from './exportRasterComposite';

const imageRef = (imageName: string, width = 64, height = 32): CanvasImageRef => ({ height, imageName, width });

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const rasterLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(id), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -4, y: 8 },
  type: 'raster',
});

const controlLayer = (id: string): CanvasLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(id), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: false,
});

const makeDoc = (layers: Parameters<typeof stacksFrom>[0]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 128, width: 128, x: 0, y: 0 },
  height: 128,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: 128,
});

const makeDeps = (document: CanvasDocumentContractV3) => {
  const stub = createTestStubRasterBackend();
  const snapshot: RasterCompositeExportSnapshot = {
    directPixelEpoch: 1,
    document,
    documentGeneration: 1,
    lifecycleGeneration: 1,
  };
  const encodeSurface = vi.fn((surface: RasterSurface, type?: string) => stub.encodeSurface(surface, type));
  const deps: ExportRasterCompositeDeps = {
    backend: {
      createSurface: (width, height) => stub.createSurface(width, height),
      encodeSurface,
    },
    captureSnapshot: vi.fn(() => snapshot),
    getLayerSurface: vi.fn(() =>
      Promise.resolve({
        rect: { height: 32, width: 64, x: 0, y: 0 },
        release: () => undefined,
        surface: stub.createSurface(64, 32),
      })
    ),
    isSnapshotCurrent: vi.fn(() => true),
  };
  return { deps, encodeSurface };
};

describe('exportRasterComposite', () => {
  it('exports tight content as a PNG without uploading it', async () => {
    const { deps, encodeSurface } = makeDeps(makeDoc([rasterLayer('raster')]));

    const result = await exportRasterComposite({ bounds: 'content' }, deps);

    expect(result).toMatchObject({ status: 'ok', rect: { x: -4, y: 8, width: 64, height: 32 } });
    expect(encodeSurface).toHaveBeenCalledOnce();
    expect(encodeSurface).toHaveBeenCalledWith(expect.anything(), 'image/png');
    expect(deps).not.toHaveProperty('uploadImage');
  });

  it('exports the exact requested bbox', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));
    const bbox: Rect = { height: 48, width: 96, x: 12, y: -6 };

    await expect(exportRasterComposite({ bounds: 'rect', rect: bbox }, deps)).resolves.toMatchObject({
      status: 'ok',
      rect: bbox,
    });
  });

  it('returns empty when the document has no enabled raster content', async () => {
    const { deps: noContentDeps } = makeDeps(makeDoc([controlLayer('control')]));

    await expect(exportRasterComposite({ bounds: 'content' }, noContentDeps)).resolves.toEqual({ status: 'empty' });
  });

  it('returns empty for a zero-area requested rect', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));

    await expect(
      exportRasterComposite({ bounds: 'rect', rect: { x: 0, y: 0, width: 0, height: 32 } }, deps)
    ).resolves.toEqual({ status: 'empty' });
  });

  it('returns over-budget before allocating a background composite', async () => {
    const { deps, encodeSurface } = makeDeps(makeDoc([rasterLayer('raster')]));
    deps.reserve = vi.fn(() => ({ availableBytes: 1_000, requestedBytes: 8_192, status: 'over-budget' as const }));

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toEqual({ status: 'over-budget' });
    expect(deps.getLayerSurface).not.toHaveBeenCalled();
    expect(encodeSurface).not.toHaveBeenCalled();
  });

  it('releases the reservation and cache pins after export', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));
    const releaseReservation = vi.fn();
    const releasePins = vi.fn();
    deps.reserve = vi.fn(() => ({ lease: { release: releaseReservation }, status: 'ok' as const }));
    deps.pin = vi.fn(() => ({ release: releasePins }));

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toMatchObject({ status: 'ok' });
    expect(deps.pin).toHaveBeenCalledWith(['raster']);
    expect(releasePins).toHaveBeenCalledOnce();
    expect(releaseReservation).toHaveBeenCalledOnce();
  });

  it('keeps its reservation and cache pins while rendering across generation release', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));
    const memory = new RasterMemoryBudgetController({ budgetBytes: 100_000 });
    const layerSurface = createDeferred<{ rect: Rect; release(): void; surface: RasterSurface }>();
    deps.getLayerSurface = vi.fn(() => layerSurface.promise);
    deps.reserve = (bytes) => memory.reserveOperation(bytes, { purpose: 'background-snapshot' });
    deps.pin = (layerIds) => {
      const leases = layerIds.map((layerId) => memory.pin(layerId));
      return { release: () => leases.forEach((lease) => lease.release()) };
    };

    const exported = exportRasterComposite({ bounds: 'content' }, deps);
    await vi.waitFor(() => expect(memory.snapshot().reservedBytes).toBeGreaterThan(0));
    memory.releaseGeneration(1);

    expect(memory.snapshot().reservedBytes).toBeGreaterThan(0);
    expect(memory.isPinned('raster')).toBe(true);

    const backend = createTestStubRasterBackend();
    const releaseLease = vi.fn();
    layerSurface.resolve({
      rect: { height: 32, width: 64, x: 0, y: 0 },
      release: releaseLease,
      surface: backend.createSurface(64, 32),
    });
    await expect(exported).resolves.toMatchObject({ status: 'ok' });
    expect(releaseLease).toHaveBeenCalledOnce();
    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(memory.isPinned('raster')).toBe(false);
  });

  it('reserves both the temporary surface and ImageData for an adjusted layer', async () => {
    const layer = rasterLayer('raster');
    if (layer.type !== 'raster') {
      throw new Error('Expected raster layer');
    }
    const { deps } = makeDeps(
      makeDoc([
        {
          ...layer,
          adjustments: [
            { brightness: 0.1, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
          ],
        },
      ])
    );
    const reserve = vi.fn(() => ({ lease: { release: vi.fn() }, status: 'ok' as const }));
    deps.reserve = reserve;

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toMatchObject({ status: 'ok' });

    expect(reserve).toHaveBeenCalledWith(24_576);
  });

  describe('a noisy text layer whose measured bounds exceed the estimate', () => {
    const { bbox, layer, measured, plannedScratchPixels } = MEASURED_NOISY_TEXT;
    /** Instruments allocation and reservation; `uncovered` lists allocations made beyond outstanding reservations. */
    const measuredDeps = () => {
      const stub = createTestStubRasterBackend();
      const layerSurface = stub.createSurface(measured.width, measured.height);
      const { deps } = makeDeps(makeDoc([layer]));
      let allocatedBytes = 0;
      let reservedBytes = 0;
      const uncovered: number[] = [];
      const allocate = (pixels: number) => {
        allocatedBytes += pixels * 4;
        if (allocatedBytes > reservedBytes) {
          uncovered.push(pixels);
        }
      };
      deps.backend.createSurface = (width, height) => {
        allocate(width * height);
        return stub.createSurface(width, height);
      };
      deps.readImageData = (surface, rect) => {
        allocate(rect.width * rect.height);
        return surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
      };
      deps.reserve = (bytes) => {
        reservedBytes += bytes;
        return { lease: { release: () => undefined }, status: 'ok' };
      };
      const releaseLayer = vi.fn();
      deps.getLayerSurface = vi.fn(() =>
        Promise.resolve({ rect: measured, release: releaseLayer, surface: layerSurface })
      );
      return { allocatedBytes: () => allocatedBytes, deps, releaseLayer, uncovered };
    };

    it('reserves the measured scratch before allocating it', async () => {
      const { allocatedBytes, deps, uncovered } = measuredDeps();

      await expect(exportRasterComposite({ bounds: 'rect', rect: bbox }, deps)).resolves.toMatchObject({
        status: 'ok',
      });

      // The measured 400×100 scratch and its ImageData were allocated, each within outstanding reservations.
      expect(allocatedBytes()).toBeGreaterThanOrEqual(
        (bbox.width * bbox.height + measured.width * measured.height * 2) * 4
      );
      expect(uncovered).toEqual([]);
    });

    it('refuses as over-budget instead of allocating past its reservation', async () => {
      const { allocatedBytes, deps, releaseLayer } = measuredDeps();
      // Output, plus the planned scratch surface and its ImageData.
      const upFrontBytes = (bbox.width * bbox.height + plannedScratchPixels * 2) * 4;
      const memory = new RasterMemoryBudgetController({ budgetBytes: upFrontBytes });
      deps.reserve = (bytes) => memory.reserveOperation(bytes, { purpose: 'raster-export' });

      await expect(exportRasterComposite({ bounds: 'rect', rect: bbox }, deps)).resolves.toEqual({
        status: 'over-budget',
      });

      // Only the output surface was allocated; the layer lease and every reservation are released.
      expect(allocatedBytes()).toBe(bbox.width * bbox.height * 4);
      expect(releaseLayer).toHaveBeenCalledOnce();
      expect(memory.snapshot().reservedBytes).toBe(0);
    });
  });

  describe('spatial adjustments', () => {
    const CLARITY = [{ amount: 50, id: 'clarity', isEnabled: true, size: 10, type: 'clarity' as const }];
    /** A 64×32 image layer at (-4, 8); a 4×4 bbox samples a sliver of it. */
    const spatialLayer = (adjustments = CLARITY): CanvasLayerContract => ({
      ...(rasterLayer('portrait') as CanvasRasterLayerContractV2),
      adjustments,
    });
    const SLIVER: Rect = { height: 4, width: 4, x: 0, y: 10 };
    const withReserve = (deps: ExportRasterCompositeDeps) => {
      const reserve = vi.fn((_bytes: number) => ({ lease: { release: vi.fn() }, status: 'ok' as const }));
      deps.reserve = reserve;
      return reserve;
    };

    it("reserves a spatial layer's whole surface and its ImageData, whatever the bbox samples", async () => {
      const { deps } = makeDeps(makeDoc([spatialLayer()]));
      const reserve = withReserve(deps);
      const runSpatial = vi.fn((request: { pixels: Uint8ClampedArray }) =>
        Promise.resolve({ pixels: request.pixels, status: 'ok' as const })
      );
      deps.runSpatial = runSpatial;

      await expect(exportRasterComposite({ bounds: 'rect', rect: SLIVER }, deps)).resolves.toMatchObject({
        status: 'ok',
      });

      expect(reserve).toHaveBeenCalledExactlyOnceWith((4 * 4 + 64 * 32 * 2) * 4);
      // The job reads all of the layer, not the sampled region.
      expect(runSpatial).toHaveBeenCalledOnce();
      expect(runSpatial.mock.calls[0]![0]).toMatchObject({ height: 32, priority: 'now', width: 64 });
    });

    it("reserves a spatial group's buffer over its members' bounds, not the bbox", async () => {
      const layer = rasterLayer('member');
      const { deps } = makeDeps(makeDoc([groupContract('group', [layer], { adjustments: CLARITY })]));
      const reserve = withReserve(deps);
      const runSpatial = vi.fn((request: { pixels: Uint8ClampedArray }) =>
        Promise.resolve({ pixels: request.pixels, status: 'ok' as const })
      );
      deps.runSpatial = runSpatial;

      await expect(exportRasterComposite({ bounds: 'rect', rect: SLIVER }, deps)).resolves.toMatchObject({
        status: 'ok',
      });

      expect(reserve).toHaveBeenCalledExactlyOnceWith((4 * 4 + 64 * 32 * 2) * 4);
      expect(runSpatial.mock.calls[0]![0]).toMatchObject({ height: 32, width: 64 });
    });

    it.each(['too-large', 'over-budget', 'error'] as const)(
      'refuses %s with the name of the layer whose stack could not run',
      async (reason) => {
        const { deps, encodeSurface } = makeDeps(makeDoc([spatialLayer()]));
        deps.runSpatial = () => Promise.resolve({ status: reason });

        await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toEqual({
          ownerId: 'portrait',
          ownerName: 'portrait',
          reason,
          status: 'adjustment-refused',
        });
        expect(encodeSurface).not.toHaveBeenCalled();
      }
    );

    it('names the group whose stack could not run', async () => {
      const { deps } = makeDeps(
        makeDoc([groupContract('group', [rasterLayer('member')], { adjustments: CLARITY, name: 'Sky' })])
      );
      deps.runSpatial = () => Promise.resolve({ status: 'too-large' });

      await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toMatchObject({
        ownerId: 'group',
        ownerName: 'Sky',
        status: 'adjustment-refused',
      });
    });

    it('never exports a spatial stack without its spatial entries', async () => {
      const { deps, encodeSurface } = makeDeps(makeDoc([spatialLayer()]));

      await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toMatchObject({
        reason: 'error',
        status: 'adjustment-refused',
      });
      expect(encodeSurface).not.toHaveBeenCalled();
    });

    it('reports a job cancelled by cooldown or cache release as stale', async () => {
      const { deps } = makeDeps(makeDoc([spatialLayer()]));
      deps.runSpatial = () => Promise.resolve({ status: 'aborted' });

      await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toEqual({ status: 'stale' });
    });
  });

  it('discards an encoded blob if the document changes during export', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));
    vi.mocked(deps.isSnapshotCurrent).mockReturnValueOnce(true).mockReturnValue(false);

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toEqual({ status: 'stale' });
  });
});
