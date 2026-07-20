import type { CanvasDocumentContractV2, CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
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

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 128, width: 128, x: 0, y: 0 },
  height: 128,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 128,
});

const makeDeps = (document: CanvasDocumentContractV2) => {
  const stub = createTestStubRasterBackend();
  const snapshot: RasterCompositeExportSnapshot = {
    contentEpoch: 1,
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
    const layerSurface = createDeferred<{ rect: Rect; surface: RasterSurface }>();
    deps.getLayerSurface = vi.fn(() => layerSurface.promise);
    deps.reserve = (bytes) => memory.reserveOperation(bytes, { purpose: 'background-snapshot' });
    deps.pin = (layerIds) => {
      const leases = layerIds.map((layerId) => memory.pinOperation(layerId));
      return { release: () => leases.forEach((lease) => lease.release()) };
    };

    const exported = exportRasterComposite({ bounds: 'content' }, deps);
    await vi.waitFor(() => expect(memory.snapshot().reservedBytes).toBeGreaterThan(0));
    memory.releaseGeneration(1);

    expect(memory.snapshot().reservedBytes).toBeGreaterThan(0);
    expect(memory.isPinned('raster')).toBe(true);

    const backend = createTestStubRasterBackend();
    layerSurface.resolve({
      rect: { height: 32, width: 64, x: 0, y: 0 },
      surface: backend.createSurface(64, 32),
    });
    await expect(exported).resolves.toMatchObject({ status: 'ok' });
    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(memory.isPinned('raster')).toBe(false);
  });

  it('reserves both the temporary surface and ImageData for an adjusted layer', async () => {
    const layer = rasterLayer('raster');
    if (layer.type !== 'raster') {
      throw new Error('Expected raster layer');
    }
    const { deps } = makeDeps(makeDoc([{ ...layer, adjustments: { brightness: 0.1, contrast: 0, saturation: 0 } }]));
    const reserve = vi.fn(() => ({ lease: { release: vi.fn() }, status: 'ok' as const }));
    deps.reserve = reserve;

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toMatchObject({ status: 'ok' });

    expect(reserve).toHaveBeenCalledWith(24_576);
  });

  it('discards an encoded blob if the document changes during export', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));
    vi.mocked(deps.isSnapshotCurrent).mockReturnValueOnce(true).mockReturnValue(false);

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toEqual({ status: 'stale' });
  });
});
