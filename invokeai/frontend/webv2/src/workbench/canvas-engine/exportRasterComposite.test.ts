import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasImageRef, CanvasLayerContract } from '@workbench/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type { ExportRasterCompositeDeps, RasterCompositeExportSnapshot } from './exportRasterComposite';

import { exportRasterComposite } from './exportRasterComposite';

const imageRef = (imageName: string, width = 64, height = 32): CanvasImageRef => ({ height, imageName, width });

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
  const snapshot: RasterCompositeExportSnapshot = { document, documentGeneration: 1 };
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

  it('discards an encoded blob if the document changes during export', async () => {
    const { deps } = makeDeps(makeDoc([rasterLayer('raster')]));
    vi.mocked(deps.isSnapshotCurrent).mockReturnValueOnce(true).mockReturnValue(false);

    await expect(exportRasterComposite({ bounds: 'content' }, deps)).resolves.toEqual({ status: 'stale' });
  });
});
