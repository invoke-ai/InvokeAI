import type { CanvasDocumentSnapshot } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasRasterLayerContractV2,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type { ExecutePsdExportDeps, PsdExportPlan } from '@workbench/canvas-engine/export/psdExport';
import type { CanvasRasterSnapshot } from '@workbench/canvas-engine/rasterTransactions';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { derivePsdPixelAreaLimit, PSD_ALLOCATION_BYTES_PER_PIXEL, PsdExportController } from './psdExportController';

const document: CanvasDocumentContractV3 = {
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([
    {
      blendMode: 'normal',
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 100, imageName: 'layer.png', width: 100 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    },
  ]),
  selectedLayerId: null,
  version: 3,
  width: 100,
};

const canvas: CanvasStateContractV3 = {
  document,
  documentRevision: 0,
  snapshots: [],
  stagingArea: {
    areThumbnailsVisible: false,
    autoSwitchMode: 'off',
    isVisible: false,
    pendingImageIds: [],
    pendingImages: [],
    selectedImageIndex: 0,
  },
  version: 3,
};

const canvasWithDocument = (nextDocument: CanvasDocumentContractV3): CanvasStateContractV3 => ({
  ...canvas,
  document: nextDocument,
});

describe('PsdExportController', () => {
  it('derives the PSD allocation pixel-area limit from currently available reserved bytes', () => {
    expect(derivePsdPixelAreaLimit(159_999)).toBe(19_999);
    expect(derivePsdPixelAreaLimit(160_000)).toBe(20_000);
  });

  it('executes from one immutable raster snapshot instead of live layer pixels', async () => {
    const backend = createTestStubRasterBackend();
    const detached = backend.createSurface(100, 100);
    const documentSnapshot: CanvasDocumentSnapshot = {
      canvas,
      documentGeneration: 4,
    };
    const release = vi.fn();
    const rasterSnapshot: CanvasRasterSnapshot = {
      ...documentSnapshot,
      emptyLayerIds: new Set<string>(),
      layerSurfaces: new Map([['layer', { rect: { height: 100, width: 100, x: 0, y: 0 }, surface: detached }]]),
      release,
    };
    const captureRasterSnapshot = vi.fn(() => Promise.resolve({ snapshot: rasterSnapshot, status: 'ok' as const }));
    const execute = vi.fn(async (_plan, _fileName, deps: ExecutePsdExportDeps) => {
      await expect(deps.getLayerSurface('layer')).resolves.toEqual({
        rect: { height: 100, width: 100, x: 0, y: 0 },
        surface: detached,
      });
    });
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot,
      execute,
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: () => ({ lease: { release: vi.fn() }, status: 'ok' }),
    });

    await expect(controller.export('layers.psd')).resolves.toBe('exported');
    expect(captureRasterSnapshot).toHaveBeenCalledWith(documentSnapshot, ['layer'], expect.any(Object));
    expect(execute).toHaveBeenCalledOnce();
    expect(release).toHaveBeenCalledOnce();
  });

  it('plans a new bitmap-less paint layer from its captured unflushed live pixels', async () => {
    const backend = createTestStubRasterBackend();
    const paintDocument: CanvasDocumentContractV3 = {
      ...document,
      stacks: stacksFrom([
        {
          blendMode: 'normal',
          id: 'paint',
          isEnabled: true,
          isLocked: false,
          name: 'Paint',
          opacity: 1,
          source: { bitmap: null, type: 'paint' },
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
          type: 'raster',
        },
      ]),
    };
    const paintCanvas = canvasWithDocument(paintDocument);
    const documentSnapshot: CanvasDocumentSnapshot = { canvas: paintCanvas, documentGeneration: 6 };
    const rect = { height: 30, width: 20, x: -5, y: -6 };
    const execute = vi.fn((plan) => {
      expect(plan).toMatchObject({ height: 30, status: 'ok', width: 20 });
      if (plan.status !== 'ok') {
        throw new Error('Expected an executable PSD plan');
      }
      expect(plan.layers[0]?.contentRect).toEqual(rect);
      return Promise.resolve();
    });
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set<string>(),
            layerSurfaces: new Map([['paint', { rect, surface: backend.createSurface(20, 30) }]]),
            release: vi.fn(),
          },
          status: 'ok',
        }),
      execute,
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: () => ({ lease: { release: vi.fn() }, status: 'ok' }),
    });

    await expect(controller.export('paint.psd')).resolves.toBe('exported');
    expect(execute).toHaveBeenCalledOnce();
  });

  it('plans and reserves from a grown live-cache rect instead of smaller persisted source bounds', async () => {
    const backend = createTestStubRasterBackend();
    const smallLayer: CanvasRasterLayerContractV2 = {
      ...(getDocumentLeaves(document)[0] as CanvasRasterLayerContractV2),
      source: { image: { height: 10, imageName: 'small.png', width: 10 }, type: 'image' },
    };
    const smallDocument: CanvasDocumentContractV3 = {
      ...document,
      stacks: stacksFrom([smallLayer]),
    };
    const grownCanvas = canvasWithDocument(smallDocument);
    const documentSnapshot: CanvasDocumentSnapshot = { canvas: grownCanvas, documentGeneration: 7 };
    const grownRect = { height: 40, width: 30, x: -12, y: -8 };
    const reserve = vi.fn(() => ({ lease: { release: vi.fn() }, status: 'ok' as const }));
    const execute = vi.fn((plan) => {
      expect(plan).toMatchObject({ height: 40, status: 'ok', width: 30 });
      if (plan.status !== 'ok') {
        throw new Error('Expected an executable PSD plan');
      }
      expect(plan.layers[0]?.contentRect).toEqual(grownRect);
      return Promise.resolve();
    });
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set<string>(),
            layerSurfaces: new Map([['layer', { rect: grownRect, surface: backend.createSurface(30, 40) }]]),
            release: vi.fn(),
          },
          status: 'ok',
        }),
      execute,
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve,
    });

    await expect(controller.export('grown.psd')).resolves.toBe('exported');
    expect(reserve).toHaveBeenCalledWith(30 * 40 * 2 * PSD_ALLOCATION_BYTES_PER_PIXEL);
  });

  it('exports valid captured pixels while skipping a confirmed-empty paint layer', async () => {
    const backend = createTestStubRasterBackend();
    const validLayer = getDocumentLeaves(document)[0] as CanvasRasterLayerContractV2;
    const blankLayer: CanvasRasterLayerContractV2 = {
      ...validLayer,
      id: 'blank',
      name: 'Blank',
      source: { bitmap: null, type: 'paint' },
    };
    const mixedCanvas = canvasWithDocument({ ...document, stacks: stacksFrom([validLayer, blankLayer]) });
    const documentSnapshot: CanvasDocumentSnapshot = { canvas: mixedCanvas, documentGeneration: 8 };
    const execute = vi.fn((plan: PsdExportPlan) => {
      if (plan.status !== 'ok') {
        throw new Error('Expected an executable PSD plan');
      }
      expect(plan.layers.map((layer) => layer.id)).toEqual(['layer']);
      return Promise.resolve();
    });
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set(['blank']),
            layerSurfaces: new Map([
              [
                'layer',
                {
                  rect: { height: 100, width: 100, x: 0, y: 0 },
                  surface: backend.createSurface(100, 100),
                },
              ],
            ]),
            release: vi.fn(),
          },
          status: 'ok',
        }),
      execute,
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: () => ({ lease: { release: vi.fn() }, status: 'ok' }),
    });

    await expect(controller.export('mixed.psd')).resolves.toBe('exported');
    expect(execute).toHaveBeenCalledOnce();
  });

  it('returns nothing when every requested raster layer is confirmed empty', async () => {
    const backend = createTestStubRasterBackend();
    const blankLayer: CanvasRasterLayerContractV2 = {
      ...(getDocumentLeaves(document)[0] as CanvasRasterLayerContractV2),
      id: 'blank',
      name: 'Blank',
      source: { bitmap: null, type: 'paint' },
    };
    const blankCanvas = canvasWithDocument({ ...document, stacks: stacksFrom([blankLayer]) });
    const documentSnapshot: CanvasDocumentSnapshot = { canvas: blankCanvas, documentGeneration: 9 };
    const release = vi.fn();
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set(['blank']),
            layerSurfaces: new Map(),
            release,
          },
          status: 'ok',
        }),
      execute: vi.fn(),
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: vi.fn(),
    });

    await expect(controller.export('blank.psd')).resolves.toBe('nothing');
    expect(release).toHaveBeenCalledOnce();
  });

  it('derives the execution limit after detached capture and releases the snapshot when over budget', async () => {
    const backend = createTestStubRasterBackend();
    const snapshotRelease = vi.fn();
    const reserve = vi.fn();
    const captureRasterSnapshot = vi.fn(() =>
      Promise.resolve({
        snapshot: {
          canvas,
          documentGeneration: 1,
          emptyLayerIds: new Set<string>(),
          layerSurfaces: new Map([
            ['layer', { rect: { height: 100, width: 100, x: 0, y: 0 }, surface: backend.createSurface(100, 100) }],
          ]),
          release: snapshotRelease,
        },
        status: 'ok' as const,
      })
    );
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => ({
        canvas,
        documentGeneration: 1,
      }),
      captureRasterSnapshot,
      getAvailableBytes: () => 0,
      isDocumentSnapshotCurrent: () => true,
      reserve,
    });

    await expect(controller.export('layers.psd')).resolves.toBe('over-budget');
    expect(captureRasterSnapshot).toHaveBeenCalledOnce();
    expect(snapshotRelease).toHaveBeenCalledOnce();
    expect(reserve).not.toHaveBeenCalled();
  });

  it('propagates over-budget when immutable raster snapshot capture is refused', async () => {
    const controller = new PsdExportController({
      backend: createTestStubRasterBackend(),
      captureDocumentSnapshot: () => ({
        canvas,
        documentGeneration: 1,
      }),
      captureRasterSnapshot: () => Promise.resolve({ status: 'over-budget' }),
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: () => ({ lease: { release: vi.fn() }, status: 'ok' }),
    });

    await expect(controller.export('layers.psd')).resolves.toBe('over-budget');
  });

  it('rejects a stale document generation after capture and releases the raster snapshot', async () => {
    const backend = createTestStubRasterBackend();
    const documentSnapshot: CanvasDocumentSnapshot = {
      canvas,
      documentGeneration: 3,
    };
    const release = vi.fn();
    const isDocumentSnapshotCurrent = vi.fn().mockReturnValueOnce(true).mockReturnValueOnce(false);
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set<string>(),
            layerSurfaces: new Map(),
            release,
          },
          status: 'ok',
        }),
      execute: vi.fn(),
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent,
      reserve: vi.fn(),
    });

    await expect(controller.export('layers.psd')).resolves.toBe('stale');
    expect(release).toHaveBeenCalledOnce();
  });

  it('aborts active execution on disposal and releases snapshot and reservation resources', async () => {
    const backend = createTestStubRasterBackend();
    const documentSnapshot: CanvasDocumentSnapshot = { canvas, documentGeneration: 5 };
    const snapshotRelease = vi.fn();
    const reservationRelease = vi.fn();
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () =>
        Promise.resolve({
          snapshot: {
            ...documentSnapshot,
            emptyLayerIds: new Set<string>(),
            layerSurfaces: new Map([
              [
                'layer',
                {
                  rect: { height: 100, width: 100, x: 0, y: 0 },
                  surface: backend.createSurface(100, 100),
                },
              ],
            ]),
            release: snapshotRelease,
          },
          status: 'ok',
        }),
      execute: (_plan, _fileName, deps) =>
        new Promise((_resolve, reject) => {
          deps.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {
            once: true,
          });
        }),
      getAvailableBytes: () => 1_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: () => ({ lease: { release: reservationRelease }, status: 'ok' }),
    });

    const exportPromise = controller.export('layers.psd');
    await vi.waitFor(() => expect(snapshotRelease).not.toHaveBeenCalled());
    controller.dispose();

    await expect(exportPromise).resolves.toBe('aborted');
    expect(snapshotRelease).toHaveBeenCalledOnce();
    expect(reservationRelease).toHaveBeenCalledOnce();
  });
});

describe('PsdExportController — raster tree', () => {
  it('plans the raster tree as folders with own flags and skips leaves with nothing captured', async () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(100, 100);
    const rect = { height: 100, width: 100, x: 0, y: 0 };
    const leaf = (id: string, isEnabled = true): CanvasRasterLayerContractV2 => ({
      ...(getDocumentLeaves(document)[0] as CanvasRasterLayerContractV2),
      id,
      isEnabled,
      name: id,
    });
    const treeDocument: CanvasDocumentContractV3 = {
      ...document,
      stacks: stacksFrom([
        groupContract('G', [leaf('inner', false), groupContract('empty', [leaf('blank')])], { isEnabled: false }),
        leaf('root'),
      ]),
    };
    const documentSnapshot: CanvasDocumentSnapshot = {
      canvas: canvasWithDocument(treeDocument),
      documentGeneration: 1,
    };
    const rasterSnapshot: CanvasRasterSnapshot = {
      ...documentSnapshot,
      emptyLayerIds: new Set(['blank']),
      layerSurfaces: new Map([
        ['inner', { rect, surface }],
        ['root', { rect, surface }],
      ]),
      release: vi.fn(),
    };
    const execute = vi.fn((_plan: PsdExportPlan) => Promise.resolve());
    const controller = new PsdExportController({
      backend,
      captureDocumentSnapshot: () => documentSnapshot,
      captureRasterSnapshot: () => Promise.resolve({ snapshot: rasterSnapshot, status: 'ok' as const }),
      execute,
      getAvailableBytes: () => 10_000_000,
      isDocumentSnapshotCurrent: () => true,
      reserve: () => ({ lease: { release: vi.fn() }, status: 'ok' }),
    });

    await expect(controller.export('tree')).resolves.toBe('exported');
    const plan = execute.mock.calls[0]![0];
    if (plan.status !== 'ok') {
      throw new Error(plan.status);
    }
    expect(plan.tree.map((node) => (node.kind === 'folder' ? `${node.name}/` : node.id))).toEqual(['root', 'G/']);
    const folder = plan.tree[1];
    expect(folder.kind === 'folder' && folder.hidden).toBe(true);
    expect(folder.kind === 'folder' && folder.children.map((node) => node.id)).toEqual(['inner']);
    expect(plan.layers.map((layer) => [layer.id, layer.hidden, layer.contributes])).toEqual([
      ['root', false, true],
      ['inner', true, false],
    ]);
  });
});
