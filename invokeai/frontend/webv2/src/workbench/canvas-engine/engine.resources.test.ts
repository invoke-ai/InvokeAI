import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';

import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type * as RasterControllerModule from './controllers/rasterController';
import type { RasterController } from './controllers/rasterController';

import { createCanvasEngine } from './engine';

const owners = vi.hoisted(() => ({ raster: null as RasterController | null }));

// Captures the engine's real raster owner; its behavior is unchanged.
vi.mock('./controllers/rasterController', async (importOriginal) => {
  const actual = await importOriginal<typeof RasterControllerModule>();
  class CapturedRasterController extends actual.RasterController {
    constructor(options: ConstructorParameters<typeof actual.RasterController>[0]) {
      super(options);
      owners.raster = this;
    }
  }
  return { ...actual, RasterController: CapturedRasterController };
});

const RECT = { height: 10, width: 10, x: 0, y: 0 };

const createEngine = (bitmapStore?: BitmapStore) => {
  const document = documentFrom([groupContract('group', [layerContract('layer')], { opacity: 0.5 })]);
  const canvas = { document, documentRevision: 0, snapshots: [], version: 3 };
  const mutationPort = {
    commitEdit: () => undefined,
    dispatch: () => false,
    getCanvasState: () => canvas,
    subscribe: () => () => undefined,
  } as unknown as CanvasProjectMutationPort;
  const { engine } = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    bitmapStore,
    fonts: null,
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort,
    projectId: 'project',
    reportError: () => undefined,
    uploadImage: () => Promise.resolve({ height: 10, imageName: 'unused', width: 10 }),
    uploadIntermediateImage: () => Promise.resolve({ height: 10, imageName: 'unused', width: 10 }),
  });
  const raster = owners.raster!;
  raster.layers.getOrCreateRect('layer', RECT);
  raster.layers.publishPixels('layer');
  const leaves = compileDocumentLeaves(document);
  raster.groups.get(
    { adjustments: [], blendMode: 'normal', children: [], end: 1, id: 'group', opacity: 0.5, start: 0 },
    leaves,
    leaves.map((leaf) => leaf.worldTransform),
    new Set()
  );
  return { engine, raster };
};

describe('engine raster resource lifecycle', () => {
  it('releases base, derived and group surfaces after a successful cooldown', async () => {
    const { engine, raster } = createEngine();
    expect(raster.memory.snapshot()).toMatchObject({ baseBytes: 400, groupBytes: 400 });

    await expect(engine.lifecycle.beginCooldown()).resolves.toBe('cooled');

    expect(raster.memory.snapshot()).toMatchObject({ baseBytes: 0, derivedBytes: 0, groupBytes: 0 });
    engine.lifecycle.dispose();
  });

  it('retains unsaved pixels when the cooldown persistence barrier fails', async () => {
    const store = {
      discardLayer: vi.fn(),
      dispose: vi.fn(),
      flushPendingUploads: () => Promise.reject(new Error('offline')),
      hasPendingClear: () => false,
      hasPendingWork: () => true,
      isSelfEcho: () => false,
      markLayerDirty: vi.fn(),
      reset: vi.fn(),
      suspendLayer: () => () => undefined,
    } satisfies BitmapStore;
    const { engine, raster } = createEngine(store);

    await expect(engine.lifecycle.beginCooldown()).resolves.toBe('dirty');

    expect(raster.layers.peek('layer')).toBeDefined();
    expect(raster.memory.snapshot().baseBytes).toBe(400);
    engine.lifecycle.dispose();
  });

  it('releases every resource category deterministically on disposal', () => {
    const { engine, raster } = createEngine();
    raster.memory.reserve(100, { generation: 0, purpose: 'thumbnail' });
    raster.memory.reserveOperation(100, { purpose: 'layer-operation' });
    raster.memory.trackDetached(100);
    raster.pin('layer');

    engine.lifecycle.dispose();

    expect(raster.memory.snapshot()).toEqual({
      baseBytes: 0,
      decodedBytes: 0,
      derivedBytes: 0,
      detachedBytes: 0,
      groupBytes: 0,
      overageBytes: 0,
      reservedBytes: 0,
      totalBytes: 0,
    });
    expect(raster.isProtected('layer')).toBe(false);
  });
});
