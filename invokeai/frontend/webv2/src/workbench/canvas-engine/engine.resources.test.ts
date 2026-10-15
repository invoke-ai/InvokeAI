import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { GroupSurfaceContent } from '@workbench/canvas-engine/render/groupCompositeScopes';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';

import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import type * as RasterControllerModule from './controllers/rasterController';
import type { RasterController } from './controllers/rasterController';

import { createCanvasEngine } from './engine';

const NO_GROUP_CONTENT: GroupSurfaceContent = { excludeIds: new Set(), float: null, previews: null };

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
// Stub surfaces record calls without allocating, so a cache this size only exceeds the 512 MiB budget on paper.
const OVER_BUDGET_RECT = { height: 12_000, width: 12_000, x: 0, y: 0 };

const fakeCanvas = (): HTMLCanvasElement => {
  const surface = createTestStubRasterBackend().createSurface(100, 100);
  return {
    addEventListener: () => undefined,
    getBoundingClientRect: () => ({ height: 100, left: 0, top: 0, width: 100 }),
    getContext: () => surface.ctx,
    height: 100,
    removeEventListener: () => undefined,
    width: 100,
  } as unknown as HTMLCanvasElement;
};

const createEngine = (bitmapStore?: BitmapStore) => {
  const document = documentFrom([
    groupContract('group', [layerContract('layer')], { opacity: 0.5 }),
    layerContract('other'),
  ]);
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document,
    type: 'replaceCanvasDocument',
  });
  const listeners = new Set<() => void>();
  const mutationPort = {
    commitEdit: () => undefined,
    dispatch: () => false,
    getCanvasState: () => project.canvas,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  } as unknown as CanvasProjectMutationPort;
  const apply = (mutation: CanvasProjectMutation): void => {
    project = applyCanvasProjectMutation(project, mutation);
    for (const listener of listeners) {
      listener();
    }
  };
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
  const start = leaves.findIndex((leaf) => leaf.id === 'layer');
  raster.groups.get(
    { adjustments: [], blendMode: 'normal', children: [], end: start + 1, id: 'group', opacity: 0.5, start },
    leaves.slice(start, start + 1),
    [leaves[start]!.worldTransform],
    NO_GROUP_CONTENT
  );
  return { apply, engine, raster };
};

const offscreenDocument = () =>
  documentFrom(
    ['visible', 'dirty', 'held', 'hidden'].map((id, index) =>
      layerContract(id, 'raster', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: index * 100_000, y: 0 } })
    )
  );

const createFramedEngine = (pendingLayerIds: ReadonlySet<string>) => {
  const document = offscreenDocument();
  const canvas = { document, documentRevision: 0, snapshots: [], version: 3 };
  const store = {
    discardLayer: vi.fn(),
    dispose: vi.fn(),
    flushPendingUploads: () => Promise.resolve(),
    hasPendingClear: () => false,
    hasPendingWork: (layerId: string) => pendingLayerIds.has(layerId),
    isSelfEcho: () => false,
    markLayerDirty: vi.fn(),
    reset: vi.fn(),
    suspendLayer: () => () => undefined,
  } satisfies BitmapStore;
  const { engine } = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    bitmapStore: store,
    fonts: null,
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: {
      commitEdit: () => undefined,
      dispatch: () => false,
      getCanvasState: () => canvas,
      subscribe: () => () => undefined,
    } as unknown as CanvasProjectMutationPort,
    projectId: 'project',
    reportError: () => undefined,
    uploadImage: () => Promise.resolve({ height: 10, imageName: 'unused', width: 10 }),
    uploadIntermediateImage: () => Promise.resolve({ height: 10, imageName: 'unused', width: 10 }),
  });
  const raster = owners.raster!;
  for (const layerId of ['dirty', 'held', 'hidden']) {
    raster.layers.getOrCreateRect(layerId, OVER_BUDGET_RECT);
    raster.layers.publishPixels(layerId);
  }
  return { engine, raster };
};

describe('engine working-set protection', () => {
  it('keeps unsaved and session-held offscreen pixels when a frame enforces the budget', () => {
    const { engine, raster } = createFramedEngine(new Set(['dirty']));
    engine.stores.transformSession.set({
      layerId: 'held',
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100_000 * 2, y: 0 },
    } as Parameters<typeof engine.stores.transformSession.set>[0]);

    engine.surface.attach(fakeCanvas(), fakeCanvas());
    engine.surface.resize(100, 100, 1);

    expect(raster.layers.peek('hidden')).toBeUndefined();
    expect(raster.layers.peek('dirty')).toBeDefined();
    expect(raster.layers.peek('held')).toBeDefined();
    expect(raster.memory.snapshot().overageBytes).toBeGreaterThan(0);
    engine.stores.transformSession.set(null);
    engine.lifecycle.dispose();
  });
});

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

  it('releases every resource category deterministically on disposal', async () => {
    const { engine, raster } = createEngine();
    const bitmap = { close: vi.fn(), height: 10, width: 10 } as unknown as ImageBitmap;
    await raster.bitmaps.acquire('decoded', () => Promise.resolve(bitmap));
    expect(raster.memory.snapshot().decodedBytes).toBe(400);
    raster.memory.reserve(100, { generation: 0, purpose: 'thumbnail' });
    raster.memory.reserveOperation(100, { purpose: 'layer-operation' });
    raster.memory.trackDetached(100);
    raster.memory.pin('layer');

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
    expect(bitmap.close).toHaveBeenCalledOnce();
  });
});

describe('engine group surfaces', () => {
  it('releases a deleted group surface', () => {
    const { apply, engine, raster } = createEngine();
    expect(raster.memory.snapshot().groupBytes).toBeGreaterThan(0);

    apply({ ids: ['group'], type: 'removeCanvasLayers' });
    expect(raster.memory.snapshot().groupBytes).toBe(0);
    engine.lifecycle.dispose();
  });
});
