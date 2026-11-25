import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasAdjustmentsContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type * as ImagePatchModule from '@workbench/canvas-engine/history/imagePatch';
import type * as LayerSnapshotModule from '@workbench/canvas-engine/history/layerSnapshot';
import type * as AdjustedSurfaceCacheModule from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type {
  RasterCallLogEntry,
  StubRasterBackend,
  StubRasterSurface,
} from '@workbench/canvas-engine/render/raster.testStub';
import type { ToolId } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { WorkbenchAction } from '@workbench/workbenchState.testing';

import { resetMaskFillColors } from '@workbench/canvas-engine/document/layerFactories';
import {
  applyCanvasProjectMutation,
  isCanvasProjectMutation,
  type CanvasProjectMutation,
} from '@workbench/canvasProjectMutations';

type EngineTestAction = WorkbenchAction | CanvasProjectMutation;

import type { SpatialAdjustmentExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { createVectorLayer } from '@workbench/canvas-engine/document/vectorLayerFactories';
import * as compositor from '@workbench/canvas-engine/render/compositor';
import { DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/render/compositor';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createInlineSpatialExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService.testing';
import { canvasApplicationPort } from '@workbench/canvas-operations/applicationPort';
import {
  createCanvasEngine as createApplicationCanvasEngine,
  type CanvasEngine,
  type CanvasEngineOptions,
} from '@workbench/canvas-operations/createCanvasEngine';
import { getCanvasOperationImplementation as getCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { afterEach, beforeEach, describe, expect, it, type Mock, vi } from 'vitest';

import type { BitmapStore } from './document/bitmapStore';
import type { StrokeCommittedEvent } from './tools/tool';

import { createBitmapStore } from './document/bitmapStore';
import { mergeDownMatrix } from './document/mergeDown';
import { createCanvasEngine as createCoreCanvasEngine } from './engine';
import { RASTERIZATION_RETRY_BASE_MS } from './layerRasterizer';
import { CanvasImageMissingError } from './render/rasterizers/imageRasterizer';

interface EngineStore {
  dispatch(action: EngineTestAction): void;
  getState(): WorkbenchState;
  reducesCanvasMutations?: true;
  subscribe(listener: () => void): () => void;
}

const createTestMutationPort = (store: EngineStore, projectId: string): CanvasProjectMutationPort => {
  const readStoreCanvas = () => store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null;
  let canvas = readStoreCanvas();
  let observedStoreCanvas = canvas;
  const listeners = new Set<() => void>();
  const unsubscribeStore = store.subscribe(() => {
    const next = readStoreCanvas();
    if (next !== observedStoreCanvas) {
      observedStoreCanvas = next;
      canvas = next;
    }
    for (const listener of listeners) {
      listener();
    }
  });
  return {
    commitEdit: () => undefined,
    dispatch: (mutation) => {
      const before = canvas;
      if (!before) {
        return false;
      }
      store.dispatch(mutation);
      if (store.reducesCanvasMutations) {
        const reduced = readStoreCanvas();
        observedStoreCanvas = reduced;
        canvas = reduced;
      } else {
        const project = store.getState().projects.find((candidate) => candidate.id === projectId);
        canvas = applyCanvasProjectMutation({ ...(project as Project), canvas: before }, mutation).canvas;
        for (const listener of listeners) {
          listener();
        }
      }
      return canvas !== before;
    },
    getCanvasState: () => {
      const current = readStoreCanvas();
      if (current !== observedStoreCanvas) {
        observedStoreCanvas = current;
        canvas = current;
      }
      return canvas;
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) {
          unsubscribeStore();
        }
      };
    },
  };
};

type TestCanvasEngineOptions = Omit<
  CanvasEngineOptions,
  'ensureProjectOnServer' | 'getMainModelBase' | 'mutationPort' | 'reportError'
> & {
  ensureProjectOnServer?: () => Promise<void>;
  getMainModelBase?: () => string | null;
  mutationPort?: CanvasProjectMutationPort;
  reportError?: CanvasEngineOptions['reportError'];
  store: EngineStore;
};

const createCanvasEngine = ({
  getMainModelBase,
  mutationPort,
  projectId,
  reportError,
  store,
  ...options
}: TestCanvasEngineOptions): CanvasEngine =>
  createApplicationCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    ...options,
    getMainModelBase:
      getMainModelBase ?? (() => canvasApplicationPort.getSelectedModelBase(store.getState(), projectId)),
    mutationPort: mutationPort ?? createTestMutationPort(store, projectId),
    projectId,
    reportError: reportError ?? ((report) => reportedErrors.push(report)),
  });

const reportedErrors: Parameters<CanvasEngineOptions['reportError']>[0][] = [];

const expectReplayFailure = async (replay: Promise<string>, failure: string | RegExp): Promise<void> => {
  expect(await replay).toBe('failed');
  const report = reportedErrors.at(-1);
  expect(report?.message).toBe('History replay failed');
  expect((report?.context.error as Error | undefined)?.message).toMatch(failure);
};

const adjustedSurfaceCacheDeletes = vi.hoisted(() => [] as string[]);
const adjustedSurfaceCacheGets = vi.hoisted(() => [] as string[]);
const adjustedSurfaceCacheDeleteFaults = vi.hoisted(() => new Set<string>());
const historyPreparationFaults = vi.hoisted(() => ({
  imagePatch: false,
  imagePatchBefore: null as ImageData | null,
  layerSnapshot: false,
}));

vi.mock('@workbench/canvas-engine/render/adjustedSurfaceCache', async (importOriginal) => {
  const actual = await importOriginal<typeof AdjustedSurfaceCacheModule>();
  return {
    ...actual,
    createAdjustedSurfaceCache: (...args: Parameters<typeof actual.createAdjustedSurfaceCache>) => {
      const cache = actual.createAdjustedSurfaceCache(...args);
      return {
        ...cache,
        delete: (layerId: string) => {
          adjustedSurfaceCacheDeletes.push(layerId);
          if (adjustedSurfaceCacheDeleteFaults.has(layerId)) {
            adjustedSurfaceCacheDeleteFaults.delete(layerId);
            throw new Error('adjusted surface cache delete failed');
          }
          cache.delete(layerId);
        },
        get: (...args: Parameters<typeof cache.get>) => {
          adjustedSurfaceCacheGets.push(args[0]);
          return cache.get(...args);
        },
      };
    },
  };
});

vi.mock('./history/imagePatch', async (importOriginal) => {
  const actual = await importOriginal<typeof ImagePatchModule>();
  return {
    ...actual,
    createImagePatchEntry: (...args: Parameters<typeof actual.createImagePatchEntry>) => {
      historyPreparationFaults.imagePatchBefore = args[0].before;
      if (historyPreparationFaults.imagePatch) {
        throw new Error('image patch preparation failed');
      }
      return actual.createImagePatchEntry(...args);
    },
  };
});

vi.mock('./history/layerSnapshot', async (importOriginal) => {
  const actual = await importOriginal<typeof LayerSnapshotModule>();
  return {
    ...actual,
    createLayerSnapshotEntry: (...args: Parameters<typeof actual.createLayerSnapshotEntry>) => {
      if (historyPreparationFaults.layerSnapshot) {
        throw new Error('layer snapshot preparation failed');
      }
      return actual.createLayerSnapshotEntry(...args);
    },
  };
});

const rasterLayer = (id: string, opts: { imageName?: string; opacity?: number } = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: opts.opacity ?? 1,
  source: { image: { height: 10, imageName: opts.imageName ?? id, width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const textRasterLayer = (id: string): CanvasLayerContract => ({
  ...(rasterLayer(id) as CanvasRasterLayerContractV2),
  source: {
    align: 'left',
    color: '#000000',
    content: id,
    fontFamily: 'sans',
    fontSize: 20,
    fontWeight: 400,
    lineHeight: 1.2,
    type: 'text',
  },
});

const makeDoc = (): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([rasterLayer('a')]),
  selectedLayerId: null,
  version: 3,
  width: 100,
});

const makeCanvas = (document: CanvasDocumentContractV3, documentRevision = 0): CanvasStateContractV3 => ({
  document,
  documentRevision,
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
});

const createFakeStore = (
  document: CanvasDocumentContractV3
): { store: EngineStore; unsubscribe: ReturnType<typeof vi.fn> } => {
  const state = {
    activeProjectId: 'p1',
    projects: [{ canvas: makeCanvas(document), id: 'p1' }],
  } as unknown as WorkbenchState;
  const unsubscribe = vi.fn();
  return {
    store: {
      dispatch: vi.fn(),
      getState: () => state,
      subscribe: () => unsubscribe,
    },
    unsubscribe,
  };
};

const createEngine = () => {
  const doc = makeDoc();
  const { store, unsubscribe } = createFakeStore(doc);
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    projectId: 'p1',
    store,
  });
  return { doc, engine, unsubscribe };
};

const createReactiveStore = (
  document: CanvasDocumentContractV3
): {
  listenerCount: () => number;
  setActiveProjectId: (projectId: string) => void;
  setDocument: (next: CanvasDocumentContractV3, documentRevision?: number) => void;
  store: EngineStore;
} => {
  let revision = 0;
  let activeProjectId = 'p1';
  let state = {
    activeProjectId,
    projects: [{ canvas: makeCanvas(document), id: 'p1' }],
  } as unknown as WorkbenchState;
  const listeners = new Set<() => void>();
  const notify = (): void => {
    for (const listener of listeners) {
      listener();
    }
  };
  return {
    listenerCount: () => listeners.size,
    setActiveProjectId: (projectId) => {
      activeProjectId = projectId;
      state = { ...state, activeProjectId };
      notify();
    },
    setDocument: (next, documentRevision = revision) => {
      revision = documentRevision;
      state = {
        activeProjectId,
        projects: [{ canvas: makeCanvas(next, documentRevision), id: 'p1' }],
      } as unknown as WorkbenchState;
      notify();
    },
    store: {
      dispatch: vi.fn(),
      getState: () => state,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
    },
  };
};

const createControllableRaf = () => {
  let nextHandle = 1;
  const callbacks = new Map<number, FrameRequestCallback>();
  return {
    cancelFrame: (handle: number): void => {
      callbacks.delete(handle);
    },
    flush: (): void => {
      const queued = [...callbacks.entries()];
      callbacks.clear();
      for (const [, cb] of queued) {
        cb(0);
      }
    },
    pendingCount: (): number => callbacks.size,
    requestFrame: (cb: FrameRequestCallback): number => {
      const handle = nextHandle++;
      callbacks.set(handle, cb);
      return handle;
    },
  };
};

const createFakeCanvas = (width = 100, height = 100): { element: HTMLCanvasElement; surface: StubRasterSurface } => {
  const surface = createTestStubRasterBackend().createSurface(width, height);
  const listeners = new Map<string, Set<(event: Event) => void>>();
  const element = {
    addEventListener: (type: string, handler: (event: Event) => void) => {
      let set = listeners.get(type);
      if (!set) {
        set = new Set();
        listeners.set(type, set);
      }
      set.add(handler);
    },
    getBoundingClientRect: () => ({
      bottom: height,
      height,
      left: 0,
      right: width,
      toJSON: () => ({}),
      top: 0,
      width,
      x: 0,
      y: 0,
    }),
    getContext: () => surface.ctx,
    height,
    releasePointerCapture: () => {},
    removeEventListener: (type: string, handler: (event: Event) => void) => {
      listeners.get(type)?.delete(handler);
    },
    setPointerCapture: () => {},
    width,
  } as unknown as HTMLCanvasElement;
  return { element, surface };
};

const createDeferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const createAbortableImageResolver = () => {
  const requests = new Map<
    string,
    { deferred: ReturnType<typeof createDeferred<Blob>>; signal: AbortSignal | undefined }
  >();
  const resolver = vi.fn((imageName: string, signal?: AbortSignal) => {
    const deferred = createDeferred<Blob>();
    requests.set(imageName, { deferred, signal });
    if (signal?.aborted) {
      return Promise.reject(signal.reason);
    }
    return new Promise<Blob>((resolve, reject) => {
      deferred.promise.then(resolve, reject);
      signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
    });
  });
  return { requests, resolver };
};

type RecordingRasterBackend = StubRasterBackend & {
  drawSourcesFor(surface: StubRasterSurface): string[];
  surfaceById(id: string): StubRasterSurface | undefined;
  surfaceId(surface: StubRasterSurface): string;
};

const createRecordingRasterBackend = (): RecordingRasterBackend => {
  const base = createTestStubRasterBackend();
  let nextSurfaceId = 1;
  const surfaces = new Map<string, StubRasterSurface>();
  return {
    ...base,
    createSurface: (width, height) => {
      const surface = base.createSurface(width, height);
      const id = `surface-${nextSurfaceId++}`;
      Object.assign(surface.canvas, { __recordingId: id });
      surfaces.set(id, surface);
      return surface;
    },
    drawSourcesFor: (surface) =>
      surface.callLog
        .filter((entry) => entry.op === 'drawImage')
        .map((entry) => (entry.args[0] as { __recordingId?: string }).__recordingId ?? 'unknown'),
    surfaceById: (id) => surfaces.get(id),
    surfaceId: (surface) => (surface.canvas as unknown as { __recordingId: string }).__recordingId,
  };
};

const recordingBitmap = (id: string, width = 10, height = 10): ImageBitmap =>
  ({ __recordingId: `bitmap-${id}`, close: vi.fn(), height, width }) as unknown as ImageBitmap;

const createStructuralFaultBackend = () => {
  const base = createTestStubRasterBackend();
  let allocationCountdown: number | null = null;
  let drawCountdown: number | null = null;
  const backend: StubRasterBackend = {
    ...base,
    createSurface: (width, height) => {
      if (allocationCountdown !== null) {
        if (allocationCountdown === 0) {
          allocationCountdown = null;
          throw new Error('structural cache allocation failed');
        }
        allocationCountdown -= 1;
      }
      const surface = base.createSurface(width, height);
      const ctx = new Proxy(surface.ctx, {
        get: (target, property, receiver) => {
          const value = Reflect.get(target, property, receiver);
          if (property !== 'drawImage' || typeof value !== 'function') {
            return value;
          }
          return (...args: unknown[]) => {
            if (drawCountdown !== null) {
              if (drawCountdown === 0) {
                drawCountdown = null;
                throw new Error('structural cache draw failed');
              }
              drawCountdown -= 1;
            }
            return Reflect.apply(value, target, args);
          };
        },
      });
      Object.defineProperty(surface, 'ctx', { value: ctx });
      return surface;
    },
  };

  return {
    armAllocation: (successfulCreates: number) => {
      allocationCountdown = successfulCreates;
    },
    armDraw: (successfulDraws: number) => {
      drawCountdown = successfulDraws;
    },
    backend,
  };
};

interface LayerCacheTestSnapshot {
  calls: RasterCallLogEntry[];
  rect: { height: number; width: number; x: number; y: number };
  surface: RasterSurface;
  version: number;
}

const snapshotLayerCache = async (engine: ReturnType<typeof createCanvasEngine>, layerId: string) => {
  const exported = await engine.exports.exportLayerPixels(layerId, { includeDisabled: true });
  if (exported.status !== 'ok') {
    throw new Error(`Expected a ready cache for ${layerId}, got ${exported.status}`);
  }
  return {
    calls: structuredClone((exported.surface as StubRasterSurface).callLog),
    rect: { ...exported.rect },
    surface: exported.surface,
    version: exported.guard.cacheVersion,
  } satisfies LayerCacheTestSnapshot;
};

const expectLayerCacheExact = async (
  engine: ReturnType<typeof createCanvasEngine>,
  layerId: string,
  expectedSnapshot: LayerCacheTestSnapshot
): Promise<void> => {
  const actual = await engine.exports.exportLayerPixels(layerId, { includeDisabled: true });
  expect(actual.status).toBe('ok');
  if (actual.status !== 'ok') {
    return;
  }
  expect(actual.surface).toBe(expectedSnapshot.surface);
  expect(actual.rect).toEqual(expectedSnapshot.rect);
  expect(actual.guard.cacheVersion).toBe(expectedSnapshot.version);
  expect((actual.surface as StubRasterSurface).callLog).toEqual(expectedSnapshot.calls);
};

const drawGraphContains = (
  surface: StubRasterSurface,
  backend: RecordingRasterBackend,
  targetId: string,
  visited = new Set<string>()
): boolean => {
  for (const sourceId of backend.drawSourcesFor(surface)) {
    if (sourceId === targetId) {
      return true;
    }
    if (visited.has(sourceId)) {
      continue;
    }
    visited.add(sourceId);
    const source = backend.surfaceById(sourceId);
    if (source && drawGraphContains(source, backend, targetId, visited)) {
      return true;
    }
  }
  return false;
};

const flushMicrotasks = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

const drainMicrotasksUntil = async (predicate: () => boolean, maxTicks = 100): Promise<void> => {
  for (let tick = 0; tick < maxTicks && !predicate(); tick += 1) {
    await Promise.resolve();
  }
};

afterEach(() => {
  reportedErrors.length = 0;
  adjustedSurfaceCacheDeleteFaults.clear();
  historyPreparationFaults.imagePatch = false;
  historyPreparationFaults.imagePatchBefore = null;
  historyPreparationFaults.layerSnapshot = false;
  vi.unstubAllGlobals();
});

describe('createCanvasEngine', () => {
  it('captures a detached clone of the exact reducer canvas contract', () => {
    const { doc, engine } = createEngine();

    const snapshot = engine.document.captureSnapshot();

    expect(snapshot?.canvas.document).toEqual(doc);
    expect(snapshot?.canvas.document).not.toBe(doc);
    expect(snapshot?.canvas.documentRevision).toBe(0);
    doc.bbox.x = 99;
    expect(snapshot?.canvas.document.bbox.x).toBe(0);
    engine.lifecycle.dispose();
  });

  it('replaces the whole document through its own mutation port', () => {
    const document = makeDoc();
    const { store } = createFakeStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const replacement = { ...makeDoc(), height: 32, width: 32 };

    engine.document.replaceDocument(replacement);

    expect(store.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ document: replacement, type: 'replaceCanvasDocument' })
    );
    engine.lifecycle.dispose();
  });

  it('returns stale and releases a partial raster snapshot when the document changes during detachment', async () => {
    const pending = createDeferred<Blob>();
    const document = makeDoc();
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });
    const documentSnapshot = engine.document.captureSnapshot();
    expect(documentSnapshot).not.toBeNull();

    const capture = engine.exports.captureRasterSnapshot(documentSnapshot!, ['a']);
    setDocument({
      ...document,
      stacks: stacksFrom([{ ...getDocumentLeaves(document)[0]!, opacity: 0.5 }]),
    });
    pending.resolve(new Blob(['pixels']));

    await expect(capture).resolves.toEqual({ status: 'stale' });
    engine.lifecycle.dispose();
  });

  it('retries a layer whose image failed only once its backoff ends, and shows it once the image loads', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
      const document = { ...makeDoc(), stacks: stacksFrom([rasterLayer('flaky')]), selectedLayerId: 'flaky' };
      const { projectId, store } = createReducerBackedStore(document);
      let available = false;
      const imageResolver = vi.fn(() =>
        available ? Promise.resolve(new Blob()) : Promise.reject(new Error('backend restarting'))
      );
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore: createSpyBitmapStore(),
        imageResolver,
        projectId,
        reportError: vi.fn(),
        store,
      });
      engine.surface.attach(createInputCanvas().element, createInputCanvas().element);
      // The shared helper waits on a timer, which these fake timers hold.
      const settle = async () => {
        for (let tick = 0; tick < 50; tick += 1) {
          await Promise.resolve();
        }
      };
      const renderFrames = async () => {
        for (let frame = 0; frame < 3; frame += 1) {
          engine.viewport.getViewport().zoomAtPoint(1 + frame / 10, { x: 0, y: 0 });
          raf.flush();
          await settle();
        }
      };

      await renderFrames();
      expect(imageResolver).toHaveBeenCalledOnce();
      expect(engine.stores.thumbnailStatus.get('flaky')).toBe('error');

      vi.advanceTimersByTime(RASTERIZATION_RETRY_BASE_MS - 1);
      await renderFrames();
      expect(imageResolver).toHaveBeenCalledOnce();

      available = true;
      vi.advanceTimersByTime(1);
      raf.flush();
      await settle();
      expect(imageResolver).toHaveBeenCalledTimes(2);
      expect(engine.stores.thumbnailStatus.get('flaky')).toBe('ready');
      await renderFrames();
      expect(imageResolver).toHaveBeenCalledTimes(2);
      engine.lifecycle.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('settles a claimed color sample on release, after the sampling gesture ends', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const layer = {
      ...rasterLayer('a'),
      source: { bitmap: { height: 20, imageName: 'paint-pixels', width: 20 }, type: 'paint' as const },
    };
    const document = { ...makeDoc(), stacks: stacksFrom([layer]), selectedLayerId: layer.id };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    let settled: string | null | undefined;
    void engine.tools.requestColorSample().then((hex) => {
      settled = hex;
    });
    overlay.fire('pointerdown', pointerAt(5, 5));
    await flushMicrotasks();
    expect(settled).toBeUndefined();

    overlay.fire('pointerup', pointerAt(5, 5, { buttons: 0 }));
    await flushMicrotasks();
    expect(settled).toBe('#000000');
    engine.lifecycle.dispose();
  });

  it('promptly aborts raster snapshot cache preparation without waiting for a stalled decode', async () => {
    const { requests, resolver } = createAbortableImageResolver();
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    const controller = new AbortController();
    const documentSnapshot = engine.document.captureSnapshot();
    const capture = engine.exports.captureRasterSnapshot(documentSnapshot!, ['a'], { signal: controller.signal });
    const settled = vi.fn();
    void capture.then(settled);
    await drainMicrotasksUntil(() => requests.has('a'));

    controller.abort(new DOMException('invoke cancelled', 'AbortError'));
    await drainMicrotasksUntil(() => settled.mock.calls.length > 0);

    expect(requests.get('a')?.signal?.aborted).toBe(true);
    expect(settled).toHaveBeenCalledWith({ status: 'aborted' });
    engine.lifecycle.dispose();
  });

  it('returns caller-owned detached layer surfaces with an idempotent release', async () => {
    const { engine } = createEngine();
    const documentSnapshot = engine.document.captureSnapshot();

    const result = await engine.exports.captureRasterSnapshot(documentSnapshot!, ['a', 'a']);

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('raster snapshot was not captured');
    }
    expect([...result.snapshot.layerSurfaces.keys()]).toEqual(['a']);
    expect(result.snapshot.layerSurfaces.get('a')?.surface).toBeDefined();
    result.snapshot.release();
    result.snapshot.release();
    expect(result.snapshot.layerSurfaces.size).toBe(0);
    engine.lifecycle.dispose();
  });

  it('releases detached raster snapshot surfaces when the engine is disposed', async () => {
    const { engine } = createEngine();
    const documentSnapshot = engine.document.captureSnapshot();
    const result = await engine.exports.captureRasterSnapshot(documentSnapshot!, ['a']);
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('raster snapshot was not captured');
    }

    engine.lifecycle.dispose();

    expect(result.snapshot.layerSurfaces.size).toBe(0);
  });
  it('invalidates edit leases across cooldown, reactivation, and disposal', async () => {
    const { engine } = createEngine();
    const first = engine.edits.tryAcquire({ kind: 'filter', layerId: 'a' });
    expect(first?.isCurrent()).toBe(true);

    await engine.lifecycle.beginCooldown();
    expect(first?.signal.aborted).toBe(true);
    expect(first?.isCurrent()).toBe(false);
    expect(engine.edits.tryAcquire({ kind: 'filter', layerId: 'a' })).toBeNull();

    engine.lifecycle.activate();
    const second = engine.edits.tryAcquire({ kind: 'select-object', layerId: 'a' });
    expect(second?.isCurrent()).toBe(true);
    engine.lifecycle.dispose();
    expect(second?.signal.aborted).toBe(true);
    expect(second?.isCurrent()).toBe(false);
  });

  it('retains base pixels when cooldown persistence fails', async () => {
    const doc = makeDoc();
    const { store } = createFakeStore(doc);
    const bitmapStore = createSpyBitmapStore();
    vi.mocked(bitmapStore.flushPendingUploads).mockRejectedValueOnce(new Error('offline'));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const before = await engine.exports.exportLayerPixels('a');
    expect(before.status).toBe('ok');

    await expect(engine.lifecycle.beginCooldown()).resolves.toBe('dirty');
    const retained = await engine.exports.exportLayerPixels('a');
    expect(retained.status).toBe('ok');
    if (before.status === 'ok' && retained.status === 'ok') {
      expect(retained.surface).toBe(before.surface);
    }
    engine.lifecycle.dispose();
  });

  it('refuses a composite export with a layer whose image is gone as unreadable, naming the layer', async () => {
    const { store } = createReactiveStore({
      ...makeDoc(),
      stacks: stacksFrom([{ ...rasterLayer('a'), name: 'Sky' }, rasterLayer('b')]),
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: (imageName) =>
        imageName === 'a' ? Promise.reject(new CanvasImageMissingError(imageName)) : Promise.resolve(new Blob()),
      projectId: 'p1',
      reportError: vi.fn(),
      store,
    });

    await expect(engine.exports.exportRasterComposite({ bounds: 'content' })).resolves.toEqual({
      layerId: 'a',
      layerName: 'Sky',
      status: 'unreadable',
    });
    engine.lifecycle.dispose();
  });

  it('refuses a composite export as not ready, never unreadable, while a layer image cannot be fetched', async () => {
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.reject(new Error('Failed to load canvas image "a" (503)')),
      projectId: 'p1',
      reportError: vi.fn(),
      store,
    });

    await expect(engine.exports.exportRasterComposite({ bounds: 'content' })).resolves.toEqual({
      status: 'not-ready',
    });
    engine.lifecycle.dispose();
  });

  it('uploads per-generation composites through the intermediate seam, not the durable one', async () => {
    const doc = makeDoc();
    const { store } = createFakeStore(doc);
    const uploadImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'durable', width: 1 }));
    const uploadIntermediateImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'transient', width: 1 }));
    const { engine } = createCoreCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createTestMutationPort(store, 'p1'),
      projectId: 'p1',
      reportError: () => undefined,
      uploadImage,
      uploadIntermediateImage,
    });

    await engine.exports.getCompositeExecutorDeps().uploadImage(new Blob());

    expect(uploadIntermediateImage).toHaveBeenCalledOnce();
    expect(uploadImage).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('does not release reconstructed pixels when reactivated during a cooldown flush', async () => {
    const doc = makeDoc();
    const { store } = createFakeStore(doc);
    const bitmapStore = createSpyBitmapStore();
    const flush = createDeferred<void>();
    vi.mocked(bitmapStore.flushPendingUploads).mockReturnValueOnce(flush.promise);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const before = await engine.exports.exportLayerPixels('a');
    const cooling = engine.lifecycle.beginCooldown();
    engine.lifecycle.activate();
    flush.resolve();
    await cooling;

    expect(engine.lifecycle.getLifecycleState()).toBe('active');
    const retained = await engine.exports.exportLayerPixels('a');
    if (before.status === 'ok' && retained.status === 'ok') {
      expect(retained.surface).toBe(before.surface);
    }
    engine.lifecycle.dispose();
  });

  it('exportLayerPixels rasterizes a visible layer and returns its cache surface plus content rect', async () => {
    const { engine } = createEngine();

    const result = await engine.exports.exportLayerPixels('a');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.rect).toEqual({ height: 10, width: 10, x: 0, y: 0 });
      expect(result.surface.width).toBe(10);
      expect(result.surface.height).toBe(10);
      expect((result.surface as StubRasterSurface).callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    }
    engine.lifecycle.dispose();
  });

  it('does not publish an older rasterization after a newer source wins', async () => {
    const first = createDeferred<Blob>();
    const second = createDeferred<Blob>();
    const blobA = new Blob(['A']);
    const blobB = new Blob(['B']);
    const bitmapA = recordingBitmap('A');
    const bitmapB = recordingBitmap('B');
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn((blob) => Promise.resolve(blob === blobA ? bitmapA : bitmapB));
    const imageResolver = vi.fn((imageName: string) => (imageName === 'A' ? first.promise : second.promise));
    const document = { ...makeDoc(), stacks: stacksFrom([rasterLayer('L', { imageName: 'A' })]) };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({ backend, imageResolver, projectId: 'p1', store });

    const exportA = engine.exports.exportLayerPixels('L');
    setDocument({ ...document, stacks: stacksFrom([rasterLayer('L', { imageName: 'B' })]) });
    const exportB = engine.exports.exportLayerPixels('L');

    second.resolve(blobB);
    const resultB = await exportB;
    expect(resultB.status).toBe('ok');
    if (resultB.status !== 'ok') {
      throw new Error('newer rasterization did not publish');
    }
    const publicationAfterB = backend.drawSourcesFor(resultB.surface as StubRasterSurface);

    first.resolve(blobA);
    expect(await exportA).toEqual({ status: 'not-ready' });
    expect(backend.drawSourcesFor(resultB.surface as StubRasterSurface)).toEqual(publicationAfterB);
    engine.lifecycle.dispose();
  });

  it('shares one rasterization between concurrent exports of the same version', async () => {
    const blob = new Blob(['shared']);
    const imageResolver = vi.fn(() => Promise.resolve(blob));
    const { store } = createFakeStore({ ...makeDoc(), stacks: stacksFrom([rasterLayer('L')]) });
    const engine = createCanvasEngine({
      backend: createRecordingRasterBackend(),
      imageResolver,
      projectId: 'p1',
      store,
    });

    const [a, b] = await Promise.all([engine.exports.exportLayerPixels('L'), engine.exports.exportLayerPixels('L')]);

    expect(imageResolver).toHaveBeenCalledTimes(1);
    expect(a.status).toBe('ok');
    expect(b.status).toBe('ok');
    engine.lifecycle.dispose();
  });

  it('exportBakedLayerPixels draws the cache through the layer transform into a document-space surface', async () => {
    const layer = {
      ...rasterLayer('a'),
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 5, y: 6 },
    };
    const { store } = createFakeStore({ ...makeDoc(), stacks: stacksFrom([layer]) });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    const result = await engine.exports.exportBakedLayerPixels('a');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.rect).toEqual({ height: 30, width: 20, x: 5, y: 6 });
      expect(result.surface.width).toBe(20);
      expect(result.surface.height).toBe(30);
      expect((result.surface as StubRasterSurface).callLog).toEqual(
        expect.arrayContaining([
          { args: [1, 0, 0, 1, 0, 0], op: 'setTransform' },
          { args: [2, 0, 0, 3, 0, 0], op: 'setTransform' },
          expect.objectContaining({ op: 'drawImage' }),
        ])
      );
    }
    engine.lifecycle.dispose();
  });

  it('invalidates a LayerExportGuard when the immutable layer contract changes', async () => {
    const document = makeDoc();
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const exported = await engine.exports.exportLayerPixels('a');
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }
    const cleanupPreview = vi.fn();
    getCanvasOperations(engine).controller.start({
      cleanupPreview,
      guard: exported.guard,
      identity: { kind: 'filter', layerId: 'a', projectId: 'p1' },
    });

    setDocument({ ...document, stacks: stacksFrom([{ ...getDocumentLeaves(document)[0]!, opacity: 0.5 }]) });

    expect(engine.exports.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    engine.lifecycle.dispose();
  });

  it('invalidates a canvas operation when its layer source changes or its layer is removed', async () => {
    for (const nextLayers of [
      [
        {
          ...getDocumentLeaves(makeDoc())[0]!,
          source: { image: { height: 10, imageName: 'replacement', width: 10 }, type: 'image' as const },
        },
      ],
      [],
    ]) {
      const document = makeDoc();
      const { setDocument, store } = createReactiveStore(document);
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        imageResolver: () => Promise.resolve(new Blob()),
        projectId: 'p1',
        store,
      });
      const exported = await engine.exports.exportLayerPixels('a');
      if (exported.status !== 'ok') {
        throw new Error('expected successful export');
      }
      const cleanupPreview = vi.fn();
      getCanvasOperations(engine).controller.start({
        cleanupPreview,
        guard: exported.guard,
        identity: { kind: 'filter', layerId: 'a', projectId: 'p1' },
      });

      setDocument({ ...document, stacks: stacksFrom(nextLayers) });

      expect(cleanupPreview).toHaveBeenCalledOnce();
      expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
      engine.lifecycle.dispose();
    }
  });

  it('keeps a project-bound LayerExportGuard current when another project becomes active', async () => {
    const document = makeDoc();
    const { setActiveProjectId, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const exported = await engine.exports.exportLayerPixels('a');
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }
    const cleanupPreview = vi.fn();
    getCanvasOperations(engine).controller.start({
      cleanupPreview,
      guard: exported.guard,
      identity: { kind: 'filter', layerId: 'a', projectId: 'p1' },
    });

    setActiveProjectId('p2');

    expect(engine.exports.isLayerExportGuardCurrent(exported.guard)).toBe(true);
    expect(cleanupPreview).not.toHaveBeenCalled();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({ status: 'active' });
    engine.lifecycle.dispose();
  });

  it('disposes the active canvas operation with the engine', async () => {
    const { engine } = createEngine();
    const exported = await engine.exports.exportLayerPixels('a');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }
    const cleanupPreview = vi.fn();
    getCanvasOperations(engine).controller.start({
      cleanupPreview,
      guard: exported.guard,
      identity: { kind: 'filter', layerId: 'a', projectId: 'p1' },
    });

    engine.lifecycle.dispose();

    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(
      getCanvasOperations(engine).controller.start({
        cleanupPreview,
        guard: exported.guard,
        identity: { kind: 'filter', layerId: 'a', projectId: 'p1' },
      })
    ).toBeNull();
  });

  it('cropLayerToBbox crops only the bbox overlap and restores exact contracts/cache snapshots on undo/redo', async () => {
    const layer = {
      ...rasterLayer('a'),
      adjustments: [
        { brightness: 0.2, contrast: -0.1, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
        { id: 'adj-hsl', isEnabled: true, saturation: 0.3, type: 'hsl' as const },
      ],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 5, y: 5 },
    };
    const document = { ...makeDoc(), bbox: { height: 8, width: 10, x: 8, y: 2 }, stacks: stacksFrom([layer]) };
    const { projectId, store } = createReducerBackedStore(document);
    const backend = createRecordingRasterBackend();
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const beforeContract = structuredClone(getDocumentLeaves(engine.document.getDocument()!)[0]!);
    const beforeExport = await engine.exports.exportLayerPixels('a');
    expect(beforeExport.status).toBe('ok');
    if (beforeExport.status !== 'ok') {
      throw new Error('expected original cache pixels');
    }
    const thumbnailListener = vi.fn();
    engine.stores.thumbnailVersion.subscribeKey('a', thumbnailListener);
    bitmapStore.markLayerDirty.mockClear();

    expect(await engine.layers.cropLayerToBbox('a')).toEqual({ status: 'cropped' });

    const afterContract = getDocumentLeaves(engine.document.getDocument()!)[0]!;
    expect(afterContract).toEqual({
      blendMode: 'normal',
      id: 'a',
      isEnabled: true,
      isLocked: false,
      name: 'a',
      opacity: 1,
      source: { bitmap: null, offset: { x: 8, y: 5 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect('adjustments' in afterContract).toBe(false);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledTimes(1);
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    const forwardExport = await engine.exports.exportLayerPixels('a');
    expect(forwardExport.status).toBe('ok');
    if (forwardExport.status !== 'ok') {
      throw new Error('expected cropped cache pixels');
    }
    expect(forwardExport.rect).toEqual({ height: 5, width: 7, x: 8, y: 5 });
    const forwardSources = backend.drawSourcesFor(forwardExport.surface as StubRasterSurface);

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(beforeContract);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledTimes(2);
    expect(thumbnailListener).toHaveBeenCalledTimes(2);
    const undoExport = await engine.exports.exportLayerPixels('a');
    expect(undoExport.status).toBe('ok');
    if (undoExport.status !== 'ok') {
      throw new Error('expected restored cache pixels');
    }
    const undoSources = backend.drawSourcesFor(undoExport.surface as StubRasterSurface);
    const beforeSnapshot = backend.surfaceById(undoSources.at(-1)!);
    expect(beforeSnapshot).toBeDefined();
    expect(backend.drawSourcesFor(beforeSnapshot!)).toContain(
      backend.surfaceId(beforeExport.surface as StubRasterSurface)
    );

    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(afterContract);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledTimes(3);
    expect(thumbnailListener).toHaveBeenCalledTimes(3);
    const redoExport = await engine.exports.exportLayerPixels('a');
    expect(redoExport.status).toBe('ok');
    if (redoExport.status !== 'ok') {
      throw new Error('expected redone cache pixels');
    }
    expect(backend.drawSourcesFor(redoExport.surface as StubRasterSurface)).toEqual(forwardSources);
    engine.lifecycle.dispose();
  });

  it('cropAllLayersToBbox crops every unlocked leaf as one undo step and replays every contract', async () => {
    const document = {
      ...makeDoc(),
      bbox: { height: 8, width: 10, x: 8, y: 2 },
      stacks: stacksFrom([
        { ...rasterLayer('a'), transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 5, y: 5 } },
        { ...rasterLayer('locked'), isLocked: true },
        { ...rasterLayer('outside'), isEnabled: false, transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 60, y: 60 } },
        {
          children: [
            {
              ...maskLayer('mask'),
              isHidden: true,
              mask: {
                bitmap: { height: 10, imageName: 'mask-img', width: 10 },
                fill: { color: '#123456', style: 'crosshatch' as const },
                offset: { x: 0, y: 0 },
              },
            },
          ],
          id: 'masks',
          isEnabled: true,
          isLocked: false,
          name: 'Masks',
          type: 'group',
        },
      ]),
    };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const before = structuredClone(engine.document.getDocument()!);

    expect(await engine.layers.cropAllLayersToBbox()).toEqual({
      cropped: 3,
      skippedLocked: 1,
      skippedUnreadable: [],
      status: 'cropped',
    });
    const cropped = structuredClone(engine.document.getDocument()!);
    expect(getDocumentLayer(cropped, 'a')).toMatchObject({ source: { bitmap: null, offset: { x: 8, y: 5 } } });
    expect(getDocumentLayer(cropped, 'outside')).toMatchObject({ isEnabled: false, source: { bitmap: null } });
    expect(getDocumentLayer(cropped, 'mask')).toMatchObject({
      isHidden: true,
      mask: { bitmap: null, fill: { color: '#123456', style: 'crosshatch' }, offset: { x: 8, y: 2 } },
    });
    expect(getDocumentLayer(cropped, 'locked')).toEqual(getDocumentLayer(before, 'locked'));
    expect(cropped.stacks.inpaint_mask.map((node) => node.id)).toEqual(['masks']);
    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: [], past: ['Crop all layers to frame'] });

    await engine.history.undo();
    expect(engine.document.getDocument()).toEqual(before);
    await engine.history.redo();
    expect(engine.document.getDocument()).toEqual(cropped);
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox reports missing, locked, unsupported, and unreadable layers explicitly', async () => {
    const polygon: CanvasLayerSourceContract = {
      fill: '#000',
      height: 10,
      kind: 'polygon',
      points: [],
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: 10,
    };
    const layers = [
      { ...rasterLayer('locked'), isLocked: true },
      { ...rasterLayer('polygon'), source: polygon },
      rasterLayer('unreadable'),
    ];
    const document = { ...makeDoc(), stacks: stacksFrom(layers) };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: (imageName) => Promise.reject(new CanvasImageMissingError(imageName)),
      projectId,
      store,
    });

    expect(await engine.layers.cropLayerToBbox('missing')).toEqual({ status: 'missing' });
    expect(await engine.layers.cropLayerToBbox('locked')).toEqual({ status: 'locked' });
    expect(await engine.layers.cropLayerToBbox('polygon')).toEqual({ status: 'unsupported' });
    expect(await engine.layers.cropLayerToBbox('unreadable')).toEqual({
      layerId: 'unreadable',
      layerName: 'unreadable',
      status: 'unreadable',
    });
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox returns not-ready without mutation when the source changes during rasterization', async () => {
    const pending = createDeferred<Blob>();
    const document = { ...makeDoc(), stacks: stacksFrom([rasterLayer('a', { imageName: 'A' })]) };
    const { setDocument, store } = createReactiveStore(document);
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });

    const crop = engine.layers.cropLayerToBbox('a');
    setDocument({ ...document, stacks: stacksFrom([rasterLayer('a', { imageName: 'B' })]) });
    pending.resolve(new Blob());

    expect(await crop).toEqual({ status: 'not-ready' });
    expect(store.dispatch).not.toHaveBeenCalled();
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('copyLayerToRaster adds a baked paint copy directly above the source layer', async () => {
    const doc = { ...makeDoc(), stacks: stacksFrom([rasterLayer('top'), rasterLayer('a')]) };
    const { projectId, store } = createReducerBackedStore(doc);
    const dispatch = store.dispatch as Mock;
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const engine = createCanvasEngine({
      backend: {
        ...base,
        createSurface: (width, height) => {
          const surface = base.createSurface(width, height);
          surfaces.push(surface);
          return surface;
        },
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    const copied = await engine.layers.copyLayerToRaster('a');
    const newId = copied.status === 'copied' ? copied.layerId : null;

    expect(newId).toMatch(/^layer-/);
    const mutation = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'applyCanvasLayerStackMutation');
    expect(mutation).toBeDefined();
    const added = mutation?.type === 'applyCanvasLayerStackMutation' ? mutation.add?.[0] : undefined;
    const addedLayer = added?.nodes[0];
    if (added && addedLayer?.type === 'raster') {
      expect(added.anchor).toMatchObject({ beforeId: 'a', projectId, stack: 'raster' });
      expect(addedLayer.id).toBe(newId);
      expect(addedLayer.name).toBe('a copy');
      expect(addedLayer.source).toEqual({ bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' });
      expect(addedLayer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    } else {
      throw new Error('expected atomic layer-stack mutation with raster copy');
    }
    expect(
      surfaces.some(
        (surface) =>
          surface.width === 10 && surface.height === 10 && surface.callLog.some((entry) => entry.op === 'drawImage')
      )
    ).toBe(true);
    engine.lifecycle.dispose();
  });

  it('setTool switches the active tool and updates the store', () => {
    const { engine } = createEngine();
    const listener = vi.fn();
    engine.stores.activeTool.subscribe(listener);

    engine.tools.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');
    expect(listener).toHaveBeenCalledTimes(1);

    engine.tools.setTool('brush');
    expect(listener).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('resize records the viewport size and clamps the device-pixel ratio', () => {
    const { engine } = createEngine();
    engine.surface.resize(800, 600, 4);
    expect(engine.viewport.getViewport().getViewportSize()).toEqual({ height: 600, width: 800 });
    expect(engine.viewport.getViewport().getDpr()).toBe(2);
    engine.lifecycle.dispose();
  });

  it('dispose is idempotent', () => {
    const { engine, unsubscribe } = createEngine();
    engine.lifecycle.dispose();
    engine.lifecycle.dispose();
    expect(unsubscribe).toHaveBeenCalledOnce();
  });
});

describe('document mirror wiring: layer reorder', () => {
  it('recomposites in the new z-order without re-rasterizing any layer', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const a = rasterLayer('a', { opacity: 0.25 });
    const b = rasterLayer('b', { opacity: 0.75 });
    const doc: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      stacks: stacksFrom([a, b]),
      selectedLayerId: null,
      version: 3,
      width: 100,
    };
    const { setDocument, store } = createReactiveStore(doc);
    const resolver = vi.fn(() => Promise.resolve(new Blob()));

    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);

    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);

    screen.surface.callLog.length = 0;

    setDocument({ ...doc, stacks: stacksFrom([b, a]) });
    raf.flush();

    expect(resolver).toHaveBeenCalledTimes(2);

    const alphaOrder = screen.surface.callLog
      .filter((entry) => entry.op === 'set' && entry.args[0] === 'globalAlpha')
      .map((entry) => entry.args[1]);
    expect(alphaOrder).toEqual([0.25, 0.75]);

    engine.lifecycle.dispose();
  });
});

describe('layer removal: adjusted-surface cache cleanup', () => {
  it("drops the removed layer's adjusted surface (no cache leak)", async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    adjustedSurfaceCacheDeletes.length = 0;

    const doc: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      stacks: stacksFrom([rasterLayer('a'), rasterLayer('b')]),
      selectedLayerId: null,
      version: 3,
      width: 100,
    };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.surface.attach(createFakeCanvas().element, createFakeCanvas().element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(adjustedSurfaceCacheDeletes).not.toContain('a');

    setDocument({ ...doc, stacks: stacksFrom([rasterLayer('b')]) });
    raf.flush();

    expect(adjustedSurfaceCacheDeletes).toContain('a');

    engine.lifecycle.dispose();
  });
});

describe('ensureLayerCaches: edit-during-rasterize race', () => {
  it('re-rasterizes with the newest source when an edit lands while a rasterize is in flight', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const doc = makeDoc();
    const { setDocument, store } = createReactiveStore(doc);

    const deferreds = new Map<string, ReturnType<typeof createDeferred<Blob>>>();
    const resolver = vi.fn((imageName: string) => {
      const deferred = createDeferred<Blob>();
      deferreds.set(imageName, deferred);
      return deferred.promise;
    });

    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);

    const thumbnailListener = vi.fn();
    const unsubscribe = engine.stores.thumbnailVersion.subscribeKey('a', thumbnailListener);
    const contentListener = vi.fn();
    const unsubscribeContent = engine.interaction.subscribe('layerPixelEpoch', contentListener);
    expect(engine.interaction.get('layerPixelEpoch')).toBe(0);

    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'a', expect.any(AbortSignal));

    setDocument({ ...doc, stacks: stacksFrom([rasterLayer('a', { imageName: 'a-v2' })]) });

    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'a-v2', expect.any(AbortSignal));

    deferreds.get('a-v2')!.resolve(new Blob());
    await flushMicrotasks();
    raf.flush();
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    expect(engine.stores.thumbnailVersion.get('a')).toBe(2);
    expect(contentListener).toHaveBeenCalledTimes(1);
    expect(engine.interaction.get('layerPixelEpoch')).toBe(1);

    deferreds.get('a')!.resolve(new Blob());
    await flushMicrotasks();
    raf.flush();
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    expect(contentListener).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenCalledTimes(2);

    unsubscribe();
    unsubscribeContent();
    engine.lifecycle.dispose();
  });
});

const paintDoc = (): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([
    {
      blendMode: 'normal',
      id: 'paint1',
      isEnabled: true,
      isLocked: false,
      name: 'paint1',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    },
  ]),
  selectedLayerId: 'paint1',
  version: 3,
  width: 100,
});

type ControlPaintHarnessOverrides = Partial<Omit<CanvasControlLayerContract, 'source' | 'type'>> & {
  source: CanvasControlLayerContract['source'];
};

const createPixelWritingBackend = <Backend extends StubRasterBackend = StubRasterBackend>(
  pixelWrites: { enabled: boolean } = { enabled: true },
  baseBackend: Backend = createTestStubRasterBackend() as Backend
): Backend => {
  return {
    ...baseBackend,
    createSurface: (width, height) => {
      const surface = baseBackend.createSurface(width, height);
      const originalCtx = surface.ctx;
      const ctx = new Proxy(originalCtx, {
        get(target, property, receiver) {
          const value = Reflect.get(target, property, receiver);
          if (property !== 'getImageData' || typeof value !== 'function') {
            return value;
          }
          return (...args: unknown[]) => {
            const imageData = Reflect.apply(value, target, args) as ImageData;
            if (pixelWrites.enabled && imageData.data.length > 0) {
              const drawCount = surface.callLog.filter((entry) => entry.op === 'drawImage').length;
              imageData.data[0] = drawCount % 256;
            }
            return imageData;
          };
        },
      });
      Object.defineProperty(surface, 'ctx', { configurable: true, value: ctx });
      return surface;
    },
  };
};

interface ControlPaintHarnessOptions {
  bitmapStoreFactory?: (deps: {
    backend: StubRasterBackend;
    projectId: string;
    store: EngineStore;
  }) => ReturnType<typeof createSpyBitmapStore>;
  pixelWrites?: { enabled: boolean };
}

const createPixelEditHarness = (
  layer: CanvasControlLayerContract | CanvasRasterLayerContractV2,
  options: ControlPaintHarnessOptions = {}
) => {
  const raf = createControllableRaf();
  vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
  vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
  vi.stubGlobal(
    'Path2D',
    class FakePath2D {
      closePath() {}
      lineTo() {}
      moveTo() {}
      quadraticCurveTo() {}
    }
  );

  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom([layer]),
    selectedLayerId: layer.id,
    version: 3,
    width: 100,
  };
  const { projectId, store } = createReducerBackedStore(document);
  const backend = createPixelWritingBackend(options.pixelWrites);
  const bitmapStore = options.bitmapStoreFactory?.({ backend, projectId, store }) ?? createSpyBitmapStore();
  const engine = createCanvasEngine({
    backend,
    bitmapStore,
    imageResolver: () => Promise.resolve(new Blob()),
    projectId,
    store,
  });
  const strokes: StrokeCommittedEvent[] = [];
  engine.tools.onStrokeCommitted((event) => strokes.push(event));
  const overlay = createInputCanvas();
  const screen = createInputCanvas();
  engine.surface.attach(screen.element, overlay.element);

  return {
    backend,
    bitmapStore,
    document,
    engine,
    layer,
    overlay,
    publishInitialCache: async () => {
      raf.flush();
      await flushMicrotasks();
      raf.flush();
      await flushMicrotasks();
    },
    raf,
    screen,
    store,
    strokes,
  };
};

const createControlPaintHarness = (
  overrides: ControlPaintHarnessOverrides,
  options: ControlPaintHarnessOptions = {}
) => {
  const { source, ...layerOverrides } = overrides;
  const layer: CanvasControlLayerContract = {
    adapter: {
      beginEndStepPct: [0.1, 0.9],
      controlMode: 'more_control',
      kind: 'controlnet',
      model: 'control-model',
      weight: 0.7,
    },
    blendMode: 'screen',
    filter: { settings: { low: 10 }, type: 'canny' },
    id: 'control',
    isEnabled: true,
    isLocked: false,
    name: 'Control',
    opacity: 0.6,
    source,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: true,
    ...layerOverrides,
  };
  return createPixelEditHarness(layer, options);
};

const createRasterImageEraserHarness = (
  overrides: Partial<Omit<CanvasRasterLayerContractV2, 'source' | 'type'>> = {},
  options: ControlPaintHarnessOptions = {}
) => {
  const layer: CanvasRasterLayerContractV2 = {
    adjustments: [
      { brightness: 0.1, contrast: 0.2, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      { id: 'adj-hsl', isEnabled: true, saturation: -0.1, type: 'hsl' as const },
    ],
    blendMode: 'screen',
    id: 'image',
    isEnabled: true,
    isLocked: false,
    name: 'Image',
    opacity: 0.6,
    source: { image: { height: 10, imageName: 'raster-image', width: 20 }, type: 'image' },
    transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 7, y: 11 },
    type: 'raster',
    ...overrides,
  };
  return createPixelEditHarness(layer, options);
};

const createControlSelectionHarness = (overrides: ControlPaintHarnessOverrides) => {
  const selectionPixelWrites = { enabled: true };
  return {
    ...createControlPaintHarness(overrides, { pixelWrites: selectionPixelWrites }),
    setSelectionPixelWrites: (enabled: boolean) => {
      selectionPixelWrites.enabled = enabled;
    },
  };
};

const createSpyBitmapStore = (): BitmapStore & {
  markLayerDirty: Mock<(layerId: string) => void>;
  releaseSuspendedLayer: Mock<() => void>;
  reset: Mock<() => void>;
  suspendLayer: Mock<(layerId: string) => () => void>;
} => {
  const releaseSuspendedLayer = vi.fn();
  return {
    discardLayer: vi.fn(),
    dispose: vi.fn(),
    flushPendingUploads: vi.fn(() => Promise.resolve()),
    hasPendingClear: () => false,
    hasPendingWork: () => false,
    isSelfEcho: () => false,
    markLayerDirty: vi.fn<(layerId: string) => void>(),
    releaseSuspendedLayer,
    reset: vi.fn<() => void>(),
    suspendLayer: vi.fn(() => releaseSuspendedLayer),
  };
};

const createRealControlPersistenceHarness = async (
  uploadImage: (blob: Blob) => Promise<{ height: number; imageName: string; width: number }>
) => {
  let placed: { offset: { x: number; y: number }; surface: RasterSurface } | null = null;
  const encodeSurface = vi.fn(() => Promise.resolve(new Blob(['control-pixels'], { type: 'image/png' })));
  const h = createControlPaintHarness(
    { source: { bitmap: { height: 10, imageName: 'control-paint', width: 10 }, type: 'paint' } },
    {
      bitmapStoreFactory: ({ projectId, store }) => {
        const real = createBitmapStore({
          debounceMs: 1500,
          dispatch: createTestMutationPort(store, projectId).dispatch,
          encodeSurface,
          getLayerSource: (layerId) => {
            const layer = getDocumentLeaves(
              store.getState().projects.find((project) => project.id === projectId)?.canvas.document
            ).find((candidate) => candidate.id === layerId);
            return layer?.type === 'control' ? layer.source : null;
          },
          getLayerSurface: () => placed,
          hashBlob: (blob) => blob.text(),
          maxUploadAttempts: 1,
          retryDelaysMs: [],
          sleep: () => Promise.resolve(),
          uploadImage,
        });
        const releaseSuspendedLayer = vi.fn();
        return {
          ...real,
          markLayerDirty: vi.fn(real.markLayerDirty),
          releaseSuspendedLayer,
          reset: vi.fn(real.reset),
          suspendLayer: vi.fn((layerId: string) => {
            const release = real.suspendLayer(layerId);
            return () => {
              releaseSuspendedLayer();
              release();
            };
          }),
        };
      },
    }
  );
  await h.publishInitialCache();
  const exported = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
  if (exported.status !== 'ok') {
    throw new Error(`expected ready direct control cache, got ${exported.status}`);
  }
  placed = { offset: { x: exported.rect.x, y: exported.rect.y }, surface: exported.surface };
  return { ...h, encodeSurface };
};

const createInputCanvas = (
  width = 100,
  height = 100
): {
  element: HTMLCanvasElement;
  fire: (type: string, event: Partial<PointerEvent>) => void;
  listenerCount: (type: string) => number;
  surface: StubRasterSurface;
} => {
  const surface = createTestStubRasterBackend().createSurface(width, height);
  const listeners = new Map<string, Set<(event: Event) => void>>();
  const element = {
    addEventListener: (type: string, handler: (event: Event) => void) => {
      const set = listeners.get(type) ?? new Set();
      set.add(handler);
      listeners.set(type, set);
    },
    getBoundingClientRect: () => ({ bottom: height, height, left: 0, right: width, top: 0, width, x: 0, y: 0 }),
    getContext: () => surface.ctx,
    height,
    releasePointerCapture: () => {},
    removeEventListener: (type: string, handler: (event: Event) => void) => {
      listeners.get(type)?.delete(handler);
    },
    setPointerCapture: () => {},
    width,
  } as unknown as HTMLCanvasElement;
  const fire = (type: string, event: Partial<PointerEvent>): void => {
    for (const handler of listeners.get(type) ?? []) {
      handler({ preventDefault: () => {}, ...event } as unknown as Event);
    }
  };
  return { element, fire, listenerCount: (type) => listeners.get(type)?.size ?? 0, surface };
};

const pointerAt = (x: number, y: number, opts: { button?: number; buttons?: number } = {}): Partial<PointerEvent> => ({
  altKey: false,
  button: opts.button ?? 0,
  buttons: opts.buttons ?? 1,
  clientX: x,
  clientY: y,
  ctrlKey: false,
  metaKey: false,
  pointerId: 1,
  pointerType: 'mouse',
  pressure: 0.5,
  shiftKey: false,
  timeStamp: 0,
});

const putImageDataCalls = (surfaces: StubRasterSurface[]): { image: unknown; x: unknown; y: unknown }[] =>
  surfaces.flatMap((surface) =>
    surface.callLog
      .filter((entry) => entry.op === 'putImageData')
      .map((entry) => ({ image: entry.args[0], x: entry.args[1], y: entry.args[2] }))
  );

describe('engine-owned pixel editing', () => {
  it.each(['eraser', 'brush'] as const)(
    'materializes an image layer plus %s stroke as one reversible edit',
    async (tool) => {
      const h = createRasterImageEraserHarness();
      await h.publishInitialCache();
      const before = structuredClone(getDocumentLeaves(h.engine.document.getDocument()!)[0]);

      h.engine.tools.setTool(tool);
      h.overlay.fire('pointerdown', pointerAt(20, 20));
      h.overlay.fire('pointermove', pointerAt(25, 25));
      h.overlay.fire('pointerup', pointerAt(25, 25, { buttons: 0 }));

      const after = structuredClone(getDocumentLeaves(h.engine.document.getDocument()!)[0]);
      expect(getDocumentLeaves(h.engine.document.getDocument()!)).toHaveLength(1);
      expect(after).toMatchObject({
        id: 'image',
        source: { type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'raster',
      });
      expect(after).not.toHaveProperty('adjustments');
      expect(h.strokes).toHaveLength(1);
      expect(h.strokes[0]).toMatchObject({ layerId: 'image', tool });
      expect(h.bitmapStore.suspendLayer).toHaveBeenCalledWith('image');
      expect(h.bitmapStore.markLayerDirty).toHaveBeenCalledWith('image');
      expect(h.engine.stores.canUndo.get()).toBe(true);

      await h.engine.history.undo();
      expect(getDocumentLeaves(h.engine.document.getDocument()!)[0]).toEqual(before);
      expect(h.engine.stores.canRedo.get()).toBe(true);

      await h.engine.history.redo();
      expect(getDocumentLeaves(h.engine.document.getDocument()!)[0]).toEqual(after);
      h.engine.lifecycle.dispose();
    }
  );

  it('draws a Shift-click line from the last stroke end until a replay or document swap forgets it', async () => {
    const h = createRasterImageEraserHarness({ adjustments: [], blendMode: 'normal', opacity: 1 });
    await h.publishInitialCache();
    let xs: number[] = [];
    vi.stubGlobal(
      'Path2D',
      class RecordingPath2D {
        closePath() {}
        lineTo(x: number) {
          xs.push(x);
        }
        moveTo(x: number) {
          xs.push(x);
        }
        quadraticCurveTo(_cx: number, _cy: number, x: number) {
          xs.push(x);
        }
      }
    );
    const strokeSpan = (fire: () => void): number => {
      xs = [];
      fire();
      return Math.max(...xs) - Math.min(...xs);
    };
    const click = (x: number, shiftKey: boolean) => () => {
      h.overlay.fire('pointerdown', { ...pointerAt(x, 20), shiftKey });
      h.overlay.fire('pointerup', { ...pointerAt(x, 20, { buttons: 0 }), shiftKey });
    };

    h.engine.tools.setTool('brush');
    const dab = strokeSpan(click(12, false));
    h.engine.tools.setTool('eraser');
    expect(strokeSpan(click(42, true))).toBeGreaterThanOrEqual(dab + 30);
    expect(h.strokes).toHaveLength(2);

    await h.engine.history.undo();
    expect(strokeSpan(click(12, true))).toBeCloseTo(dab, 6);

    h.engine.document.replaceDocument(structuredClone(h.engine.document.getDocument()!));
    await h.publishInitialCache();
    expect(strokeSpan(click(42, true))).toBeCloseTo(dab, 6);
    h.engine.lifecycle.dispose();
  });

  it('persists and releases a materialized image erase when a thumbnail observer throws', async () => {
    const h = createRasterImageEraserHarness();
    await h.publishInitialCache();
    const unsubscribe = h.engine.stores.thumbnailVersion.subscribe(() => {
      throw new Error('thumbnail observer failed');
    });

    h.engine.tools.setTool('eraser');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(25, 25));

    expect(() => h.overlay.fire('pointerup', pointerAt(25, 25, { buttons: 0 }))).not.toThrow();
    expect(getDocumentLeaves(h.engine.document.getDocument()!)[0]).toMatchObject({
      source: { type: 'paint' },
      type: 'raster',
    });
    expect(h.bitmapStore.markLayerDirty).toHaveBeenCalledWith('image');
    expect(h.bitmapStore.markLayerDirty.mock.invocationCallOrder[0]).toBeLessThan(
      h.bitmapStore.releaseSuspendedLayer.mock.invocationCallOrder[0]!
    );
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.canUndo.get()).toBe(true);

    unsubscribe();
    h.engine.lifecycle.dispose();
  });

  it('rolls a byte-identical image-layer erase back without converting the layer', async () => {
    const h = createRasterImageEraserHarness({}, { pixelWrites: { enabled: false } });
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'image');

    h.engine.tools.setTool('eraser');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(25, 25));
    h.overlay.fire('pointerup', pointerAt(25, 25, { buttons: 0 }));

    expect(h.engine.document.getDocument()).toEqual(before);
    const restored = await h.engine.exports.exportLayerPixels('image', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }
    expect(h.strokes).toHaveLength(0);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('rolls a cancelled image-layer erase back without converting the layer', async () => {
    const h = createRasterImageEraserHarness();
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'image');

    h.engine.tools.setTool('eraser');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(25, 25));
    h.overlay.fire('pointercancel', pointerAt(25, 25, { buttons: 0 }));

    expect(h.engine.document.getDocument()).toEqual(before);
    const restored = await h.engine.exports.exportLayerPixels('image', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }
    expect(h.strokes).toHaveLength(0);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('does not erase or auto-create while an image-layer cache is still loading', () => {
    const h = createRasterImageEraserHarness();
    h.raf.flush();
    const before = structuredClone(h.engine.document.getDocument());

    h.engine.tools.setTool('eraser');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointerup', pointerAt(20, 20, { buttons: 0 }));

    expect(h.engine.document.getDocument()).toEqual(before);
    expect(getDocumentLeaves(h.engine.document.getDocument()!)).toHaveLength(1);
    expect(h.strokes).toHaveLength(0);
    expect(h.bitmapStore.suspendLayer).not.toHaveBeenCalled();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('brushes an empty control in place and never creates a raster layer', () => {
    const h = createControlPaintHarness({ source: { bitmap: null, type: 'paint' } });
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(40, 40));
    h.overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    expect(getDocumentLeaves(h.engine.document.getDocument()!)).toHaveLength(1);
    expect(getDocumentLeaves(h.engine.document.getDocument()!)[0]).toMatchObject({ id: 'control', type: 'control' });
    expect(h.strokes).toHaveLength(1);
    expect(h.strokes[0]!.layerId).toBe('control');
    expect(h.bitmapStore.markLayerDirty).toHaveBeenCalledWith('control');
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledWith('control');
    expect(h.bitmapStore.markLayerDirty.mock.invocationCallOrder[0]).toBeLessThan(
      h.bitmapStore.releaseSuspendedLayer.mock.invocationCallOrder[0]!
    );
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('keeps a pending direct-control upload suspended through cancel, then resumes the preserved dirty work', async () => {
    const uploadImage = vi.fn(() => Promise.resolve({ height: 10, imageName: 'resumed-control', width: 10 }));
    const h = await createRealControlPersistenceHarness(uploadImage);
    vi.useFakeTimers();
    try {
      h.bitmapStore.markLayerDirty('control');
      h.engine.tools.setTool('brush');
      h.overlay.fire('pointerdown', pointerAt(20, 20));
      h.overlay.fire('pointermove', pointerAt(30, 30));
      const barrier = h.bitmapStore.flushPendingUploads();
      let settled = false;
      void barrier.then(() => {
        settled = true;
      });

      await vi.advanceTimersByTimeAsync(3000);
      expect(settled).toBe(false);
      expect(h.encodeSurface).not.toHaveBeenCalled();
      expect(uploadImage).not.toHaveBeenCalled();

      h.overlay.fire('pointercancel', pointerAt(30, 30, { buttons: 0 }));
      await drainMicrotasksUntil(() => settled);

      expect(settled).toBe(true);
      expect(h.encodeSurface).toHaveBeenCalledOnce();
      expect(uploadImage).toHaveBeenCalledOnce();
      expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
      expect(h.engine.stores.canUndo.get()).toBe(false);
      expect(h.strokes).toHaveLength(0);
      h.engine.lifecycle.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('restores a direct control stroke when image-patch history preparation fails', async () => {
    const h = createControlPaintHarness({ source: { bitmap: null, type: 'paint' } });
    const before = structuredClone(h.engine.document.getDocument());
    const beforeExport = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(40, 40));
    historyPreparationFaults.imagePatch = true;

    expect(() => h.overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }))).toThrow(
      'image patch preparation failed'
    );

    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.strokes).toHaveLength(0);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe(beforeExport.status);
    h.engine.lifecycle.dispose();
  });

  it('finishes document-replacement cleanup after materialized gesture rollback throws', async () => {
    const h = createControlPaintHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(8, 8));
    h.overlay.fire('pointermove', pointerAt(12, 12));
    adjustedSurfaceCacheDeleteFaults.add('control');
    const replacement = { ...h.document, width: h.document.width + 1 };

    expect(() => h.store.dispatch({ document: replacement, type: 'replaceCanvasDocument' })).toThrow(
      'adjusted surface cache delete failed'
    );

    expect(h.engine.document.getDocument()).toEqual(replacement);
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.reset).toHaveBeenCalledOnce();
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.strokes).toHaveLength(0);
    h.engine.lifecycle.dispose();
  });

  it('does not clean up a materialized gesture merely because another project becomes active', async () => {
    const h = createControlPaintHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
    });
    await h.publishInitialCache();
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(8, 8));
    h.overlay.fire('pointermove', pointerAt(12, 12));
    adjustedSurfaceCacheDeleteFaults.add('control');

    expect(() => h.store.dispatch({ type: 'createProject' })).not.toThrow();

    expect(h.store.getState().activeProjectId).not.toBe(h.store.getState().projects[0]!.id);
    expect(h.bitmapStore.releaseSuspendedLayer).not.toHaveBeenCalled();
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.strokes).toHaveLength(0);

    adjustedSurfaceCacheDeleteFaults.delete('control');
    h.store.dispatch({ projectId: h.store.getState().projects[0]!.id, type: 'switchProject' });
    h.overlay.fire('pointercancel', pointerAt(8, 8, { buttons: 0 }));
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    h.engine.lifecycle.dispose();
  });

  it('finishes disposal after direct gesture rollback throws and remains idempotent', async () => {
    const h = createControlPaintHarness({
      source: { bitmap: { height: 10, imageName: 'control-paint', width: 10 }, type: 'paint' },
    });
    await h.publishInitialCache();
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(8, 8));
    h.overlay.fire('pointermove', pointerAt(12, 12));
    adjustedSurfaceCacheDeleteFaults.add('control');

    expect(() => h.engine.lifecycle.dispose()).toThrow('adjusted surface cache delete failed');

    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.dispose).toHaveBeenCalledOnce();
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.strokes).toHaveLength(0);
    expect(() => h.engine.lifecycle.dispose()).not.toThrow();
    expect(h.bitmapStore.dispose).toHaveBeenCalledOnce();
  });
});

describe('engine-owned history: stroke → undo → redo', () => {
  const drawStroke = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    // The engine builds real `Path2D`s for stroke outlines; node lacks it.
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const bitmapStore = createSpyBitmapStore();
    const mutationPort = createTestMutationPort(store, 'p1');
    const commitEdit = vi.fn(mutationPort.commitEdit);
    mutationPort.commitEdit = commitEdit;

    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort,
      projectId: 'p1',
      store,
    });
    const strokes: StrokeCommittedEvent[] = [];
    engine.tools.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    return { bitmapStore, commitEdit, engine, strokes, surfaces };
  };

  it('records a stroke and restores before/after pixels on undo/redo', async () => {
    const { bitmapStore, commitEdit, engine, strokes, surfaces } = drawStroke();

    expect(strokes).toHaveLength(1);
    const event = strokes[0]!;
    expect(event.layerId).toBe('paint1');
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    expect(commitEdit).toHaveBeenCalledOnce();
    expect(commitEdit).toHaveBeenCalledWith({ kind: 'paint' });
    const dirtyAfterStroke = bitmapStore.markLayerDirty.mock.calls.length;
    expect(dirtyAfterStroke).toBeGreaterThanOrEqual(1);

    await engine.history.undo();
    const undoPut = putImageDataCalls(surfaces).find((call) => call.image === event.beforeImageData);
    expect(undoPut).toBeDefined();
    expect(undoPut!.x).toBe(0);
    expect(undoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    expect(bitmapStore.markLayerDirty.mock.calls.length).toBeGreaterThan(dirtyAfterStroke);

    await engine.history.redo();
    const redoPut = putImageDataCalls(surfaces).find((call) => call.image === event.afterImageData);
    expect(redoPut).toBeDefined();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    expect(commitEdit).toHaveBeenCalledOnce();

    engine.lifecycle.dispose();
  });

  it('clears history when the document is replaced', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { setDocument, store } = createReactiveStore(paintDoc());
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    expect(engine.stores.canUndo.get()).toBe(true);

    const replaced = { ...paintDoc(), height: 200, width: 200 };
    setDocument(replaced);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('re-rasterizes and resets persistence bookkeeping on a revision-bump swap that reuses a layer id with a different source', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const docV1: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      stacks: stacksFrom([rasterLayer('a', { imageName: 'src-v1' })]),
      selectedLayerId: 'a',
      version: 3,
      width: 100,
    };
    const { setDocument, store } = createReactiveStore(docV1);
    const resolver = vi.fn((_imageName: string) => Promise.resolve(new Blob()));
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);

    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'src-v1', expect.any(AbortSignal));

    setDocument({ ...docV1, stacks: stacksFrom([rasterLayer('a', { imageName: 'src-v2' })]) }, 1);

    expect(bitmapStore.reset).toHaveBeenCalledTimes(1);

    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'src-v2', expect.any(AbortSignal));

    engine.lifecycle.dispose();
  });
});

describe('engine-owned history: undo/redo guarded during an active gesture', () => {
  it('no-ops undo/redo mid-stroke, then works after the gesture ends', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const strokes: StrokeCommittedEvent[] = [];
    engine.tools.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    expect(engine.stores.canUndo.get()).toBe(true);

    overlay.fire('pointerdown', pointerAt(50, 50));
    overlay.fire('pointermove', pointerAt(60, 60));
    const putsMidGesture = putImageDataCalls(surfaces);

    await engine.history.undo();
    await engine.history.redo();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(putImageDataCalls(surfaces)).toEqual(putsMidGesture);

    overlay.fire('pointerup', pointerAt(60, 60, { buttons: 0 }));
    expect(strokes).toHaveLength(2);
    await engine.history.undo();
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[1]!.beforeImageData)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.lifecycle.dispose();
  });
});

describe('commitStructural', () => {
  const forward: EngineTestAction = { id: 'a', type: 'setCanvasSelectedLayer' };
  const inverse: EngineTestAction = { id: null, type: 'setCanvasSelectedLayer' };

  it('keeps a removed image held while Canvas can undo the removal', () => {
    const layer = rasterLayer('L', { imageName: 'older-intermediate.png' });
    const { projectId, store } = createReducerBackedStore({ ...makeDoc(), stacks: stacksFrom([layer]) });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(
      engine.layers.commitStructural(
        'Remove layer',
        { ids: ['L'], type: 'removeCanvasLayers' },
        { anchor: stackTopAnchor(projectId), layer, type: 'addCanvasLayer' }
      )
    ).toEqual({ status: 'committed' });
    expect(getDocumentLeaves(engine.document.getDocument())).toEqual([]);
    expect(engine.history.getHeldAssetRefs().images).toContain('older-intermediate.png');
    engine.history.clearHistory();
    expect(engine.history.getHeldAssetRefs().images).toEqual([]);
    engine.lifecycle.dispose();
  });

  it('dispatches forward immediately and records a reversible history entry', async () => {
    const { store } = createFakeStore(makeDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(engine.layers.canCommitStructural()).toBe(true);
    expect(engine.layers.commitStructural('Select layer', forward, inverse)).toEqual({ status: 'committed' });
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenNthCalledWith(1, forward);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    await engine.history.undo();
    expect(dispatch).toHaveBeenNthCalledWith(2, inverse);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    await engine.history.redo();
    expect(dispatch).toHaveBeenNthCalledWith(3, forward);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
    expect(engine.layers.canCommitStructural()).toBe(false);
    expect(engine.layers.commitStructural('Select layer', forward, inverse)).toEqual({ status: 'not-ready' });
  });
});

const controlLayerForThumbnail = (id: string): CanvasLayerContract =>
  ({
    adapter: {
      beginEndStepPct: [0, 1],
      controlMode: null,
      kind: 'controlnet',
      model: null,
      weight: 1,
    },
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: true,
  }) as CanvasLayerContract;

const thumbnailMaskLayer = (id: string, type: 'inpaint_mask' | 'regional_guidance'): CanvasLayerContract =>
  ({
    autoNegative: true,
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    mask: {
      bitmap: { height: 10, imageName: id, width: 10 },
      fill: { color: '#e07575', style: 'diagonal' },
    },
    name: id,
    negativePrompt: null,
    opacity: 1,
    positivePrompt: null,
    referenceImages: [],
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type,
  }) as CanvasLayerContract;

const largeThumbnailLayer = (
  kind: 'raster' | 'control' | 'inpaint_mask' | 'regional_guidance'
): CanvasLayerContract => {
  const image = { height: 2048, imageName: `large-${kind}`, width: 4096 };
  if (kind === 'raster') {
    return {
      ...rasterLayer('a'),
      adjustments: [
        { brightness: 0.25, contrast: -0.1, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
        { id: 'adj-hsl', isEnabled: true, saturation: 0.3, type: 'hsl' as const },
      ],
      source: { bitmap: image, offset: { x: -300, y: 250 }, type: 'paint' },
    } as CanvasRasterLayerContractV2;
  }
  if (kind === 'control') {
    return { ...controlLayerForThumbnail('a'), source: { image, type: 'image' } } as CanvasLayerContract;
  }
  const mask = thumbnailMaskLayer('a', kind);
  return {
    ...mask,
    mask: { ...('mask' in mask ? mask.mask : {}), bitmap: image, offset: { x: -300, y: 250 } },
  } as CanvasLayerContract;
};

const createThumbnailTarget = (): {
  calls: { args: unknown[]; op: string }[];
  target: HTMLCanvasElement;
} => {
  const calls: { args: unknown[]; op: string }[] = [];
  const ctx = {
    clearRect: (...args: unknown[]) => calls.push({ args, op: 'clearRect' }),
    createPattern: (...args: unknown[]) => {
      calls.push({ args, op: 'createPattern' });
      return 'checker-pattern';
    },
    drawImage: (...args: unknown[]) => calls.push({ args, op: 'drawImage' }),
    fillRect: (...args: unknown[]) => calls.push({ args, op: 'fillRect' }),
    set globalAlpha(value: unknown) {
      calls.push({ args: [value], op: 'globalAlpha' });
    },
    set fillStyle(value: unknown) {
      calls.push({ args: [value], op: 'fillStyle' });
    },
  };
  const target = { getContext: () => ctx, height: 0, width: 0 } as unknown as HTMLCanvasElement;
  return { calls, target };
};

describe('drawLayerThumbnail', () => {
  it('scales the layer cache into the target and reports success', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const { calls, target } = createThumbnailTarget();
    expect(engine.previews.drawLayerThumbnail('a', target, 96)).toBe(true);
    expect(target.width).toBe(10);
    expect(target.height).toBe(10);
    expect(calls.map((call) => call.op)).toEqual([
      'clearRect',
      'createPattern',
      'fillStyle',
      'fillRect',
      'globalAlpha',
      'drawImage',
    ]);

    engine.lifecycle.dispose();
  });

  it("publishes a spatial layer's adjustment status and redraws its thumbnail when the result lands", async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const layer = {
      ...rasterLayer('a'),
      adjustments: [{ amount: 40, id: 'adj-cl', isEnabled: true, size: 40, type: 'clarity' as const }],
    } as CanvasRasterLayerContractV2;
    const { store } = createReactiveStore({ ...makeDoc(), stacks: stacksFrom([layer]) });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      createSpatialExecutor: createInlineSpatialExecutor,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.surface.attach(createInputCanvas().element, createInputCanvas().element);
    raf.flush();
    await flushMicrotasks();
    const statuses: string[] = [];
    engine.interaction.subscribeLayerAdjustmentStatus('a', () =>
      statuses.push(engine.interaction.getLayerAdjustmentStatus('a'))
    );
    const thumbnailBefore = engine.interaction.getLayerThumbnailVersion('a');
    raf.flush();
    expect(engine.interaction.getLayerAdjustmentStatus('a')).toBe('computing');

    await flushMicrotasks();
    expect(statuses).toEqual(['computing', 'idle']);
    const thumbnailAfter = engine.interaction.getLayerThumbnailVersion('a');
    expect(thumbnailAfter).not.toBe(thumbnailBefore);
    expect(thumbnailAfter).toBeLessThan(0);
    engine.lifecycle.dispose();
  });

  it.each(['raster', 'control', 'inpaint_mask', 'regional_guidance'] as const)(
    'bounds every %s thumbnail effect allocation before processing a large source',
    async (kind) => {
      const layer = largeThumbnailLayer(kind);
      const backend = createTestStubRasterBackend();
      const createSurface = vi.spyOn(backend, 'createSurface');
      const { store } = createReactiveStore({ ...makeDoc(), stacks: stacksFrom([layer]) });
      const engine = createCanvasEngine({
        backend,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId: 'p1',
        store,
      });
      await engine.previews.requestLayerThumbnail('a');
      createSurface.mockClear();

      const { target } = createThumbnailTarget();
      expect(engine.previews.drawLayerThumbnail('a', target, 96)).toBe(true);

      expect(target.width).toBe(96);
      expect(target.height).toBe(48);
      expect(createSurface.mock.calls.length).toBeGreaterThan(0);
      expect(createSurface.mock.calls.every(([width, height]) => width <= 96 && height <= 96)).toBe(true);
      const scratch = createSurface.mock.results
        .map((result) => result.value as StubRasterSurface)
        .find((surface) =>
          surface.callLog.some(
            (call) =>
              call.op === 'drawImage' &&
              call.args[1] === 0 &&
              call.args[2] === 0 &&
              call.args[3] === 96 &&
              call.args[4] === 48
          )
        );
      expect(scratch).toBeDefined();
      engine.lifecycle.dispose();
    }
  );
});

describe('requestLayerThumbnail', () => {
  it('starts a request for its bound project while another project is active', async () => {
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const { setActiveProjectId, store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    setActiveProjectId('p2');

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    expect(resolver).toHaveBeenCalledOnce();
    expect(engine.stores.thumbnailStatus.get('a')).toBe('ready');
    engine.lifecycle.dispose();
  });

  it('rasterizes while detached and publishes thumbnail readiness', async () => {
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();
    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(engine.stores.thumbnailStatus.get('a')).toBe('ready');
    expect(engine.previews.drawLayerThumbnail('a', createThumbnailTarget().target, 96)).toBe(true);
    engine.lifecycle.dispose();
  });

  it('releases a detached thumbnail bitmap when its layer source is replaced', async () => {
    const bitmap = recordingBitmap('detached-replaced');
    const backend = createTestStubRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
    const doc = { ...makeDoc(), stacks: stacksFrom([rasterLayer('a', { imageName: 'old' })]) };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    setDocument({ ...doc, stacks: stacksFrom([rasterLayer('a', { imageName: 'new' })]) });

    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('deduplicates concurrent requests for the same source version', async () => {
    const pending = createDeferred<Blob>();
    const resolver = vi.fn(() => pending.promise);
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    const first = engine.previews.requestLayerThumbnail('a');
    const second = engine.previews.requestLayerThumbnail('a');
    expect(resolver).toHaveBeenCalledTimes(1);

    pending.resolve(new Blob());
    expect(await Promise.all([first, second])).toEqual(['ready', 'ready']);
    engine.lifecycle.dispose();
  });

  it('keeps loading until the latest source wins', async () => {
    const { requests, resolver } = createAbortableImageResolver();
    const doc = makeDoc();
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    const oldRequest = engine.previews.requestLayerThumbnail('a');
    setDocument({ ...doc, stacks: stacksFrom([rasterLayer('a', { imageName: 'a-v2' })]) });
    expect(requests.get('a')?.signal?.aborted).toBe(true);
    const newRequest = engine.previews.requestLayerThumbnail('a');

    expect(await oldRequest).toBe('stale');
    expect(engine.stores.thumbnailStatus.get('a')).toBe('loading');

    requests.get('a-v2')?.deferred.resolve(new Blob());
    expect(await newRequest).toBe('ready');
    expect(engine.stores.thumbnailStatus.get('a')).toBe('ready');
    engine.lifecycle.dispose();
  });

  it('reports failures through status and diagnostics, then retries', async () => {
    const resolver = vi.fn().mockRejectedValueOnce(new Error('decode failed')).mockResolvedValueOnce(new Blob());
    const { store } = createReactiveStore(makeDoc());
    const reportError = vi.fn();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      reportError,
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('error');
    expect(engine.stores.thumbnailStatus.get('a')).toBe('error');
    expect(reportError).toHaveBeenCalledWith(
      expect.objectContaining({
        area: 'canvas-engine',
        context: expect.objectContaining({
          error: expect.objectContaining({ message: 'Canvas image "a" could not be fetched.' }),
          layerId: 'a',
        }),
        message: 'Layer thumbnail rasterization failed',
        namespace: 'canvas',
        projectId: 'p1',
      })
    );

    const retry = engine.previews.requestLayerThumbnail('a');
    expect(engine.stores.thumbnailStatus.get('a')).toBe('loading');
    expect(await retry).toBe('ready');
    expect(resolver).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('clears state and rejects stale completion after deletion', async () => {
    const { requests, resolver } = createAbortableImageResolver();
    const doc = makeDoc();
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    const request = engine.previews.requestLayerThumbnail('a');
    expect(engine.stores.thumbnailStatus.get('a')).toBe('loading');
    setDocument({ ...doc, stacks: stacksFrom([]) });
    expect(requests.get('a')?.signal?.aborted).toBe(true);
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();

    expect(await request).toBe('stale');
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();
    expect(engine.previews.drawLayerThumbnail('a', createThumbnailTarget().target, 96)).toBe(false);
    engine.lifecycle.dispose();
  });

  it('clears state and rejects stale completion after disposal', async () => {
    const { requests, resolver } = createAbortableImageResolver();
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    const request = engine.previews.requestLayerThumbnail('a');
    engine.lifecycle.dispose();
    expect(requests.get('a')?.signal?.aborted).toBe(true);
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();

    expect(await request).toBe('stale');
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();
  });
});

const twoPaintDoc = (): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([
    {
      blendMode: 'multiply',
      id: 'upper',
      isEnabled: true,
      isLocked: false,
      name: 'upper',
      opacity: 0.5,
      source: { bitmap: { height: 40, imageName: 'upper-bmp', width: 40 }, offset: { x: 0, y: 0 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 30, y: 40 },
      type: 'raster',
    },
    {
      blendMode: 'normal',
      id: 'below',
      isEnabled: true,
      isLocked: false,
      name: 'below',
      opacity: 1,
      source: { bitmap: { height: 60, imageName: 'below-bmp', width: 60 }, offset: { x: 0, y: 0 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
      type: 'raster',
    },
  ]),
  selectedLayerId: 'upper',
  version: 3,
  width: 100,
});

const maskLayer = (id: string): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

describe('mergeLayerDown', () => {
  it('composites below then the transformed upper cache, and dispatches mergeCanvasLayersDown', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(twoPaintDoc());
    const dispatch = store.dispatch as Mock;
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const surfacesBeforeMerge = surfaces.length;
    expect(await engine.layers.mergeLayerDown('upper')).toBe('merged');

    const mergeCall = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'mergeCanvasLayersDown');
    expect(mergeCall).toEqual({
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
      type: 'mergeCanvasLayersDown',
      upperLayerId: 'upper',
    });

    const merged = surfaces
      .slice(surfacesBeforeMerge)
      .find((surface) => surface.callLog.some((entry) => entry.op === 'clearRect'));
    expect(merged).toBeDefined();
    expect(merged!.width).toBe(60);
    expect(merged!.height).toBe(60);
    const log = merged!.callLog;

    const expectedMatrix = mergeDownMatrix(
      { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
      { rotation: 0, scaleX: 1, scaleY: 1, x: 30, y: 40 }
    )!;
    // upper{30,40} → below-local via inverse(below{10,20}) = translate(20,20).
    expect(expectedMatrix.e).toBe(20);
    expect(expectedMatrix.f).toBe(20);
    const mergedOrigin = { x: 0, y: 0 };
    const matrixIndex = log.findIndex(
      (entry) =>
        entry.op === 'setTransform' &&
        entry.args[0] === expectedMatrix.a &&
        entry.args[4] === expectedMatrix.e - mergedOrigin.x &&
        entry.args[5] === expectedMatrix.f - mergedOrigin.y
    );
    expect(matrixIndex).toBeGreaterThan(-1);

    const drawIndices = log.reduce<number[]>((acc, entry, index) => {
      if (entry.op === 'drawImage') {
        acc.push(index);
      }
      return acc;
    }, []);
    expect(drawIndices).toHaveLength(2);
    expect(drawIndices[0]).toBeLessThan(matrixIndex);
    expect(drawIndices[1]).toBeGreaterThan(matrixIndex);

    const alphaSet = log.find(
      (entry) => entry.op === 'set' && entry.args[0] === 'globalAlpha' && entry.args[1] === 0.5
    );
    const blendSet = log.find(
      (entry) => entry.op === 'set' && entry.args[0] === 'globalCompositeOperation' && entry.args[1] === 'multiply'
    );
    expect(alphaSet).toBeDefined();
    expect(blendSet).toBeDefined();

    engine.lifecycle.dispose();
  });
});

describe('boolean raster operations', () => {
  const setup = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { projectId, store } = createReducerBackedStore(twoPaintDoc());
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (width: number, height: number): StubRasterSurface => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    };
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    return { engine, raf, surfaces };
  };

  it.each([
    ['intersect', 'source-in'],
    ['cutout', 'destination-in'],
    ['cutaway', 'source-out'],
    ['exclude', 'xor'],
  ] as const)('applies %s with %s, preserves its sources, and supports undo/redo', async (operation, composite) => {
    const { engine, raf, surfaces } = setup();
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.booleanMergeRasterLayers('upper', operation)).toBe('merged');

    const resultSurface = surfaces.find((surface) =>
      surface.callLog.some(
        (entry) => entry.op === 'set' && entry.args[0] === 'globalCompositeOperation' && entry.args[1] === composite
      )
    );
    expect(resultSurface).toBeDefined();
    expect(resultSurface!.callLog.filter((entry) => entry.op === 'drawImage')).toHaveLength(2);

    const merged = engine.document.getDocument()!;
    const result = getDocumentLeaves(merged).find((layer) => layer.id !== 'upper' && layer.id !== 'below');
    expect(result).toMatchObject({
      isEnabled: true,
      source: { bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(getDocumentLeaves(merged).find((layer) => layer.id === 'upper')?.isEnabled).toBe(false);
    expect(getDocumentLeaves(merged).find((layer) => layer.id === 'below')?.isEnabled).toBe(false);

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['upper', true],
      ['below', true],
    ]);

    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => [layer.id, layer.isEnabled])).toEqual([
      [result!.id, true],
      ['upper', false],
      ['below', false],
    ]);
    engine.lifecycle.dispose();
  });

  it('rejects unsupported and missing layer pairs without modifying the document', async () => {
    const doc = { ...twoPaintDoc(), stacks: stacksFrom([twoPaintDoc().stacks.raster[0]!, maskLayer('below')]) };
    const { projectId, store } = createReducerBackedStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.layers.booleanMergeRasterLayers('upper', 'exclude')).toBe('missing');
    expect(await engine.layers.booleanMergeRasterLayers('missing', 'exclude')).toBe('missing');
    expect(getDocumentLeaves(engine.document.getDocument()!)).toEqual(getDocumentLeaves(doc));
    engine.lifecycle.dispose();
  });
});

describe('async layer operations with a preview opened while they await', () => {
  const doc = (): CanvasDocumentContractV3 => {
    const base = twoPaintDoc();
    return {
      ...base,
      stacks: stacksFrom([
        maskLayer('other'),
        {
          ...maskLayer('mask'),
          mask: {
            bitmap: { height: 20, imageName: 'mask-bitmap', width: 20 },
            fill: { color: '#e07575', style: 'diagonal' as const },
            offset: { x: 15, y: 25 },
          },
        },
        ...getDocumentLeaves(base),
      ]),
    };
  };

  const setup = async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { projectId, store } = createReducerBackedStore(doc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const opacityOf = (id: string) =>
      getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === id)?.opacity;
    const hover = (id: string) => {
      const session = engine.layers.beginStructuralPreview()!;
      expect(session.apply({ id, patch: { opacity: 0.25 }, type: 'updateCanvasLayer' })).toBe(true);
      raf.flush();
      expect(opacityOf(id)).toBe(0.25);
      return session;
    };
    return { engine, hover, opacityOf };
  };

  it.each([
    ['copy to raster', (engine: CanvasEngine) => engine.layers.copyLayerToRaster('upper'), 'copied'],
    ['boolean merge', (engine: CanvasEngine) => engine.layers.booleanMergeRasterLayers('upper', 'intersect'), 'merged'],
    ['extract masked area', (engine: CanvasEngine) => engine.exports.extractMaskedArea('mask'), 'extracted'],
  ] as const)('lands a %s past a preview of another layer, ending it as the edit lands', async (_name, run, landed) => {
    const { engine, hover, opacityOf } = await setup();

    const operation = run(engine);
    const session = hover('other');
    const result = await operation;

    expect(typeof result === 'string' ? result : result.status).toBe(landed);
    expect(session.isActive()).toBe(false);
    expect(opacityOf('other')).toBe(1);
    await engine.history.undo();
    expect(opacityOf('other')).toBe(1);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('extract masked canvas area', () => {
  const maskedDoc = (): CanvasDocumentContractV3 => {
    const doc = twoPaintDoc();
    return {
      ...doc,
      stacks: stacksFrom([
        {
          ...maskLayer('mask'),
          mask: {
            bitmap: { height: 20, imageName: 'mask-bitmap', width: 20 },
            fill: { color: '#e07575', style: 'diagonal' as const },
            offset: { x: 15, y: 25 },
          },
        },
        ...getDocumentLeaves(doc),
      ]),
      selectedLayerId: 'mask',
    };
  };

  const setup = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { projectId, store } = createReducerBackedStore(maskedDoc());
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (width: number, height: number): StubRasterSurface => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    };
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    return { engine, raf, surfaces };
  };

  it('composites visible raster content through mask alpha and inserts one undoable raster layer', async () => {
    const { engine, raf, surfaces } = setup();
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const result = await engine.exports.extractMaskedArea('mask');
    expect(result.status).toBe('extracted');
    if (result.status !== 'extracted') {
      throw new Error('expected an extracted layer');
    }

    const extractedPixels = surfaces.find((surface) =>
      surface.callLog.some(
        (entry) =>
          entry.op === 'set' && entry.args[0] === 'globalCompositeOperation' && entry.args[1] === 'destination-in'
      )
    );
    expect(extractedPixels).toMatchObject({ height: 20, width: 20 });
    expect(extractedPixels!.callLog.filter((entry) => entry.op === 'drawImage').length).toBeGreaterThanOrEqual(3);

    const extracted = getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === result.layerId);
    expect(extracted).toMatchObject({
      isEnabled: true,
      source: { bitmap: null, offset: { x: 15, y: 25 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === 'mask')).toEqual(
      getDocumentLeaves(maskedDoc())[0]
    );
    expect(getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === 'upper')?.isEnabled).toBe(
      true
    );
    expect(getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === 'below')?.isEnabled).toBe(
      true
    );

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!).some((layer) => layer.id === result.layerId)).toBe(false);
    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!).some((layer) => layer.id === result.layerId)).toBe(true);
    engine.lifecycle.dispose();
  });

  it('does not extract stale contributor pixels after a source change during await', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const document = maskedDoc();
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    (store.dispatch as Mock).mockClear();

    const extraction = engine.exports.extractMaskedArea('mask');
    setDocument({
      ...document,
      stacks: stacksFrom(
        getDocumentLeaves(document).map((layer) =>
          layer.id === 'upper' && layer.type === 'raster'
            ? {
                ...layer,
                source: {
                  bitmap: { height: 40, imageName: 'upper-v2', width: 40 },
                  offset: { x: 0, y: 0 },
                  type: 'paint' as const,
                },
              }
            : layer
        )
      ),
    });

    expect(await extraction).toEqual({ status: 'not-ready' });
    expect(store.dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('rejects missing, unsupported, and empty masks without changing the document', async () => {
    const doc = maskedDoc();
    const { projectId, store } = createReducerBackedStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.exports.extractMaskedArea('missing')).toEqual({ status: 'missing' });
    expect(await engine.exports.extractMaskedArea('upper')).toEqual({ status: 'unsupported' });
    expect(await engine.exports.extractMaskedArea('mask')).toEqual({ status: 'not-ready' });
    expect(getDocumentLeaves(engine.document.getDocument()!)).toEqual(getDocumentLeaves(doc));
    engine.lifecycle.dispose();

    const emptyDoc = maskedDoc();
    emptyDoc.stacks.inpaint_mask[0] = maskLayer('mask');
    const emptyHarness = createReducerBackedStore(emptyDoc);
    const emptyEngine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: emptyHarness.projectId,
      store: emptyHarness.store,
    });
    expect(await emptyEngine.exports.extractMaskedArea('mask')).toEqual({ status: 'empty' });
    emptyEngine.lifecycle.dispose();
  });
});

const createReducerBackedStore = (
  document: CanvasDocumentContractV3,
  mainBase?: string
): { dispatch: Mock<(action: EngineTestAction) => void>; projectId: string; store: EngineStore } => {
  let state = createInitialWorkbenchState();
  const projectId = state.projects[0]!.id;
  state = workbenchReducer(state, {
    mutation: { document, type: 'replaceCanvasDocument' },
    projectId,
    type: 'applyCanvasProjectMutation',
  });
  if (mainBase) {
    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: { model: { base: mainBase, key: `${mainBase}-main`, name: `${mainBase} main`, type: 'main' } },
      widgetId: 'generate',
    });
  }
  const listeners = new Set<() => void>();
  const dispatch = vi.fn((action: EngineTestAction) => {
    state = workbenchReducer(
      state,
      isCanvasProjectMutation(action) ? { mutation: action, projectId, type: 'applyCanvasProjectMutation' } : action
    );
    for (const listener of listeners) {
      listener();
    }
  });
  return {
    dispatch,
    projectId,
    store: {
      dispatch,
      getState: () => state,
      reducesCanvasMutations: true,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
    },
  };
};

describe('staged result acceptance', () => {
  const stagedCandidate = (): CanvasStagingCandidateContract => ({
    height: 20,
    imageName: 'staged-result.png',
    imageUrl: '/staged-result.png',
    placement: { height: 40, opacity: 0.75, width: 30, x: 7, y: 9 },
    queuedAt: '2026-07-16T00:00:00.000Z',
    sourceQueueItemId: 'queue-staged',
    thumbnailUrl: '/staged-result-thumb.png',
    width: 15,
  });
  const stagedSelection = (candidate: CanvasStagingCandidateContract, selectedImageIndex = 0) => ({
    candidate,
    selectedImageIndex,
  });

  it('commits through the project-bound engine and preserves layer identity and staging semantics across undo/redo', async () => {
    const reducer = createReducerBackedStore({ ...makeDoc(), selectedLayerId: 'a' });
    const candidate = stagedCandidate();
    reducer.store.dispatch({
      candidate,
      projectId: reducer.projectId,
      type: 'appendCanvasStagingCandidate',
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });

    const result = engine.layers.commitStagedImage(stagedSelection(candidate));

    expect(result.status).toBe('committed');
    if (result.status !== 'committed') {
      throw new Error('expected commit');
    }
    const projectAfterCommit = reducer.store.getState().projects[0]!;
    const acceptedLayer = getDocumentLeaves(projectAfterCommit.canvas.document)[0]!;
    const acceptedEvent = projectAfterCommit.events[0]!;
    expect(acceptedLayer).toMatchObject({
      id: result.layerId,
      opacity: 0.75,
      source: { image: { imageName: 'staged-result.png' }, type: 'image' },
      transform: { scaleX: 2, scaleY: 2, x: 7, y: 9 },
    });
    expect(projectAfterCommit.canvas.stagingArea.pendingImages).toEqual([]);
    expect(acceptedEvent.type).toBe('canvas-layer-accepted');

    await engine.history.undo();
    const projectAfterUndo = reducer.store.getState().projects[0]!;
    expect(getDocumentLeaves(projectAfterUndo.canvas.document)).not.toContainEqual(
      expect.objectContaining({ id: result.layerId })
    );
    expect(projectAfterUndo.canvas.document.selectedLayerId).toBe('a');
    expect(projectAfterUndo.canvas.stagingArea.pendingImages).toEqual([]);

    await engine.history.redo();
    const projectAfterRedo = reducer.store.getState().projects[0]!;
    expect(getDocumentLeaves(projectAfterRedo.canvas.document)[0]).toBe(acceptedLayer);
    expect(projectAfterRedo.canvas.stagingArea.pendingImages).toEqual([]);
    expect(projectAfterRedo.events).toContain(acceptedEvent);
    expect(projectAfterRedo.events.filter((event) => event.id === acceptedEvent.id)).toHaveLength(1);

    await engine.history.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    reducer.store.dispatch({
      candidate: { ...candidate, imageName: 'new-staged-result.png' },
      projectId: reducer.projectId,
      type: 'appendCanvasStagingCandidate',
    });
    expect(
      engine.layers.commitStagedImage(stagedSelection({ ...candidate, imageName: 'new-staged-result.png' })).status
    ).toBe('committed');
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('structural raster publication failure atomicity', () => {
  const sentinelLayer = (): CanvasLayerContract => ({
    ...rasterLayer('selection-sentinel'),
    isEnabled: false,
  });

  const createFaultHarness = (document: CanvasDocumentContractV3) => {
    const faults = createStructuralFaultBackend();
    const bitmapStore = createSpyBitmapStore();
    const reducer = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: faults.backend,
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    return { ...reducer, bitmapStore, engine, faults };
  };

  type StructuralFault = 'allocation' | 'draw';
  const structuralFaults = ['allocation', 'draw'] as const satisfies readonly StructuralFault[];
  const armStructuralFault = (
    faults: ReturnType<typeof createStructuralFaultBackend>,
    failure: StructuralFault,
    countdowns: Record<StructuralFault, number>
  ): void => {
    if (failure === 'allocation') {
      faults.armAllocation(countdowns.allocation);
    } else {
      faults.armDraw(countdowns.draw);
    }
  };
  const structuralFaultMessage = (failure: StructuralFault): string => `structural cache ${failure} failed`;

  const expectInitialFailureExact = (
    harness: ReturnType<typeof createFaultHarness>,
    expectedDocument: CanvasDocumentContractV3
  ): void => {
    expect(harness.engine.document.getDocument()).toBe(expectedDocument);
    expect(harness.engine.document.getDocument()).toEqual(expectedDocument);
    expect(harness.engine.document.getDocument()!.selectedLayerId).toBe('selection-sentinel');
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(harness.bitmapStore.discardLayer).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(false);
  };

  it.each(structuralFaults)(
    'crop keeps document, cache, selection, and history exact when final cache %s fails',
    async (failure) => {
      const source = rasterLayer('crop-source');
      const document: CanvasDocumentContractV3 = {
        ...makeDoc(),
        bbox: { height: 7, width: 6, x: 2, y: 3 },
        stacks: stacksFrom([source, sentinelLayer()]),
        selectedLayerId: 'selection-sentinel',
      };
      const harness = createFaultHarness(document);
      const sourceCache = await snapshotLayerCache(harness.engine, source.id);
      const expectedDocument = harness.engine.document.getDocument()!;
      (harness.store.dispatch as Mock).mockClear();
      harness.bitmapStore.markLayerDirty.mockClear();
      armStructuralFault(harness.faults, failure, { allocation: 3, draw: 3 });

      await expect(harness.engine.layers.cropLayerToBbox(source.id)).resolves.toEqual({
        message: structuralFaultMessage(failure),
        status: 'failed',
      });

      expectInitialFailureExact(harness, expectedDocument);
      await expectLayerCacheExact(harness.engine, source.id, sourceCache);
      harness.engine.lifecycle.dispose();
    }
  );

  const createCropReplayHarness = async () => {
    const source = rasterLayer('crop-replay-source');
    const document: CanvasDocumentContractV3 = {
      ...makeDoc(),
      bbox: { height: 7, width: 6, x: 2, y: 3 },
      stacks: stacksFrom([source, sentinelLayer()]),
      selectedLayerId: 'selection-sentinel',
    };
    const harness = createFaultHarness(document);
    await snapshotLayerCache(harness.engine, source.id);
    await expect(harness.engine.layers.cropLayerToBbox(source.id)).resolves.toEqual({ status: 'cropped' });
    return { ...harness, originalDocument: document, source };
  };

  it.each(structuralFaults)(
    'crop undo cache-%s failure leaves the committed state exact and retryable',
    async (failure) => {
      const harness = await createCropReplayHarness();
      const committedDocument = harness.engine.document.getDocument()!;
      const committedCache = await snapshotLayerCache(harness.engine, harness.source.id);
      armStructuralFault(harness.faults, failure, { allocation: 0, draw: 0 });

      expect(await harness.engine.history.undo()).toBe('failed');

      expect(harness.engine.document.getDocument()).toBe(committedDocument);
      expect(harness.engine.document.getDocument()).toEqual(committedDocument);
      expect(harness.engine.stores.canUndo.get()).toBe(true);
      expect(harness.engine.stores.canRedo.get()).toBe(false);
      await expectLayerCacheExact(harness.engine, harness.source.id, committedCache);
      await harness.engine.history.undo();
      expect(harness.engine.document.getDocument()).toEqual(harness.originalDocument);
      harness.engine.lifecycle.dispose();
    }
  );

  it('publishes a prepared structural mutation to its bound project while another project is active', () => {
    const source = rasterLayer('inactive-source');
    const document = { ...makeDoc(), stacks: stacksFrom([source]), selectedLayerId: source.id };
    const harness = createFaultHarness(document);
    const liveSource = getDocumentLeaves(harness.engine.document.getDocument()!)[0]!;
    const copy = { ...structuredClone(source), id: 'inactive-copy' };
    harness.store.dispatch({ type: 'createProject' });
    const otherProjectId = harness.store.getState().activeProjectId;
    const otherBefore = structuredClone(
      harness.store.getState().projects.find((project) => project.id === otherProjectId)!.canvas.document
    );
    (harness.store.dispatch as Mock).mockClear();

    expect(
      harness.engine.layers.commitLayerCopy(
        'Copy layer',
        liveSource.id,
        copy,
        harness.engine.document.captureInsertionAnchor(copy.type, liveSource.id)
      )
    ).toBe(true);

    expect(harness.store.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'applyCanvasLayerStackMutation' })
    );
    expect(harness.store.getState().projects.find((project) => project.id === otherProjectId)!.canvas.document).toEqual(
      otherBefore
    );
    expect(getDocumentLeaves(harness.engine.document.getDocument())[0]?.id).toBe(copy.id);
    expect(harness.engine.stores.canUndo.get()).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('crop generation-cancels a deferred pre-crop upload before publishing the new paint incarnation', async () => {
    const source: CanvasRasterLayerContractV2 = {
      ...(rasterLayer('crop-persistence-source') as CanvasRasterLayerContractV2),
      source: {
        bitmap: { height: 10, imageName: 'before-crop-paint', width: 10 },
        offset: { x: 0, y: 0 },
        type: 'paint',
      },
    };
    const document: CanvasDocumentContractV3 = {
      ...makeDoc(),
      bbox: { height: 7, width: 6, x: 2, y: 3 },
      stacks: stacksFrom([source, sentinelLayer()]),
      selectedLayerId: 'selection-sentinel',
    };
    const reducer = createReducerBackedStore(document);
    const staleUpload = createDeferred<{ height: number; imageName: string; width: number }>();
    let uploadCount = 0;
    const uploadImage = vi.fn(() => {
      uploadCount += 1;
      return uploadCount === 1
        ? staleUpload.promise
        : Promise.resolve({ height: 7, imageName: 'cropped-paint', width: 6 });
    });
    let placed: { offset: { x: number; y: number }; surface: RasterSurface } | null = null;
    const bitmapStore = createBitmapStore({
      debounceMs: 1500,
      dispatch: createTestMutationPort(reducer.store, reducer.projectId).dispatch,
      encodeSurface: (surface) =>
        Promise.resolve(new Blob([`${surface.width}x${surface.height}`], { type: 'image/png' })),
      getLayerSource: (layerId) => {
        const layer = getDocumentLeaves(
          reducer.store.getState().projects.find((project) => project.id === reducer.projectId)?.canvas.document
        ).find((candidate) => candidate.id === layerId);
        return layer?.type === 'raster' || layer?.type === 'control' ? layer.source : null;
      },
      getLayerSurface: () => placed,
      hashBlob: (blob) => blob.text(),
      maxUploadAttempts: 1,
      retryDelaysMs: [],
      sleep: () => Promise.resolve(),
      uploadImage,
    });
    const discardLayer = vi.spyOn(bitmapStore, 'discardLayer');
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const before = await snapshotLayerCache(engine, source.id);
    placed = { offset: { x: before.rect.x, y: before.rect.y }, surface: before.surface };
    bitmapStore.markLayerDirty(source.id);
    const barrier = bitmapStore.flushPendingUploads();
    for (let tick = 0; tick < 50 && uploadImage.mock.calls.length === 0; tick += 1) {
      await Promise.resolve();
    }
    expect(uploadImage).toHaveBeenCalledOnce();

    await expect(engine.layers.cropLayerToBbox(source.id)).resolves.toEqual({ status: 'cropped' });
    const croppedCache = await snapshotLayerCache(engine, source.id);
    placed = {
      offset: { x: croppedCache.rect.x, y: croppedCache.rect.y },
      surface: croppedCache.surface,
    };
    staleUpload.resolve({ height: 10, imageName: 'stale-pre-crop-paint', width: 10 });
    await barrier;

    const bitmapUpdates = (reducer.store.dispatch as Mock).mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'updateCanvasLayerSource' && action.id === source.id);
    expect(discardLayer).toHaveBeenCalledWith(source.id);
    expect(bitmapUpdates).toHaveLength(1);
    expect(bitmapUpdates).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ source: { bitmap: expect.objectContaining({ imageName: 'stale-pre-crop-paint' }) } }),
      ])
    );
    expect(bitmapUpdates[0]).toMatchObject({
      source: {
        bitmap: { height: 7, imageName: 'cropped-paint', width: 6 },
        offset: { x: 2, y: 3 },
        type: 'paint',
      },
    });
    expect(engine.document.getDocument()!.selectedLayerId).toBe('selection-sentinel');
    expect(getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === source.id)).toMatchObject({
      source: {
        bitmap: { height: 7, imageName: 'cropped-paint', width: 6 },
        offset: { x: 2, y: 3 },
        type: 'paint',
      },
    });
    expect(engine.stores.canUndo.get()).toBe(true);
    await expectLayerCacheExact(engine, source.id, croppedCache);
    engine.lifecycle.dispose();
  });
});

describe('mergeVisibleRasterLayers', () => {
  const interleavedDoc = (): CanvasDocumentContractV3 => {
    const base = twoPaintDoc();
    const [upper, below] = getDocumentLeaves(base) as [CanvasLayerContract, CanvasLayerContract];
    return { ...base, stacks: stacksFrom([upper, maskLayer('mid-mask'), below]) };
  };

  const selectedMergeDoc = (): CanvasDocumentContractV3 => {
    const doc = interleavedDoc();
    return {
      ...doc,
      stacks: stacksFrom(
        getDocumentLeaves(doc).map((layer) =>
          layer.id === 'upper' ? { ...layer, blendMode: 'normal' as const } : layer
        )
      ),
    };
  };

  const setup = (
    doc: CanvasDocumentContractV3,
    pendingClearIds: ReadonlySet<string> = new Set(),
    pendingWorkIds: ReadonlySet<string> = new Set()
  ) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { projectId, store } = createReducerBackedStore(doc);
    const baseMutationPort = createTestMutationPort(store, projectId);
    let rejectMutations = false;
    const mutationPort: CanvasProjectMutationPort = {
      ...baseMutationPort,
      dispatch: (mutation, origin) => (rejectMutations ? false : baseMutationPort.dispatch(mutation, origin)),
    };
    const base = createPixelWritingBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (width, height) => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    };
    let panelSelectedIds = doc.selectedLayerId ? [doc.selectedLayerId] : [];
    const selectionChanges: { primaryId: string | null; selectedIds: readonly string[] }[] = [];
    const bitmapStore = createSpyBitmapStore();
    bitmapStore.hasPendingClear = (layerId) => pendingClearIds.has(layerId);
    bitmapStore.hasPendingWork = (layerId) => pendingWorkIds.has(layerId);
    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      getSelectedLayerIds: () => panelSelectedIds,
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort,
      projectId,
      setSelectedLayerIds: (primaryId, selectedIds) => {
        panelSelectedIds = [...selectedIds];
        selectionChanges.push({ primaryId, selectedIds: [...selectedIds] });
      },
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    return {
      bitmapStore,
      engine,
      getPanelSelectedIds: () => panelSelectedIds,
      raf,
      selectionChanges,
      setRejectMutations: (reject: boolean) => {
        rejectMutations = reject;
      },
      setPanelSelectedIds: (selectedIds: readonly string[]) => {
        panelSelectedIds = [...selectedIds];
      },
      store,
      surfaces,
    };
  };

  it('refuses a duplicate whose parent group was locked while its source pixels loaded', async () => {
    const base = twoPaintDoc();
    const [upper, below] = getDocumentLeaves(base) as [CanvasLayerContract, CanvasLayerContract];
    const { engine, store } = setup({ ...base, stacks: stacksFrom([groupContract('g', [upper]), below]) });

    const duplicate = engine.layers.duplicateLayers('Duplicate', ['upper']);
    store.dispatch({
      enabledUpdates: [],
      lockedUpdates: [{ id: 'g', isLocked: true }],
      type: 'applyCanvasLayerStackMutation',
    });
    const locked = engine.document.getDocument();

    expect(await duplicate).toEqual({ status: 'stale' });
    expect(engine.document.getDocument()).toBe(locked);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('duplicates a selected batch directly above each source with live caches and one undo step', async () => {
    const { engine, getPanelSelectedIds, raf, setPanelSelectedIds } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    setPanelSelectedIds(['upper', 'below']);

    const result = await engine.layers.duplicateLayers('Duplicate', ['upper', 'below']);
    expect(result.status).toBe('duplicated');
    if (result.status !== 'duplicated') {
      throw new Error(`unexpected duplicate result: ${result.status}`);
    }
    const [upperCopyId, belowCopyId] = result.duplicateIds;
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual([
      'mid-mask',
      upperCopyId,
      'upper',
      belowCopyId,
      'below',
    ]);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(upperCopyId);
    expect(getPanelSelectedIds()).toEqual([upperCopyId, belowCopyId]);
    expect(engine.exports.hasExportableLayerContent(upperCopyId!)).toBe(true);
    expect(engine.exports.hasExportableLayerContent(belowCopyId!)).toBe(true);

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual([
      'mid-mask',
      'upper',
      'below',
    ]);
    expect(engine.document.getDocument()!.selectedLayerId).toBe('upper');
    expect(getPanelSelectedIds()).toEqual(['upper', 'below']);

    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual([
      'mid-mask',
      upperCopyId,
      'upper',
      belowCopyId,
      'below',
    ]);
    expect(engine.exports.hasExportableLayerContent(upperCopyId!)).toBe(true);
    expect(engine.exports.hasExportableLayerContent(belowCopyId!)).toBe(true);
    expect(getPanelSelectedIds()).toEqual([upperCopyId, belowCopyId]);
    engine.lifecycle.dispose();
  });

  it('duplicates pending empty paint as empty instead of resurrecting its stale durable bitmap', async () => {
    const { engine } = setup(twoPaintDoc(), new Set(['upper']));

    const result = await engine.layers.duplicateLayers('Duplicate', ['upper']);

    expect(result.status).toBe('duplicated');
    if (result.status !== 'duplicated') {
      throw new Error(`unexpected duplicate result: ${result.status}`);
    }
    const duplicate = getDocumentLeaves(engine.document.getDocument()!).find(
      (layer) => layer.id === result.duplicateIds[0]
    );
    expect(duplicate).toMatchObject({ source: { bitmap: null, type: 'paint' }, type: 'raster' });
    engine.lifecycle.dispose();
  });

  it('merges the selected adjacent rasters, leaves other stacks alone, and restores them on undo', async () => {
    const { engine, getPanelSelectedIds, raf, setPanelSelectedIds } = setup(selectedMergeDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    setPanelSelectedIds(['upper', 'below']);
    const before = engine.document.getDocument()!;

    expect(await engine.layers.mergeSelectedRasterLayers(['upper', 'below'])).toBe('merged');
    const merged = engine.document.getDocument()!.stacks.raster[0]!;
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual(['mid-mask', merged.id]);
    expect(merged).toMatchObject({ name: 'upper merged', type: 'raster' });
    expect(engine.document.getDocument()!.selectedLayerId).toBe(merged.id);
    expect(getPanelSelectedIds()).toEqual([merged.id]);

    await engine.history.undo();
    const restored = engine.document.getDocument()!;
    expect(haveSameStructure(restored.stacks, before.stacks)).toBe(true);
    expect(
      getDocumentLeaves(restored).map((layer) =>
        getDocumentLeaves(before).find((candidate) => candidate.id === layer.id)
      )
    ).toEqual(getDocumentLeaves(restored));
    expect(restored.selectedLayerId).toBe(before.selectedLayerId);
    expect(engine.exports.hasExportableLayerContent('upper')).toBe(true);
    expect(engine.exports.hasExportableLayerContent('below')).toBe(true);
    expect(getPanelSelectedIds()).toEqual(['upper', 'below']);

    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual(['mid-mask', merged.id]);
    expect(engine.exports.hasExportableLayerContent(merged.id)).toBe(true);
    expect(getPanelSelectedIds()).toEqual([merged.id]);
    engine.lifecycle.dispose();
  });

  it('refuses to merge across an unselected raster because that would reorder the composite', async () => {
    const doc = selectedMergeDoc();
    const middle: CanvasRasterLayerContractV2 = {
      ...(getDocumentLeaves(doc).find((layer) => layer.id === 'below') as CanvasRasterLayerContractV2),
      id: 'middle-raster',
      name: 'middle-raster',
      source: { bitmap: { height: 50, imageName: 'middle-bmp', width: 50 }, type: 'paint' },
    };
    const upper = getDocumentLeaves(doc).find((layer) => layer.id === 'upper')!;
    const below = getDocumentLeaves(doc).find((layer) => layer.id === 'below')!;
    const withRasterGap = { ...doc, stacks: stacksFrom([upper, middle, below]) };
    const { engine, raf } = setup(withRasterGap);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.mergeSelectedRasterLayers(['upper', 'below'])).toBe('nothing');
    expect(getDocumentLeaves(engine.document.getDocument()!)).toEqual(getDocumentLeaves(withRasterGap));
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('refuses selected non-normal blend modes instead of changing their backdrop interaction', async () => {
    const doc = interleavedDoc();
    const { engine, raf } = setup(doc);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.mergeSelectedRasterLayers(['upper', 'below'])).toBe('nothing');
    expect(getDocumentLeaves(engine.document.getDocument()!)).toEqual(getDocumentLeaves(doc));
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('declines oversized duplicate and merge batches before unbounded transaction copies', async () => {
    const size = 4096;
    const layers = ['top', 'upper-middle', 'middle', 'lower-middle', 'bottom'].map((id): CanvasLayerContract => ({
      blendMode: 'normal',
      id,
      isEnabled: true,
      isLocked: false,
      name: id,
      opacity: 1,
      source: { bitmap: { height: size, imageName: `${id}-bmp`, width: size }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    }));
    const doc: CanvasDocumentContractV3 = {
      ...twoPaintDoc(),
      height: size,
      stacks: stacksFrom(layers),
      selectedLayerId: 'top',
      width: size,
    };
    const { engine, raf, surfaces } = setup(doc);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const afterRasterization = surfaces.length;

    expect(
      await engine.layers.duplicateLayers(
        'Duplicate',
        layers.map((layer) => layer.id)
      )
    ).toEqual({ status: 'over-budget' });
    expect(surfaces).toHaveLength(afterRasterization);
    expect(await engine.layers.mergeSelectedRasterLayers(layers.map((layer) => layer.id))).toBe('over-budget');
    expect(getDocumentLeaves(engine.document.getDocument()!)).toEqual(layers);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('creates a selected raster at index zero and preserves every source layer', async () => {
    const { engine, raf } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');

    const layers = getDocumentLeaves(engine.document.getDocument()!);
    const merged = engine.document.getDocument()!.stacks.raster[0]!;
    expect(layers).toHaveLength(4);
    expect(merged).toMatchObject({
      blendMode: 'normal',
      isEnabled: true,
      isLocked: false,
      name: 'upper merged',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(layers.filter((layer) => layer.id !== merged.id).map((layer) => layer.id)).toEqual([
      'mid-mask',
      'upper',
      'below',
    ]);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(merged.id);

    engine.lifecycle.dispose();
  });
});

const shapeLayerDoc = (over: { isLocked?: boolean } = {}): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([
    {
      blendMode: 'normal',
      id: 'shape1',
      isEnabled: true,
      isLocked: over.isLocked ?? false,
      name: 'Shape',
      opacity: 1,
      source: { fill: '#ff0000', height: 40, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 60 },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
      type: 'raster',
    },
  ]),
  selectedLayerId: 'shape1',
  version: 3,
  width: 100,
});

describe('rasterizeLayer (parametric → paint)', () => {
  const setup = (doc: CanvasDocumentContractV3) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { setDocument, store } = createReactiveStore(doc);
    const dispatch = store.dispatch as Mock;
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    return { backend, dispatch, engine, raf, setDocument, surfaces };
  };

  const convertCalls = (dispatch: Mock): EngineTestAction[] =>
    dispatch.mock.calls.map((call) => call[0] as EngineTestAction).filter((a) => a.type === 'convertCanvasLayer');

  it('bakes to a CONTENT-sized paint layer at identity and dispatches convertCanvasLayer with the offset', async () => {
    const { dispatch, engine, surfaces } = setup(shapeLayerDoc());
    const before = surfaces.length;

    expect(await engine.layers.rasterizeLayer('shape1')).toBe('rasterized');

    const converts = convertCalls(dispatch);
    expect(converts).toHaveLength(1);
    const convert = converts[0];
    expect(convert).toMatchObject({ id: 'shape1', targetType: 'raster', type: 'convertCanvasLayer' });
    if (convert?.type === 'convertCanvasLayer') {
      expect(convert.layer.type).toBe('raster');
      if (convert.layer.type === 'raster') {
        expect(convert.layer.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });
        expect(convert.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
      }
    }

    const baked = surfaces
      .slice(before)
      .find(
        (surface) => surface.width === 60 && surface.height === 40 && surface.callLog.some((e) => e.op === 'drawImage')
      );
    expect(baked).toBeDefined();

    engine.lifecycle.dispose();
  });

  it('undo re-converts to the ORIGINAL parametric source (no pixel snapshot)', async () => {
    const { dispatch, engine } = setup(shapeLayerDoc());
    await engine.layers.rasterizeLayer('shape1');

    await engine.history.undo();
    const converts = convertCalls(dispatch);
    expect(converts).toHaveLength(2);
    const undoConvert = converts[1];
    if (undoConvert?.type === 'convertCanvasLayer' && undoConvert.layer.type === 'raster') {
      expect(undoConvert.layer.source).toEqual({
        fill: '#ff0000',
        height: 40,
        kind: 'rect',
        stroke: null,
        strokeWidth: 0,
        type: 'shape',
        width: 60,
      });
      expect(undoConvert.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 });
    } else {
      throw new Error('expected undo to re-convert to the shape source');
    }

    await engine.history.redo();
    const afterRedo = convertCalls(dispatch);
    expect(afterRedo).toHaveLength(3);
    if (afterRedo[2]?.type === 'convertCanvasLayer' && afterRedo[2].layer.type === 'raster') {
      expect(afterRedo[2].layer.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });
    }
    engine.lifecycle.dispose();
  });

  it('is a no-op for a locked layer, a missing layer, and a non-parametric (paint) layer', async () => {
    const { dispatch, engine } = setup(shapeLayerDoc({ isLocked: true }));
    expect(await engine.layers.rasterizeLayer('shape1')).toBe('locked');
    expect(await engine.layers.rasterizeLayer('nope')).toBe('missing');
    expect(convertCalls(dispatch)).toHaveLength(0);
    engine.lifecycle.dispose();

    const paintDoc = shapeLayerDoc();
    paintDoc.stacks.raster[0] = {
      ...paintDoc.stacks.raster[0],
      source: { bitmap: null, type: 'paint' },
    } as CanvasLayerContract;
    const paint = setup(paintDoc);
    expect(await paint.engine.layers.rasterizeLayer('shape1')).toBe('unsupported');
    paint.engine.lifecycle.dispose();
  });

  describe('rasterize → undo → bitmap-store flush (source-type guard)', () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    const setupWithRealBitmapStore = (doc: CanvasDocumentContractV3) => {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
      const { setDocument, store } = createReactiveStore(doc);
      const dispatch = store.dispatch as Mock;

      const encodeSurface = vi.fn(() => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })));
      const uploadImage = vi.fn(() => Promise.resolve({ height: 100, imageName: 'img-x', width: 100 }));
      const fakeSurface = createTestStubRasterBackend().createSurface(10, 10);

      let engine: ReturnType<typeof createCanvasEngine>;
      const bitmapStore = createBitmapStore({
        dispatch: createTestMutationPort(store, 'p1').dispatch,
        encodeSurface,
        getLayerSource: (layerId) => {
          const layer = getDocumentLeaves(engine.document.getDocument()).find((candidate) => candidate.id === layerId);
          return layer && (layer.type === 'raster' || layer.type === 'control') ? layer.source : null;
        },
        getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface: fakeSurface }),
        hashBlob: (blob) => blob.text(),
        retryDelaysMs: [1],
        sleep: () => Promise.resolve(),
        uploadImage,
      });

      engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId: 'p1',
        store,
      });
      const screen = createFakeCanvas();
      const overlay = createFakeCanvas();
      engine.surface.attach(screen.element, overlay.element);
      raf.flush();

      return { dispatch, encodeSurface, engine, raf, setDocument, uploadImage };
    };

    const applyLastConvert = (doc: CanvasDocumentContractV3, dispatch: Mock): CanvasDocumentContractV3 => {
      const converts = convertCalls(dispatch);
      const last = converts.at(-1);
      if (last?.type !== 'convertCanvasLayer') {
        throw new Error('expected a convertCanvasLayer dispatch');
      }
      return {
        ...doc,
        stacks: stacksFrom(getDocumentLeaves(doc).map((layer) => (layer.id === last.id ? last.layer : layer))),
      };
    };

    const rasterizeThenUndo = async () => {
      vi.useFakeTimers();
      let doc = shapeLayerDoc();
      const harness = setupWithRealBitmapStore(doc);
      const { dispatch, engine, raf, setDocument } = harness;

      expect(await engine.layers.rasterizeLayer('shape1')).toBe('rasterized');
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await engine.history.undo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      return harness;
    };

    const paintSourceDispatches = (dispatch: Mock): Extract<EngineTestAction, { type: 'updateCanvasLayerSource' }>[] =>
      dispatch.mock.calls
        .map((call) => call[0] as EngineTestAction)
        .filter(
          (action): action is Extract<EngineTestAction, { type: 'updateCanvasLayerSource' }> =>
            action.type === 'updateCanvasLayerSource' && action.source.type === 'paint'
        );

    it('await flushPendingUploads(): the layer stays the parametric shape and no paint dispatch fires', async () => {
      const { dispatch, engine, uploadImage } = await rasterizeThenUndo();

      await engine.lifecycle.flushPendingUploads();

      expect(uploadImage).not.toHaveBeenCalled();
      expect(paintSourceDispatches(dispatch)).toHaveLength(0);
      const layer = getDocumentLeaves(engine.document.getDocument()!)[0] as CanvasRasterLayerContractV2;
      expect(layer.source.type).toBe('shape');

      engine.lifecycle.dispose();
    });

    it('redo after rasterize → flush → undo re-dispatches the paint image ref (fix round 2: no permanent bitmap:null)', async () => {
      vi.useFakeTimers();
      let doc = shapeLayerDoc();
      const { dispatch, engine, raf, setDocument, uploadImage } = setupWithRealBitmapStore(doc);

      expect(await engine.layers.rasterizeLayer('shape1')).toBe('rasterized');
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await vi.advanceTimersByTimeAsync(1500);
      await engine.lifecycle.flushPendingUploads();
      expect(uploadImage).toHaveBeenCalledTimes(1);

      const firstPersist = paintSourceDispatches(dispatch).at(-1);
      expect(firstPersist).toBeDefined();
      expect(firstPersist?.source).toMatchObject({ bitmap: { imageName: 'img-x' }, type: 'paint' });
      doc = {
        ...doc,
        stacks: stacksFrom(
          getDocumentLeaves(doc).map((layer) =>
            layer.id === firstPersist?.id ? { ...layer, source: firstPersist.source } : layer
          )
        ),
      };
      setDocument(doc);
      raf.flush();

      await engine.history.undo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await engine.history.redo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();
      const layerAfterRedo = getDocumentLeaves(engine.document.getDocument()!)[0] as CanvasRasterLayerContractV2;
      expect(layerAfterRedo.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });

      await vi.advanceTimersByTimeAsync(1500);
      await engine.lifecycle.flushPendingUploads();

      expect(uploadImage).toHaveBeenCalledTimes(1);
      const dispatchesAfterRedo = paintSourceDispatches(dispatch);
      const secondPersist = dispatchesAfterRedo.at(-1);
      expect(secondPersist).toBeDefined();
      expect(secondPersist).not.toBe(firstPersist);
      expect(secondPersist?.source).toMatchObject({ bitmap: { imageName: 'img-x' }, type: 'paint' });

      doc = {
        ...doc,
        stacks: stacksFrom(
          getDocumentLeaves(doc).map((layer) =>
            layer.id === secondPersist?.id ? { ...layer, source: secondPersist.source } : layer
          )
        ),
      };
      setDocument(doc);
      const layerFinal = getDocumentLeaves(engine.document.getDocument()!)[0] as CanvasRasterLayerContractV2;
      expect(layerFinal.source.type).toBe('paint');
      if (layerFinal.source.type === 'paint') {
        expect(layerFinal.source.bitmap).not.toBeNull();
        expect(layerFinal.source.bitmap?.imageName).toBe('img-x');
      }

      engine.lifecycle.dispose();
    });
  });
});

const selectedImageDoc = (
  overrides: { isLocked?: boolean; isEnabled?: boolean } = {},
  selectedLayerId: string | null = 'a'
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([{ ...rasterLayer('a'), ...overrides }]),
  selectedLayerId,
  version: 3,
  width: 100,
});

describe('nudgeSelectedLayer', () => {
  const setup = (doc: CanvasDocumentContractV3, selectedLayerIds?: readonly string[]) => {
    const { store } = createFakeStore(doc);
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      getSelectedLayerIds: () => selectedLayerIds ?? (doc.selectedLayerId ? [doc.selectedLayerId] : []),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    return { dispatch, engine };
  };

  it('no-ops on a hidden selected layer', () => {
    const { dispatch, engine } = setup(selectedImageDoc({ isEnabled: false }));
    engine.layers.nudgeSelectedLayer(0, 1);
    expect(dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('dispatches a transform update and records an undoable entry', async () => {
    const { dispatch, engine } = setup(selectedImageDoc());
    engine.layers.nudgeSelectedLayer(3, -2);
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenNthCalledWith(1, {
      type: 'setCanvasLayerPositions',
      updates: [{ id: 'a', x: 3, y: -2 }],
    });
    expect(engine.stores.canUndo.get()).toBe(true);

    await engine.history.undo();
    expect(dispatch).toHaveBeenNthCalledWith(2, {
      type: 'setCanvasLayerPositions',
      updates: [{ id: 'a', x: 0, y: 0 }],
    });
    engine.lifecycle.dispose();
  });
});

describe('flipLayer', () => {
  it('commits one undoable flip, and inside the layer’s session flips only the preview', async () => {
    const { store } = createReducerBackedStore(selectedImageDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: store.getState().projects[0]!.id,
      store,
    });
    const transform = () => getDocumentLayer(engine.document.getDocument(), 'a')!.transform;

    expect(engine.layers.flipLayer('a', 'horizontal', 'Flip horizontal')).toEqual({ status: 'committed' });
    expect(transform()).toMatchObject({ scaleX: -1, scaleY: 1, x: 10, y: 0 });
    await engine.history.undo();
    expect(transform()).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });

    engine.tools.setTool('transform');
    expect(engine.layers.flipLayer('a', 'vertical', 'Flip vertical')).toEqual({ status: 'previewed' });
    expect(engine.stores.transformSession.get()?.transform).toMatchObject({ scaleY: -1, y: 10 });
    expect(transform()).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    engine.lifecycle.dispose();
  });
});

describe('move tool: drag through the pipeline', () => {
  it('commits one structural transform update and records an undoable entry', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(paintDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('move');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 30));
    overlay.fire('pointerup', pointerAt(50, 30, { buttons: 0 }));

    const transformUpdates = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'updateCanvasLayer');
    expect(transformUpdates).toHaveLength(1);
    expect(transformUpdates[0]).toMatchObject({ id: 'paint1', type: 'updateCanvasLayer' });
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.lifecycle.dispose();
  });
});

const imageSelectedDoc = (): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([rasterLayer('img')]),
  selectedLayerId: 'img',
  version: 3,
  width: 100,
});

const textSelectedDoc = (): CanvasDocumentContractV3 => ({
  ...imageSelectedDoc(),
  stacks: stacksFrom([textRasterLayer('img')]),
});

describe('engine-owned history: composed auto-create + stroke entry', () => {
  it('undo removes the auto-created layer (no pixel restore); redo re-adds it and re-applies the stroke', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(textSelectedDoc());
    const dispatch = store.dispatch as Mock;
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const engine = createCanvasEngine({
      backend,
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const strokes: StrokeCommittedEvent[] = [];
    engine.tools.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    expect(strokes).toHaveLength(1);
    const created = strokes[0]!.createdLayer;
    expect(created).toBeDefined();
    const newLayerId = created!.layer.id;
    expect(engine.stores.canUndo.get()).toBe(true);

    const addCalls = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'addCanvasLayer');
    expect(addCalls).toHaveLength(1);

    const putBefore = () => putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.beforeImageData);
    const beforeUndoPutBefore = putBefore();

    await engine.history.undo();
    const removeAfterUndo = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'removeCanvasLayers');
    expect(removeAfterUndo).toHaveLength(1);
    expect(removeAfterUndo[0]).toEqual({ ids: [newLayerId], type: 'removeCanvasLayers' });
    expect(putBefore()).toBe(beforeUndoPutBefore);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    await engine.history.redo();
    const addAfterRedo = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'addCanvasLayer');
    expect(addAfterRedo).toHaveLength(2);
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.afterImageData)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.lifecycle.dispose();
  });
});

describe('setStagedPreview', () => {
  const emptyDoc = (bbox = { height: 100, width: 100, x: 0, y: 0 }): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox,
    height: 100,
    stacks: stacksFrom([]),
    selectedLayerId: null,
    version: 3,
    width: 100,
  });

  const createDeferredBitmapBackend = () => {
    const base = createTestStubRasterBackend();
    const deferreds: ReturnType<typeof createDeferred<ImageBitmap>>[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createImageBitmap: () => {
        const deferred = createDeferred<ImageBitmap>();
        deferreds.push(deferred);
        return deferred.promise;
      },
    };
    return {
      backend,
      deferreds,
      resolveBitmap: (index: number, width = 0, height = 0): void =>
        deferreds[index]!.resolve({ close: () => {}, height, width } as unknown as ImageBitmap),
    };
  };

  const stagedDraws = (surface: StubRasterSurface): unknown[][] =>
    surface.callLog.filter((entry) => entry.op === 'drawImage' && entry.args.length === 5).map((entry) => entry.args);

  const dataUrl = (tag: string) => `data:image/png;base64,${tag}`;
  const at = (x: number, y: number, width: number, height: number) => ({ height, opacity: 1, width, x, y });

  it('draws the newest decode and discards a stale one that resolves later (version race)', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const bitmaps = createDeferredBitmapBackend();
    const { store } = createReactiveStore(emptyDoc());
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    screen.surface.callLog.length = 0;

    engine.previews.setStagedPreview({ dataUrl: dataUrl('AAAA'), placement: at(0, 0, 10, 10) });
    engine.previews.setStagedPreview({ dataUrl: dataUrl('BBBB'), placement: at(0, 0, 20, 20) });
    bitmaps.resolveBitmap(1);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();

    const draws = stagedDraws(screen.surface);
    expect(draws.length).toBeGreaterThan(0);
    for (const args of draws) {
      expect(args.slice(1)).toEqual([0, 0, 20, 20]);
    }

    engine.lifecycle.dispose();
  });
});

describe('commitRasterFilterResult', () => {
  const durableImage = { height: 10, imageName: 'filtered-result', width: 10 };

  const createReplayFaultBackend = () => {
    const base = createTestStubRasterBackend();
    let fault: 'allocation' | 'draw' | null = null;
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (width, height) => {
        if (fault === 'allocation') {
          fault = null;
          throw new Error('replay cache allocation failed');
        }
        const surface = base.createSurface(width, height);
        const ctx = new Proxy(surface.ctx, {
          get: (target, property, receiver) => {
            const value = Reflect.get(target, property, receiver);
            if (property !== 'drawImage' || typeof value !== 'function') {
              return value;
            }
            return (...args: unknown[]) => {
              if (fault === 'draw') {
                fault = null;
                throw new Error('replay cache draw failed');
              }
              return Reflect.apply(value, target, args);
            };
          },
        });
        Object.defineProperty(surface, 'ctx', { value: ctx });
        return surface;
      },
    };
    return {
      arm: (next: 'allocation' | 'draw') => {
        fault = next;
      },
      backend,
    };
  };

  const filterLayer = (id = 'L'): CanvasRasterLayerContractV2 => ({
    adjustments: [
      { brightness: 0.2, contrast: -0.1, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      { id: 'adj-hsl', isEnabled: true, saturation: 0.3, type: 'hsl' as const },
    ],
    blendMode: 'multiply',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 0.6,
    source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
    transform: { rotation: 15, scaleX: 2, scaleY: 3, x: 5, y: 6 },
    type: 'raster',
  });

  const filterDoc = (layers: CanvasLayerContract[] = [filterLayer()]): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom(layers),
    selectedLayerId: layers[0]?.id ?? null,
    version: 3,
    width: 100,
  });

  it('replaces with local paint pixels, clears adjustments, and restores exact state with one history entry', async () => {
    const layer = filterLayer();
    const { projectId, store } = createReducerBackedStore(filterDoc([layer]));
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(recordingBitmap('filtered')));
    const engine = createCanvasEngine({
      backend,
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const before = structuredClone(getDocumentLeaves(engine.document.getDocument()!)[0]!);
    const exported = await engine.exports.exportLayerPixels(layer.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable export');
    }
    const result = await engine.layers.commitRasterFilterResult({
      guard: exported.guard,
      image: durableImage,
      mode: 'replace',
      rect: { height: 10, width: 10, x: -4, y: 7 },
    });

    expect(result).toEqual({ layerId: layer.id, status: 'committed' });
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    const after = getDocumentLeaves(engine.document.getDocument()!)[0]!;
    const { adjustments: _adjustments, ...beforeWithoutAdjustments } = before as CanvasRasterLayerContractV2;
    expect(after).toEqual({
      ...beforeWithoutAdjustments,
      source: { bitmap: durableImage, offset: { x: -4, y: 7 }, type: 'paint' },
    });
    expect(after.transform).toEqual(layer.transform);
    expect('adjustments' in after).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(true);
    const forwardExport = await engine.exports.exportLayerPixels(layer.id);
    expect(forwardExport.status).toBe('ok');
    if (forwardExport.status !== 'ok') {
      throw new Error('expected committed pixels');
    }
    const forwardSources = backend.drawSourcesFor(forwardExport.surface as StubRasterSurface);

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    const undoExport = await engine.exports.exportLayerPixels(layer.id);
    expect(undoExport.status).toBe('ok');
    if (undoExport.status === 'ok') {
      const undoSources = backend.drawSourcesFor(undoExport.surface as StubRasterSurface);
      const beforeSnapshot = backend.surfaceById(undoSources.at(-1)!);
      expect(beforeSnapshot).toBeDefined();
      expect(backend.drawSourcesFor(beforeSnapshot!)).toContain(
        backend.surfaceId(exported.surface as StubRasterSurface)
      );
    }

    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(after);
    expect(engine.stores.canRedo.get()).toBe(false);
    const redoExport = await engine.exports.exportLayerPixels(layer.id);
    expect(redoExport.status).toBe('ok');
    if (redoExport.status === 'ok') {
      expect(backend.drawSourcesFor(redoExport.surface as StubRasterSurface)).toEqual(forwardSources);
    }
    engine.lifecycle.dispose();
  });

  it('copies filtered local pixels directly above the source and replays one structural history entry', async () => {
    const source = filterLayer('source');
    const below = filterLayer('below');
    const { projectId, store } = createReducerBackedStore({
      ...filterDoc([source, below]),
      selectedLayerId: below.id,
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const sourceBefore = structuredClone(source);
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable export');
    }

    const result = await engine.layers.commitRasterFilterResult({
      guard: exported.guard,
      image: durableImage,
      mode: 'copy',
      rect: { height: 10, width: 10, x: 3, y: 4 },
    });

    expect(result.status).toBe('committed');
    if (result.status !== 'committed') {
      throw new Error('expected a committed copy');
    }
    expect(getDocumentLeaves(engine.document.getDocument()!).map((candidate) => candidate.id)).toEqual([
      result.layerId,
      'source',
      'below',
    ]);
    expect(getDocumentLeaves(engine.document.getDocument()!).find((candidate) => candidate.id === 'source')).toEqual(
      sourceBefore
    );
    const copy = getDocumentLeaves(engine.document.getDocument()!)[0]!;
    expect(copy).toMatchObject({
      blendMode: source.blendMode,
      isEnabled: source.isEnabled,
      isLocked: false,
      opacity: source.opacity,
      source: { bitmap: durableImage, offset: { x: 3, y: 4 }, type: 'paint' },
      transform: source.transform,
      type: 'raster',
    });
    expect('adjustments' in copy).toBe(false);
    expect((await engine.exports.exportLayerPixels(result.layerId)).status).toBe('ok');

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((candidate) => candidate.id)).toEqual([
      'source',
      'below',
    ]);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(below.id);
    expect(engine.stores.canUndo.get()).toBe(false);
    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!).map((candidate) => candidate.id)).toEqual([
      result.layerId,
      'source',
      'below',
    ]);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(copy);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it.each(['allocation', 'draw'] as const)(
    'keeps replace undo retryable and exact when detached cache %s fails',
    async (failure) => {
      const source = filterLayer('source');
      const { projectId, store } = createReducerBackedStore(filterDoc([source]));
      const faults = createReplayFaultBackend();
      const bitmapStore = createSpyBitmapStore();
      const engine = createCanvasEngine({
        backend: faults.backend,
        bitmapStore,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels(source.id);
      if (exported.status !== 'ok') {
        throw new Error('expected a filterable export');
      }
      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode: 'replace',
        rect: exported.rect,
      });
      expect(result.status).toBe('committed');
      const expectedDocument = structuredClone(engine.document.getDocument()!);
      const expectedCache = await engine.exports.exportLayerPixels(source.id);
      if (expectedCache.status !== 'ok') {
        throw new Error('expected committed cache pixels');
      }
      const expectedCalls = structuredClone((expectedCache.surface as StubRasterSurface).callLog);
      const adjustedDeletesBefore = adjustedSurfaceCacheDeletes.length;
      const thumbnailListener = vi.fn();
      const unsubscribeThumbnail = engine.stores.thumbnailVersion.subscribe(thumbnailListener);
      bitmapStore.markLayerDirty.mockClear();
      (store.dispatch as Mock).mockClear();
      faults.arm(failure);

      await expectReplayFailure(engine.history.undo(), `replay cache ${failure} failed`);

      expect(engine.document.getDocument()).toEqual(expectedDocument);
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(engine.stores.canRedo.get()).toBe(false);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(thumbnailListener).not.toHaveBeenCalled();
      expect(adjustedSurfaceCacheDeletes).toHaveLength(adjustedDeletesBefore);
      const afterFailure = await engine.exports.exportLayerPixels(source.id);
      expect(afterFailure.status).toBe('ok');
      if (afterFailure.status === 'ok') {
        expect(afterFailure.surface).toBe(expectedCache.surface);
        expect(afterFailure.rect).toEqual(expectedCache.rect);
        expect(afterFailure.guard.cacheVersion).toBe(expectedCache.guard.cacheVersion);
        expect((afterFailure.surface as StubRasterSurface).callLog).toEqual(expectedCalls);
      }

      await engine.history.undo();
      expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(source);
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(true);
      unsubscribeThumbnail();
      engine.lifecycle.dispose();
    }
  );

  it('discards obsolete persistence and keeps the exact durable image ref on replace redo', async () => {
    const source = filterLayer('source');
    const { projectId, store } = createReducerBackedStore(filterDoc([source]));
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable export');
    }

    await expect(
      engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode: 'replace',
        rect: exported.rect,
      })
    ).resolves.toEqual({ layerId: source.id, status: 'committed' });
    expect(bitmapStore.discardLayer).toHaveBeenCalledTimes(1);

    await engine.history.undo();
    bitmapStore.markLayerDirty.mockClear();
    await engine.history.redo();

    expect(bitmapStore.discardLayer).toHaveBeenCalledTimes(2);
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    const redone = getDocumentLeaves(engine.document.getDocument()!)[0];
    expect(redone).toMatchObject({ source: { bitmap: durableImage, type: 'paint' } });
    if (redone?.type === 'raster' && redone.source.type === 'paint') {
      expect(redone.source.bitmap).toEqual(durableImage);
    }
    engine.lifecycle.dispose();
  });

  describe('paint persistence cancellation', () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    const createRealBitmapStoreHarness = async (
      uploadImage: (blob: Blob) => Promise<{ height: number; imageName: string; width: number }>
    ) => {
      vi.stubGlobal(
        'Path2D',
        class FakePath2D {
          closePath() {}
          lineTo() {}
          moveTo() {}
          quadraticCurveTo() {}
        }
      );
      const source: CanvasRasterLayerContractV2 = {
        ...filterLayer('source'),
        source: {
          bitmap: { height: 10, imageName: 'before-paint', width: 10 },
          offset: { x: 0, y: 0 },
          type: 'paint',
        },
      };
      const { projectId, store } = createReducerBackedStore(filterDoc([source]));
      const backend = {
        ...createPixelWritingBackend(undefined, createRecordingRasterBackend()),
        encodeSurface: vi.fn(() => Promise.resolve(new Blob(['paint-pixels'], { type: 'image/png' }))),
      };
      let bitmapCall = 0;
      backend.createImageBitmap = vi.fn(() =>
        Promise.resolve(recordingBitmap(bitmapCall++ === 0 ? 'before-paint' : 'filtered-result'))
      );
      let placed: { offset: { x: number; y: number }; surface: RasterSurface } | null = null;
      const bitmapStore = createBitmapStore({
        debounceMs: 1500,
        dispatch: createTestMutationPort(store, projectId).dispatch,
        encodeSurface: () => Promise.resolve(new Blob(['unpersisted-paint'], { type: 'image/png' })),
        getLayerSource: (layerId) => {
          const layer = getDocumentLeaves(
            store.getState().projects.find((project) => project.id === projectId)?.canvas.document
          ).find((candidate) => candidate.id === layerId);
          return layer?.type === 'raster' || layer?.type === 'control' ? layer.source : null;
        },
        getLayerSurface: () => placed,
        hashBlob: (blob) => blob.text(),
        maxUploadAttempts: 1,
        retryDelaysMs: [],
        sleep: () => Promise.resolve(),
        uploadImage,
      });
      const discardLayer = vi.spyOn(bitmapStore, 'discardLayer');
      const engine = createCanvasEngine({
        backend,
        bitmapStore,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const initial = await engine.exports.exportLayerPixels(source.id);
      if (initial.status !== 'ok') {
        throw new Error('expected persisted paint pixels');
      }
      engine.selection.selectAll();
      engine.selection.fillSelection();
      const refreshPlacement = async (layerId = source.id) => {
        const exported = await engine.exports.exportLayerPixels(layerId);
        if (exported.status !== 'ok') {
          throw new Error('expected live paint pixels');
        }
        placed = {
          offset: { x: exported.rect.x, y: exported.rect.y },
          surface: exported.surface,
        };
        return exported;
      };
      const exported = await refreshPlacement();
      return { backend, bitmapStore, discardLayer, engine, exported, projectId, refreshPlacement, source, store };
    };

    const bitmapUpdateActions = (store: ReturnType<typeof createReducerBackedStore>['store']) =>
      (store.dispatch as Mock).mock.calls
        .map((call) => call[0] as EngineTestAction)
        .filter((action) => action.type === 'updateCanvasLayerSource' && action.id === 'source');

    it('cancels pending debounced live-paint persistence before durable replace publication', async () => {
      vi.useFakeTimers();
      const uploadImage = vi.fn(() => Promise.resolve({ height: 10, imageName: 'stale-pending-paint', width: 10 }));
      const harness = await createRealBitmapStoreHarness(uploadImage);

      await expect(
        harness.engine.layers.commitRasterFilterResult({
          guard: harness.exported.guard,
          image: durableImage,
          mode: 'replace',
          rect: harness.exported.rect,
        })
      ).resolves.toEqual({ layerId: harness.source.id, status: 'committed' });
      const committed = structuredClone(harness.engine.document.getDocument()!);

      await vi.advanceTimersByTimeAsync(1500);
      await harness.bitmapStore.flushPendingUploads();

      expect(harness.discardLayer).toHaveBeenCalledWith(harness.source.id);
      expect(uploadImage).not.toHaveBeenCalled();
      expect(bitmapUpdateActions(harness.store)).toHaveLength(0);
      expect(harness.engine.document.getDocument()).toEqual(committed);
      expect(getDocumentLeaves(harness.engine.document.getDocument()!)[0]).toMatchObject({
        source: { bitmap: durableImage, type: 'paint' },
      });
      const cache = await harness.engine.exports.exportLayerPixels(harness.source.id);
      expect(cache.status).toBe('ok');
      if (cache.status === 'ok') {
        expect(drawGraphContains(cache.surface as StubRasterSurface, harness.backend, 'bitmap-filtered-result')).toBe(
          true
        );
      }

      const reloadDocument = structuredClone(harness.engine.document.getDocument()!);
      harness.engine.lifecycle.dispose();
      const reloadStore = createReducerBackedStore(reloadDocument);
      const reloadResolver = vi.fn(() => Promise.resolve(new Blob()));
      const reloadEngine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore: createSpyBitmapStore(),
        imageResolver: reloadResolver,
        projectId: reloadStore.projectId,
        store: reloadStore.store,
      });
      expect((await reloadEngine.exports.exportLayerPixels(harness.source.id)).status).toBe('ok');
      expect(reloadResolver).toHaveBeenCalledWith(durableImage.imageName, expect.any(AbortSignal));
      reloadEngine.lifecycle.dispose();
    });
  });

  const createGuardHarness = async (
    imageResolver: (imageName: string, signal?: AbortSignal) => Promise<Blob> = () => Promise.resolve(new Blob())
  ) => {
    const layer = filterLayer();
    const document = filterDoc([layer]);
    const reactive = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver,
      projectId: 'p1',
      store: reactive.store,
    });
    const exported = await engine.exports.exportLayerPixels(layer.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable export');
    }
    (reactive.store.dispatch as Mock).mockClear();
    const commit = (signal?: AbortSignal) =>
      engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode: 'replace',
        rect: exported.rect,
        signal,
      });
    return { ...reactive, commit, document, engine, guard: exported.guard, layer };
  };

  const createPendingGuardHarness = async () => {
    const decoded = createDeferred<Blob>();
    const harness = await createGuardHarness(() => decoded.promise);
    return { ...harness, decoded, pending: harness.commit() };
  };

  it('returns failed without mutation when the durable image cannot be decoded', async () => {
    const harness = await createGuardHarness(() => Promise.reject(new Error('decode failed')));

    await expect(harness.commit()).resolves.toEqual({ message: 'decode failed', status: 'failed' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('returns aborted without mutation when cancellation occurs during durable-image decode', async () => {
    const decoded = createDeferred<Blob>();
    const harness = await createGuardHarness(() => decoded.promise);
    const controller = new AbortController();

    const pending = harness.commit(controller.signal);
    controller.abort();
    decoded.resolve(new Blob());

    await expect(pending).resolves.toEqual({ status: 'aborted' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('does not publish when an operation starts during durable-image decode', async () => {
    const harness = await createPendingGuardHarness();
    const cleanupPreview = vi.fn();
    getCanvasOperations(harness.engine).controller.start({
      cleanupPreview,
      guard: harness.guard,
      identity: { kind: 'filter', layerId: harness.layer.id, projectId: 'p1' },
    });

    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'busy' });
    expect(harness.engine.document.getDocument()).toEqual(harness.document);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(cleanupPreview).not.toHaveBeenCalled();
    expect(getCanvasOperations(harness.engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter' },
      status: 'active',
    });
    harness.engine.lifecycle.dispose();
  });

  type PendingGuardHarness = Awaited<ReturnType<typeof createPendingGuardHarness>>;
  it.each<[string, string, (harness: PendingGuardHarness) => Promise<void> | void]>([
    ['missing', 'the source is deleted', (h) => h.setDocument({ ...h.document, stacks: stacksFrom([]) })],
    [
      'stale',
      'the source contract changes',
      (h) => h.setDocument({ ...h.document, stacks: stacksFrom([{ ...h.layer, opacity: 0.2 }]) }),
    ],
    [
      'locked',
      'the source becomes locked',
      (h) => h.setDocument({ ...h.document, stacks: stacksFrom([{ ...h.layer, isLocked: true }]) }),
    ],
    [
      'stale',
      'the guarded source changes layer type',
      (h) => {
        const { adjustments: _adjustments, ...base } = h.layer;
        const control: CanvasLayerContract = {
          ...base,
          adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
          type: 'control',
          withTransparencyEffect: false,
        };
        h.setDocument({ ...h.document, stacks: stacksFrom([control]) });
      },
    ],
    ['stale', 'the source cache version changes', (h) => h.engine.diagnostics.clearCaches()],
    [
      'stale',
      'document replacement reuses the source layer',
      (h) => h.setDocument({ ...h.document, height: 200, width: 200 }, 1),
    ],
  ])('returns %s without mutation when %s', async (status, _when, change) => {
    const harness = await createPendingGuardHarness();
    await change(harness);
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it.each(['replace', 'copy'] as const)(
    'commits %s when a document observer throws after the reducer applies the mutation',
    async (mode) => {
      const source = filterLayer('source');
      const { projectId, store } = createReducerBackedStore(filterDoc([source]));
      const bitmapStore = createSpyBitmapStore();
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels(source.id);
      if (exported.status !== 'ok') {
        throw new Error('expected a filterable export');
      }
      const unsubscribeFault = store.subscribe(() => {
        throw new Error('document observer failed');
      });

      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode,
        rect: exported.rect,
      });

      expect(result).toMatchObject({ status: 'committed' });
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      if (result.status === 'committed') {
        expect((await engine.exports.exportLayerPixels(result.layerId)).status).toBe('ok');
      }
      unsubscribeFault();
      engine.lifecycle.dispose();
    }
  );
});

describe('commitGeneratedImageResult', () => {
  const generatedImage = { height: 18, imageName: 'workflow-result.png', width: 24 };
  const generatedOrigin = { x: -9, y: 14 };

  const workflowRaster = (id = 'source'): CanvasRasterLayerContractV2 => ({
    adjustments: [
      { brightness: 0.2, contrast: -0.1, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      { id: 'adj-hsl', isEnabled: true, saturation: 0.3, type: 'hsl' as const },
    ],
    blendMode: 'multiply',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 0.65,
    source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
    transform: { rotation: 0.4, scaleX: 2, scaleY: 3, x: 5, y: 6 },
    type: 'raster',
  });

  const workflowControl = (id = 'source'): CanvasLayerContract => ({
    adapter: {
      beginEndStepPct: [0.15, 0.85],
      controlMode: 'more_control',
      kind: 'controlnet',
      model: 'control-model',
      weight: 0.7,
    },
    blendMode: 'screen',
    filter: { settings: { low: 12, high: 90 }, type: 'canny' },
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 0.55,
    source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
    transform: { rotation: -0.3, scaleX: 1.5, scaleY: 0.75, x: 11, y: -4 },
    type: 'control',
    withTransparencyEffect: true,
  });

  const workflowDocument = (layers: CanvasLayerContract[] = [workflowRaster()]): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom(layers),
    selectedLayerId: layers.at(-1)?.id ?? null,
    version: 3,
    width: 100,
  });

  const createGeneratedFaultBackend = () => {
    const base = createTestStubRasterBackend();
    let allocationCountdown: number | null = null;
    let drawCountdown: number | null = null;
    let createHook: { countdown: number; run: () => void } | null = null;
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (width, height) => {
        if (allocationCountdown !== null) {
          if (allocationCountdown === 0) {
            allocationCountdown = null;
            throw new Error('generated cache allocation failed');
          }
          allocationCountdown -= 1;
        }
        if (createHook) {
          if (createHook.countdown === 0) {
            const hook = createHook;
            createHook = null;
            hook.run();
          } else {
            createHook.countdown -= 1;
          }
        }
        const surface = base.createSurface(width, height);
        const ctx = new Proxy(surface.ctx, {
          get: (target, property, receiver) => {
            const value = Reflect.get(target, property, receiver);
            if (property !== 'drawImage' || typeof value !== 'function') {
              return value;
            }
            return (...args: unknown[]) => {
              if (drawCountdown !== null) {
                if (drawCountdown === 0) {
                  drawCountdown = null;
                  throw new Error('generated cache draw failed');
                }
                drawCountdown -= 1;
              }
              return Reflect.apply(value, target, args);
            };
          },
        });
        Object.defineProperty(surface, 'ctx', { value: ctx });
        return surface;
      },
    };
    return {
      armAllocation: (successfulCreates: number) => {
        allocationCountdown = successfulCreates;
      },
      armCreateHook: (successfulCreates: number, run: () => void) => {
        createHook = { countdown: successfulCreates, run };
      },
      armDraw: (successfulDraws: number) => {
        drawCountdown = successfulDraws;
      },
      backend,
    };
  };

  it('replaces a raster at the generated origin and native size, clears baked adjustments, and replays exactly', async () => {
    const source = workflowRaster();
    const before = structuredClone(source);
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(recordingBitmap('workflow-result')));
    const bitmapStore = createSpyBitmapStore();
    const { projectId, store } = createReducerBackedStore(workflowDocument([source]));
    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }

    const result = await engine.layers.commitGeneratedImageResult({
      guard: exported.guard,
      image: generatedImage,
      origin: generatedOrigin,
      target: 'replace',
    });

    expect(result).toEqual({ layerId: source.id, status: 'committed' });
    const after = getDocumentLeaves(engine.document.getDocument()!)[0]!;
    expect(after).toEqual({
      blendMode: source.blendMode,
      id: source.id,
      isEnabled: source.isEnabled,
      isLocked: source.isLocked,
      name: source.name,
      opacity: source.opacity,
      source: { bitmap: generatedImage, offset: generatedOrigin, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect('adjustments' in after).toBe(false);
    const committedPixels = await engine.exports.exportLayerPixels(source.id);
    expect(committedPixels.status).toBe('ok');
    if (committedPixels.status === 'ok') {
      expect(committedPixels.rect).toEqual({ ...generatedOrigin, height: 18, width: 24 });
      expect(drawGraphContains(committedPixels.surface as StubRasterSurface, backend, 'bitmap-workflow-result')).toBe(
        true
      );
    }
    expect(engine.stores.canUndo.get()).toBe(true);

    await engine.history.undo();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    await engine.history.redo();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(after);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    expect(bitmapStore.discardLayer).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('copies a generated result to a raster at the top of the raster stack and restores selection on undo', async () => {
    const above = workflowRaster('above');
    const source = workflowControl('source');
    const selected = workflowRaster('selected');
    const document = { ...workflowDocument([above, source, selected]), selectedLayerId: selected.id };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }

    const result = await engine.layers.commitGeneratedImageResult({
      guard: exported.guard,
      image: generatedImage,
      origin: generatedOrigin,
      target: 'copy-raster',
    });
    if (result.status !== 'committed') {
      throw new Error(`expected a committed copy, received ${JSON.stringify(result)}`);
    }

    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual([
      source.id,
      result.layerId,
      above.id,
      selected.id,
    ]);
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(source);
    expect(engine.document.getDocument()!.stacks.raster[0]).toMatchObject({
      id: result.layerId,
      isEnabled: true,
      isLocked: false,
      opacity: 1,
      source: { bitmap: generatedImage, offset: generatedOrigin, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);

    await engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    await engine.history.redo();
    expect(engine.document.getDocument()!.stacks.raster[0]?.id).toBe(result.layerId);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  const createPendingGuardHarness = async () => {
    const source = workflowRaster();
    const document = workflowDocument([source]);
    const decoded = createDeferred<Blob>();
    const reactive = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: (imageName) =>
        imageName === generatedImage.imageName ? decoded.promise : Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    (reactive.store.dispatch as Mock).mockClear();
    const pending = engine.layers.commitGeneratedImageResult({
      guard: exported.guard,
      image: generatedImage,
      origin: generatedOrigin,
      target: 'replace',
    });
    return { ...reactive, decoded, document, engine, exported, pending, source };
  };

  it('returns failed without mutation when the durable generated image cannot be resolved', async () => {
    const source = workflowRaster();
    const document = workflowDocument([source]);
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: (imageName) =>
        imageName === generatedImage.imageName
          ? Promise.reject(new Error('generated decode failed'))
          : Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    (store.dispatch as Mock).mockClear();

    await expect(
      engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        target: 'replace',
      })
    ).resolves.toEqual({ message: 'generated decode failed', status: 'failed' });
    expect(engine.document.getDocument()).toEqual(document);
    expect(store.dispatch).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  type PendingGuardHarness = Awaited<ReturnType<typeof createPendingGuardHarness>>;
  it.each<[string, string, (harness: PendingGuardHarness) => void]>([
    ['missing', 'the source was deleted', (h) => h.setDocument({ ...h.document, stacks: stacksFrom([]) })],
    [
      'locked',
      'the source became locked',
      (h) => h.setDocument({ ...h.document, stacks: stacksFrom([{ ...h.source, isLocked: true }]) }),
    ],
    [
      'unsupported',
      'the source became a mask',
      (h) => h.setDocument({ ...h.document, stacks: stacksFrom([maskLayer(h.source.id)]) }),
    ],
    [
      'stale',
      'the immutable source contract changed',
      (h) => h.setDocument({ ...h.document, stacks: stacksFrom([{ ...h.source, opacity: 0.1 }]) }),
    ],
    [
      'stale',
      'the document was replaced with the same source object',
      (h) => h.setDocument({ ...h.document, width: 200 }, 1),
    ],
  ])('returns %s after decode when %s', async (status, _when, change) => {
    const harness = await createPendingGuardHarness();
    change(harness);
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns aborted during bitmap decode, releases the late bitmap, and never mutates', async () => {
    const source = workflowRaster();
    const bitmapDeferred = createDeferred<ImageBitmap>();
    const bitmap = recordingBitmap('late-generated');
    const base = createTestStubRasterBackend();
    const createImageBitmap = vi.fn(() => bitmapDeferred.promise);
    const { projectId, store } = createReducerBackedStore(workflowDocument([source]));
    const engine = createCanvasEngine({
      backend: { ...base, createImageBitmap },
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    const controller = new AbortController();
    const pending = engine.layers.commitGeneratedImageResult({
      guard: exported.guard,
      image: generatedImage,
      origin: generatedOrigin,
      signal: controller.signal,
      target: 'replace',
    });
    await vi.waitFor(() => expect(createImageBitmap).toHaveBeenCalledOnce());

    controller.abort();
    bitmapDeferred.resolve(bitmap);

    await expect(pending).resolves.toEqual({ status: 'aborted' });
    expect(bitmap.close).toHaveBeenCalledOnce();
    expect(engine.document.getDocument()).toEqual(workflowDocument([source]));
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('keeps replace undo and redo retryable when detached cache preparation fails', async () => {
    const source = workflowRaster();
    const document = workflowDocument([source]);
    const faults = createGeneratedFaultBackend();
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: faults.backend,
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    await expect(
      engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        target: 'replace',
      })
    ).resolves.toEqual({ layerId: source.id, status: 'committed' });
    const committed = structuredClone(engine.document.getDocument()!);

    faults.armAllocation(0);
    await expectReplayFailure(engine.history.undo(), 'generated cache allocation failed');
    expect(engine.document.getDocument()).toEqual(committed);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    await engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    faults.armDraw(0);
    await expectReplayFailure(engine.history.redo(), 'generated cache draw failed');
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    await engine.history.redo();
    expect(engine.document.getDocument()).toEqual(committed);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('reloads a replaced generated image from its durable bitmap contract', async () => {
    const source = workflowRaster();
    const { projectId, store } = createReducerBackedStore(workflowDocument([source]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    await expect(
      engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        target: 'replace',
      })
    ).resolves.toEqual({ layerId: source.id, status: 'committed' });
    const reloadedDocument = structuredClone(engine.document.getDocument()!);
    engine.lifecycle.dispose();

    const reloadedStore = createReducerBackedStore(reloadedDocument);
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const reloadedEngine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: resolver,
      projectId: reloadedStore.projectId,
      store: reloadedStore.store,
    });
    expect((await reloadedEngine.exports.exportLayerPixels(source.id)).status).toBe('ok');
    expect(resolver).toHaveBeenCalledWith(generatedImage.imageName, expect.any(AbortSignal));
    reloadedEngine.lifecycle.dispose();
  });
});

describe('replaceSelectionFromImage', () => {
  const resultImage = { height: 10, imageName: 'sam-result.png', width: 10 };
  const resultRect = { height: 10, width: 10, x: -7, y: 13 };

  const sourceLayer = (): CanvasRasterLayerContractV2 => ({
    blendMode: 'normal',
    id: 'sam-source',
    isEnabled: true,
    isLocked: false,
    name: 'SAM source',
    opacity: 1,
    source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -7, y: 13 },
    type: 'raster',
  });

  const sourceDocument = (layer: CanvasLayerContract = sourceLayer()): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom([layer]),
    selectedLayerId: layer.id,
    version: 3,
    width: 100,
  });

  const alphaBackend = (alpha: number): StubRasterBackend => {
    const base = createTestStubRasterBackend();
    return {
      ...base,
      createSurface: (width, height) => {
        const surface = base.createSurface(width, height);
        Object.defineProperty(surface.ctx, 'getImageData', {
          value: (_x: number, _y: number, readWidth: number, readHeight: number) => {
            const data = new Uint8ClampedArray(readWidth * readHeight * 4);
            for (let index = 3; index < data.length; index += 4) {
              data[index] = alpha;
            }
            return { colorSpace: 'srgb', data, height: readHeight, width: readWidth } as ImageData;
          },
        });
        return surface;
      },
    };
  };

  const createHarness = async ({
    alpha = 0,
    backend = alphaBackend(alpha),
    imageResolver = () => Promise.resolve(new Blob()),
    seedSelection = true,
  }: {
    alpha?: number;
    backend?: StubRasterBackend;
    imageResolver?: (imageName: string, signal?: AbortSignal) => Promise<Blob>;
    seedSelection?: boolean;
  } = {}) => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const layer = sourceLayer();
    const document = sourceDocument(layer);
    const reactive = createReactiveStore(document);
    const engine = createCanvasEngine({ backend, imageResolver, projectId: 'p1', store: reactive.store });
    const exported = await engine.exports.exportLayerPixels(layer.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an export guard');
    }
    if (seedSelection) {
      engine.selection.selectAll();
    }
    (reactive.store.dispatch as Mock).mockClear();
    return { ...reactive, document, engine, guard: exported.guard, layer };
  };

  const createPendingHarness = async () => {
    const decoded = createDeferred<Blob>();
    const harness = await createHarness({ imageResolver: () => decoded.promise });
    const pending = harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect);
    return { ...harness, decoded, pending };
  };

  it('decodes and places a non-empty alpha result into the transient selection', async () => {
    const imageResolver = vi.fn((_imageName: string, _signal?: AbortSignal) => Promise.resolve(new Blob()));
    const harness = await createHarness({ alpha: 255, imageResolver, seedSelection: false });

    await expect(
      harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect)
    ).resolves.toEqual({
      status: 'selected',
    });
    expect(imageResolver).toHaveBeenCalledWith(resultImage.imageName, undefined);
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns aborted and closes the bitmap when cancellation lands during bitmap decode', async () => {
    const bitmap = createDeferred<ImageBitmap>();
    const close = vi.fn();
    const backend = { ...alphaBackend(0), createImageBitmap: vi.fn(() => bitmap.promise) };
    const harness = await createHarness({ backend });
    const controller = new AbortController();

    const pending = harness.engine.selection.replaceSelectionFromImage(
      harness.guard,
      resultImage,
      resultRect,
      controller.signal
    );
    await vi.waitFor(() => expect(backend.createImageBitmap).toHaveBeenCalledOnce());
    controller.abort();
    bitmap.resolve({ close, height: 10, width: 10 } as unknown as ImageBitmap);

    await expect(pending).resolves.toEqual({ status: 'aborted' });
    expect(close).toHaveBeenCalledOnce();
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('returns failed on decode failure without touching the existing selection', async () => {
    const harness = await createHarness({ imageResolver: () => Promise.reject(new Error('SAM decode failed')) });

    await expect(
      harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect)
    ).resolves.toEqual({
      message: 'SAM decode failed',
      status: 'failed',
    });
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it.each([
    {
      expected: 'missing',
      label: 'source deletion',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) =>
        harness.setDocument({ ...harness.document, stacks: stacksFrom([]) }),
    },
    {
      expected: 'stale',
      label: 'source contract edit',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) =>
        harness.setDocument({ ...harness.document, stacks: stacksFrom([{ ...harness.layer, opacity: 0.5 }]) }),
    },
    {
      expected: 'locked',
      label: 'source lock',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) =>
        harness.setDocument({ ...harness.document, stacks: stacksFrom([{ ...harness.layer, isLocked: true }]) }),
    },
    {
      expected: 'unsupported',
      label: 'source type change',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) => {
        const { source: _source, ...base } = harness.layer;
        const mask: CanvasInpaintMaskLayerContract = {
          ...base,
          mask: { bitmap: null, fill: { color: '#fff', style: 'solid' } },
          type: 'inpaint_mask',
        };
        harness.setDocument({ ...harness.document, stacks: stacksFrom([mask]) });
      },
    },
    {
      expected: 'stale',
      label: 'document replacement reusing the layer',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) => harness.setDocument({ ...harness.document }, 1),
    },
  ] as const)('refuses $label without mutating the selection', async ({ expected, mutate }) => {
    const harness = await createPendingHarness();
    mutate(harness);
    const selectionAfterExternalChange = harness.engine.stores.hasSelection.get();
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: expected });
    expect(harness.engine.stores.hasSelection.get()).toBe(selectionAfterExternalChange);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });
});

describe('commitMaskImageResult', () => {
  beforeEach(resetMaskFillColors);

  const resultImage = { height: 12, imageName: 'durable-sam-result.png', width: 16 };
  const resultRect = { height: 12, width: 16, x: -5, y: 7 };

  const source = (): CanvasRasterLayerContractV2 => ({
    blendMode: 'normal',
    id: 'source',
    isEnabled: true,
    isLocked: false,
    name: 'Source',
    opacity: 1,
    source: { fill: '#fff', height: 12, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 16 },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -5, y: 7 },
    type: 'raster',
  });

  const existingInpaint = (): CanvasInpaintMaskLayerContract => ({
    blendMode: 'normal',
    id: 'existing-mask',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
    name: 'Inpaint Mask 1',
    opacity: 1,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'inpaint_mask',
  });

  const existingRegional = (): CanvasLayerContract => ({
    autoNegative: false,
    blendMode: 'normal',
    id: 'existing-region',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: null, fill: { color: '#83d683', style: 'solid' } },
    name: 'Regional Guidance 1',
    negativePrompt: null,
    opacity: 0.5,
    positivePrompt: null,
    referenceImages: [],
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'regional_guidance',
  });

  const docFor = (target: 'inpaint_mask' | 'regional_guidance'): CanvasDocumentContractV3 => {
    const layer = source();
    const below = { ...source(), id: 'below', name: 'Below' };
    return {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      stacks: stacksFrom([target === 'inpaint_mask' ? existingInpaint() : existingRegional(), layer, below]),
      selectedLayerId: below.id,
      version: 3,
      width: 100,
    };
  };

  it.each([
    {
      expected: {
        blendMode: 'normal',
        isEnabled: true,
        isLocked: false,
        mask: {
          bitmap: resultImage,
          fill: { color: '#83d683', style: 'diagonal' },
          offset: { x: -5, y: 7 },
        },
        name: 'Inpaint Mask 2',
        opacity: 1,
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'inpaint_mask',
      },
      target: 'inpaint_mask',
    },
    {
      expected: {
        autoNegative: false,
        blendMode: 'normal',
        isEnabled: true,
        isLocked: false,
        mask: {
          bitmap: resultImage,
          fill: { color: '#f59e42', style: 'solid' },
          offset: { x: -5, y: 7 },
        },
        name: 'Regional Guidance 2',
        negativePrompt: null,
        opacity: 0.5,
        positivePrompt: null,
        referenceImages: [],
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'regional_guidance',
      },
      target: 'regional_guidance',
    },
  ] as const)(
    'adds a complete $target at the top of its stack with one exact history entry',
    async ({ expected, target }) => {
      const document = docFor(target);
      const { projectId, store } = createReducerBackedStore(document);
      const bitmapStore = createSpyBitmapStore();
      const imageResolver = vi.fn(() => Promise.resolve(new Blob()));
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore,
        imageResolver,
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels('source');
      if (exported.status !== 'ok') {
        throw new Error('expected an export guard');
      }

      const result = await engine.layers.commitMaskImageResult({
        guard: exported.guard,
        image: resultImage,
        rect: resultRect,
        target,
      });

      expect(result.status).toBe('committed');
      if (result.status !== 'committed') {
        throw new Error('expected a committed mask');
      }
      const created = getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === result.layerId)!;
      expect(created).toEqual({ ...expected, id: result.layerId });
      expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual([
        result.layerId,
        getDocumentLeaves(document)[0]!.id,
        'source',
        'below',
      ]);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
      expect(imageResolver).not.toHaveBeenCalled();
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(true);

      await engine.history.undo();
      expect(engine.document.getDocument()).toEqual(document);
      expect(engine.document.getDocument()!.selectedLayerId).toBe('below');
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(true);
      expect(engine.history.getHeldAssetRefs().images).toContain(resultImage.imageName);

      await engine.history.redo();
      expect(getDocumentLeaves(engine.document.getDocument()!).find((layer) => layer.id === result.layerId)).toEqual(
        created
      );
      expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(engine.stores.canRedo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it('keeps mask undo failure-atomic when its removal fails before reducer application', async () => {
    const document = docFor('inpaint_mask');
    const { dispatch, projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels('source');
    if (exported.status !== 'ok') {
      throw new Error('expected an export guard');
    }
    const result = await engine.layers.commitMaskImageResult({
      guard: exported.guard,
      image: resultImage,
      rect: resultRect,
      target: 'inpaint_mask',
    });
    if (result.status !== 'committed') {
      throw new Error('expected a committed mask');
    }
    const committedDocument = structuredClone(engine.document.getDocument()!);
    const reducerDispatch = dispatch.getMockImplementation();
    if (!reducerDispatch) {
      throw new Error('expected reducer-backed dispatch');
    }
    let failRemoval = true;
    dispatch.mockImplementation((action: EngineTestAction) => {
      if (failRemoval && action.type === 'applyCanvasLayerStackMutation') {
        failRemoval = false;
        throw new Error('mask removal failed');
      }
      reducerDispatch(action);
    });

    expect(await engine.history.undo()).toBe('failed');

    expect(engine.document.getDocument()).toEqual(committedDocument);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    await engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  const guardHarness = async () => {
    const layer = source();
    const document: CanvasDocumentContractV3 = { ...docFor('inpaint_mask'), stacks: stacksFrom([layer]) };
    const reactive = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });
    const exported = await engine.exports.exportLayerPixels(layer.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an export guard');
    }
    (reactive.store.dispatch as Mock).mockClear();
    return { ...reactive, document, engine, guard: exported.guard, layer };
  };

  it.each([
    {
      expected: 'missing',
      label: 'source deletion',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) =>
        harness.setDocument({ ...harness.document, stacks: stacksFrom([]) }),
    },
    {
      expected: 'stale',
      label: 'source contract edit',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) =>
        harness.setDocument({ ...harness.document, stacks: stacksFrom([{ ...harness.layer, opacity: 0.4 }]) }),
    },
    {
      expected: 'locked',
      label: 'source lock',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) =>
        harness.setDocument({ ...harness.document, stacks: stacksFrom([{ ...harness.layer, isLocked: true }]) }),
    },
    {
      expected: 'unsupported',
      label: 'source type change',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) => {
        const { source: _source, ...base } = harness.layer;
        const mask: CanvasInpaintMaskLayerContract = {
          ...base,
          mask: { bitmap: null, fill: { color: '#fff', style: 'solid' } },
          type: 'inpaint_mask',
        };
        harness.setDocument({ ...harness.document, stacks: stacksFrom([mask]) });
      },
    },
    {
      expected: 'stale',
      label: 'document replacement',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) => harness.setDocument({ ...harness.document }, 1),
    },
    {
      expected: 'stale',
      label: 'cache invalidation',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) => harness.engine.diagnostics.clearCaches(),
    },
  ] as const)('refuses $label without adding a mask or history', async ({ expected, mutate }) => {
    const harness = await guardHarness();
    await mutate(harness);

    await expect(
      harness.engine.layers.commitMaskImageResult({
        guard: harness.guard,
        image: resultImage,
        rect: resultRect,
        target: 'regional_guidance',
      })
    ).resolves.toEqual({ status: expected });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });
});

describe('guarded filter previews', () => {
  const emptyDoc = (): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom([]),
    selectedLayerId: null,
    version: 3,
    width: 100,
  });

  const previewableLayer = (id: string): CanvasLayerContract => ({
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  });

  const guardableLayer = (id: string): CanvasRasterLayerContractV2 => ({
    ...(previewableLayer(id) as CanvasRasterLayerContractV2),
    source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
  });

  const filterBitmapBackend = (width = 10, height = 10): StubRasterBackend => ({
    ...createTestStubRasterBackend(),
    createImageBitmap: () => Promise.resolve({ close: () => undefined, height, width } as unknown as ImageBitmap),
  });

  it('auto-processes a debounced preview after a filter draft update', async () => {
    const layer = guardableLayer('auto-filter');
    const { store } = createReactiveStore({ ...emptyDoc(), stacks: stacksFrom([layer]) });
    const runGraph = vi.fn(() =>
      Promise.resolve({ height: 10, imageName: 'auto.png', origin: 'test', output: {}, width: 10 })
    );
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      filterDeps: { runGraph, uploadIntermediate: () => Promise.resolve({ imageName: 'input.png' }) },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels(layer.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation(layer.id)).toBe('started');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      autoProcess: true,
      status: 'ready',
    });

    getCanvasOperations(engine).updateFilterOperation({
      settings: { high_threshold: 210, low_threshold: 90 },
      type: 'canny_edge_detection',
    });
    expect(runGraph).not.toHaveBeenCalled();
    await vi.waitFor(
      () =>
        expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
          preview: { imageName: 'auto.png' },
          status: 'ready',
        }),
      { timeout: 3000 }
    );
    expect(runGraph).toHaveBeenCalledOnce();
    engine.lifecycle.dispose();
  });

  const createDeferredBitmapBackend = () => {
    const base = createTestStubRasterBackend();
    const deferreds: ReturnType<typeof createDeferred<ImageBitmap>>[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createImageBitmap: () => {
        const deferred = createDeferred<ImageBitmap>();
        deferreds.push(deferred);
        return deferred.promise;
      },
    };
    return {
      backend,
      resolveBitmap: (index: number): void =>
        deferreds[index]!.resolve({ close: () => {}, height: 0, width: 0 } as unknown as ImageBitmap),
    };
  };

  it('publishes a guarded filter preview while the exported layer snapshot remains current', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), stacks: stacksFrom([guardableLayer('L')]) };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const exported = await engine.exports.exportLayerPixels('L');
    if (exported.status !== 'ok') {
      throw new Error('expected a guardable export');
    }

    const pending = engine.previews.setGuardedFilterPreview(
      'L',
      { imageName: 'filtered', rect: exported.rect },
      exported.guard
    );
    await flushMicrotasks();
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('shown');
    engine.lifecycle.dispose();
  });

  const filterFlowLayer = (): Extract<CanvasLayerContract, { type: 'raster' }> => ({
    blendMode: 'multiply',
    filter: { settings: { radius: 2 }, type: 'content_shuffle' },
    id: 'filter-source',
    isEnabled: true,
    isLocked: false,
    name: 'Filter source',
    opacity: 0.6,
    source: {
      bitmap: { height: 10, imageName: 'filter-source.png', width: 10 },
      offset: { x: 7, y: -3 },
      type: 'paint',
    },
    transform: { rotation: 12, scaleX: 2, scaleY: 3, x: 40, y: 50 },
    type: 'raster',
  });

  const createFilterFlowHarness = async () => {
    const source = filterFlowLayer();
    const document = { ...emptyDoc(), stacks: stacksFrom([source]), selectedLayerId: source.id };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      bitmapStore: createSpyBitmapStore(),
      filterDeps: {
        runGraph: vi.fn(() =>
          Promise.resolve({ height: 10, imageName: 'filtered.png', origin: 'canvas', output: {}, width: 10 })
        ),
        uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'filter-input.png' })),
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels(source.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');
    await getCanvasOperations(engine).processFilterOperation();
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      preview: { imageName: 'filtered.png' },
      status: 'ready',
    });
    return { document, engine, source };
  };

  it('processes, retries durability, and applies with one origin-preserving history entry', async () => {
    const { document, engine, source } = await createFilterFlowHarness();
    const makeDurable = vi
      .fn<(imageName: string) => Promise<void>>()
      .mockRejectedValueOnce(new Error('promotion failed'))
      .mockResolvedValueOnce();

    await getCanvasOperations(engine).commitFilterOperation('apply', makeDurable);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      error: { code: 'apply-failed', detail: 'promotion failed' },
      preview: { imageName: 'filtered.png' },
      status: 'error',
    });
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter' },
      status: 'active',
    });

    await getCanvasOperations(engine).commitFilterOperation('apply', makeDurable);
    expect(makeDurable).toHaveBeenCalledTimes(2);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toMatchObject({
      filter: source.filter,
      opacity: source.opacity,
      blendMode: source.blendMode,
      source: { offset: { x: 7, y: -3 }, type: 'paint' },
      transform: source.transform,
      type: 'raster',
    });
    expect(engine.stores.canUndo.get()).toBe(true);
    await engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('cancels an in-flight promotion when a replacement operation starts', async () => {
    const { engine, source } = await createFilterFlowHarness();
    let resolvePromotion!: () => void;
    const promotion = new Promise<void>((resolve) => {
      resolvePromotion = resolve;
    });
    const pending = getCanvasOperations(engine).commitFilterOperation('apply', () => promotion);
    await vi.waitFor(() => expect(getCanvasOperations(engine).stores.filterSession.get()?.status).toBe('committing'));

    expect(getCanvasOperations(engine).startSelectObject(source.id)).toBe('started');
    resolvePromotion();
    await pending;

    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toEqual(source);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object' },
      status: 'active',
    });
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('aborts an in-flight upload when a newer Process request wins', async () => {
    const source = filterFlowLayer();
    const document = { ...emptyDoc(), stacks: stacksFrom([source]), selectedLayerId: source.id };
    const { projectId, store } = createReducerBackedStore(document);
    const uploadSignals: AbortSignal[] = [];
    let uploadCount = 0;
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      bitmapStore: createSpyBitmapStore(),
      filterDeps: {
        runGraph: () =>
          Promise.resolve({ height: 10, imageName: 'newest-filter.png', origin: 'canvas', output: {}, width: 10 }),
        uploadIntermediate: (_blob, signal) => {
          if (!signal) {
            throw new Error('expected upload cancellation signal');
          }
          uploadSignals.push(signal);
          uploadCount += 1;
          if (uploadCount === 2) {
            return Promise.resolve({ imageName: 'newest-input.png' });
          }
          return new Promise((_resolve, reject) => {
            signal.addEventListener('abort', () => reject(new DOMException('superseded', 'AbortError')), {
              once: true,
            });
          });
        },
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels(source.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');

    const older = getCanvasOperations(engine).processFilterOperation();
    await vi.waitFor(() => expect(uploadSignals).toHaveLength(1));
    const newer = getCanvasOperations(engine).processFilterOperation();
    await Promise.all([older, newer]);

    expect(uploadSignals[0]?.aborted).toBe(true);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      error: null,
      preview: { imageName: 'newest-filter.png' },
      status: 'ready',
    });
    engine.lifecycle.dispose();
  });

  it('makes filter and Select Object operations mutually exclusive', async () => {
    const document = { ...emptyDoc(), stacks: stacksFrom([guardableLayer('L')]) };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('L')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('L')).toBe('started');

    expect(getCanvasOperations(engine).startSelectObject('L')).toBe('started');

    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).stores.samSession.get()).not.toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object' },
      status: 'active',
    });
    engine.lifecycle.dispose();
  });

  it('refuses ordinary structural editing while a filter operation is active', async () => {
    const document = { ...emptyDoc(), stacks: stacksFrom([guardableLayer('L')]) };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('L')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('L')).toBe('started');
    vi.mocked(store.dispatch).mockClear();

    engine.layers.commitStructural(
      'Rename',
      { id: 'L', patch: { name: 'Changed' }, type: 'updateCanvasLayer' },
      { id: 'L', patch: { name: 'L' }, type: 'updateCanvasLayer' }
    );

    expect(store.dispatch).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.tools.handleEscapePriority({ gestureWasActive: false });
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('invalidates the owned filter session when its guarded source changes', async () => {
    const layer = guardableLayer('L');
    const document = { ...emptyDoc(), stacks: stacksFrom([layer]) };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('L')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('L')).toBe('started');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      draft: { type: 'canny_edge_detection' },
      initialFilter: null,
    });

    setDocument({ ...document, stacks: stacksFrom([{ ...layer, opacity: 0.5 }]) });

    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('rejects a guarded filter preview when its source changes during decode', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const layer = guardableLayer('L');
    const document = { ...emptyDoc(), stacks: stacksFrom([layer]) };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const exported = await engine.exports.exportLayerPixels('L');
    if (exported.status !== 'ok') {
      throw new Error('expected a guardable export');
    }

    const pending = engine.previews.setGuardedFilterPreview(
      'L',
      { imageName: 'filtered', rect: exported.rect },
      exported.guard
    );
    await flushMicrotasks();
    setDocument({
      ...document,
      stacks: stacksFrom([{ ...layer, source: { ...layer.source, fill: '#000' } } as CanvasLayerContract]),
    });
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('stale');
    engine.lifecycle.dispose();
  });

  it('lets only the newest guarded filter preview publish', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), stacks: stacksFrom([guardableLayer('L')]) };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const exported = await engine.exports.exportLayerPixels('L');
    if (exported.status !== 'ok') {
      throw new Error('expected a guardable export');
    }

    const older = engine.previews.setGuardedFilterPreview(
      'L',
      { imageName: 'older', rect: exported.rect },
      exported.guard
    );
    const newer = engine.previews.setGuardedFilterPreview(
      'L',
      { imageName: 'newer', rect: exported.rect },
      exported.guard
    );
    await flushMicrotasks();
    bitmaps.resolveBitmap(1);
    await expect(newer).resolves.toBe('shown');
    bitmaps.resolveBitmap(0);

    await expect(older).resolves.toBe('stale');
    engine.lifecycle.dispose();
  });

  const screenDrawsBitmap = (screen: StubRasterSurface, backend: RecordingRasterBackend, bitmapId: string): boolean =>
    screen.callLog
      .filter((entry) => entry.op === 'drawImage')
      .some((entry) => {
        const surfaceId = (entry.args[0] as { __recordingId?: string }).__recordingId;
        const surface = surfaceId ? backend.surfaceById(surfaceId) : undefined;
        return !!surface && backend.drawSourcesFor(surface).includes(`bitmap-${bitmapId}`);
      });

  const createPublishedGuardedPreview = async (layer: CanvasRasterLayerContractV2 = guardableLayer('L')) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const backend = createPixelWritingBackend(undefined, createRecordingRasterBackend());
    const sourceNeedsBitmap =
      layer.type === 'raster' &&
      (layer.source.type === 'image' || (layer.source.type === 'paint' && layer.source.bitmap !== null));
    let bitmapCall = 0;
    backend.createImageBitmap = vi.fn(() =>
      Promise.resolve(recordingBitmap(sourceNeedsBitmap && bitmapCall++ === 0 ? 'layer-source' : 'guarded-preview'))
    );
    const document = { ...emptyDoc(), stacks: stacksFrom([layer]), selectedLayerId: layer.id };
    const reactive = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend,
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });
    const screen = createFakeCanvas();
    engine.surface.attach(screen.element, createFakeCanvas().element);
    raf.flush();
    const exported = await engine.exports.exportLayerPixels(layer.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a guardable export');
    }
    await expect(
      engine.previews.setGuardedFilterPreview(
        layer.id,
        { imageName: 'guarded-preview', rect: exported.rect },
        exported.guard
      )
    ).resolves.toBe('shown');
    raf.flush();
    expect(screenDrawsBitmap(screen.surface, backend, 'guarded-preview')).toBe(true);
    screen.surface.callLog.length = 0;
    return { backend, document, engine, layer, raf, screen: screen.surface, ...reactive };
  };

  it('publishes an in-flight guarded preview across an active-project away/back transition', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), stacks: stacksFrom([guardableLayer('L')]) };
    const reactive = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });
    const exported = await engine.exports.exportLayerPixels('L');
    if (exported.status !== 'ok') {
      throw new Error('expected a guardable export');
    }
    const pending = engine.previews.setGuardedFilterPreview(
      'L',
      { imageName: 'filtered', rect: exported.rect },
      exported.guard
    );
    await flushMicrotasks();

    reactive.setActiveProjectId('p2');
    reactive.setActiveProjectId('p1');
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('shown');
    engine.lifecycle.dispose();
  });

  it('unsubscribes project-preview lifecycle handling on dispose', () => {
    const reactive = createReactiveStore(emptyDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });

    expect(reactive.listenerCount()).toBe(1);
    engine.lifecycle.dispose();
    expect(reactive.listenerCount()).toBe(0);
  });

  it('clears an already-published guarded preview when the source contract changes', async () => {
    const harness = await createPublishedGuardedPreview();
    harness.setDocument({
      ...harness.document,
      stacks: stacksFrom([
        { ...harness.layer, source: { ...harness.layer.source, fill: '#000' } } as CanvasLayerContract,
      ]),
    });
    harness.raf.flush();

    expect(screenDrawsBitmap(harness.screen, harness.backend, 'guarded-preview')).toBe(false);
    harness.engine.lifecycle.dispose();
  });
});

const fullClearCount = (surface: StubRasterSurface): number =>
  surface.callLog.filter((entry) => entry.op === 'clearRect' && entry.args[0] === 0 && entry.args[1] === 0).length;

describe('document mirror wiring: prop vs source change (paint-pixel survival)', () => {
  const paintOneStroke = async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { setDocument, store } = createReactiveStore(paintDoc());
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const bitmapStore = createSpyBitmapStore();
    const resolver = vi.fn((_imageName: string) => Promise.resolve(new Blob()));
    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    raf.flush();
    await flushMicrotasks();
    raf.flush();

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    const paintCache = surfaces.find((surface) => surface.callLog.some((entry) => entry.op === 'getImageData'));
    return { bitmapStore, engine, paintCache: paintCache!, raf, resolver, setDocument };
  };

  it('keeps an unflushed paint layer’s pixels on a transform/opacity-only change (no re-rasterize)', async () => {
    const { engine, paintCache, raf, resolver, setDocument } = await paintOneStroke();

    expect(paintCache.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    const clearsBefore = fullClearCount(paintCache);

    const doc = engine.document.getDocument()!;
    const layer = getDocumentLeaves(doc)[0]!;
    setDocument({
      ...doc,
      stacks: stacksFrom([{ ...layer, opacity: 0.5, transform: { ...layer.transform, x: 12, y: -4 } }]),
    });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(fullClearCount(paintCache)).toBe(clearsBefore);
    expect(resolver).not.toHaveBeenCalled();
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]!.opacity).toBe(0.5);

    engine.lifecycle.dispose();
  });

  it('commitLayerConversion refuses conversion when the live layer is locked', async () => {
    const { dispatch, projectId, store } = createReducerBackedStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels('a')).status).toBe('ok');
    const expectedUnlocked = getDocumentLeaves(engine.document.getDocument()!)[0]!;
    if (expectedUnlocked.type !== 'raster') {
      throw new Error('expected raster layer');
    }
    const converted: CanvasLayerContract = {
      ...expectedUnlocked,
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      type: 'control',
      withTransparencyEffect: false,
    };
    dispatch({ id: expectedUnlocked.id, patch: { isLocked: true }, type: 'updateCanvasLayer' });
    dispatch.mockClear();

    expect(engine.layers.commitLayerConversion('Convert', expectedUnlocked, converted)).toBe(false);
    expect(dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'convertCanvasLayer' }));
    expect(getDocumentLeaves(engine.document.getDocument()!)[0]).toMatchObject({ isLocked: true, type: 'raster' });
    engine.lifecycle.dispose();
  });

  describe('raster layers whose adjustments reach the control stack', () => {
    const CLARITY: CanvasAdjustmentsContract = [
      { amount: 50, id: 'clarity', isEnabled: true, size: 10, type: 'clarity' },
    ];
    const adjustedEngine = (createSpatialExecutor: () => SpatialAdjustmentExecutor) => {
      const layer = { ...rasterLayer('a'), adjustments: CLARITY, isEnabled: false } as CanvasRasterLayerContractV2;
      const { dispatch, projectId, store } = createReducerBackedStore({ ...makeDoc(), stacks: stacksFrom([layer]) });
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        createSpatialExecutor,
        getMainModelBase: () => 'anima',
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      return { dispatch, engine, layer: getDocumentLayer(engine.document.getDocument(), 'a')! };
    };
    const tooLarge = () => ({
      dispose: () => undefined,
      run: (job: { buffer: ArrayBuffer }) => Promise.resolve({ buffer: job.buffer, status: 'too-large' as const }),
    });

    it('converts in place with baked pixels at an identity transform; undo restores the stack', async () => {
      const { engine, layer } = adjustedEngine(createInlineSpatialExecutor);

      expect(await engine.layers.convertLayerToControl('Convert to control', layer)).toBe('converted');
      expect(getDocumentLayer(engine.document.getDocument(), 'a')).toMatchObject({
        adapter: { kind: 'anima_lllite', model: null },
        isEnabled: false,
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'control',
      });
      expect(engine.history.getEntries().past.at(-1)).toBe('Convert to control');

      await engine.history.undo();
      expect(getDocumentLayer(engine.document.getDocument(), 'a')).toEqual(layer);
      engine.lifecycle.dispose();
    });

    it('refuses both, naming the layer, when its stack cannot run', async () => {
      const { dispatch, engine, layer } = adjustedEngine(tooLarge);
      dispatch.mockClear();
      const refusal = { ownerId: 'a', ownerName: 'a', reason: 'too-large', status: 'adjustment-refused' };

      expect(await engine.layers.copyLayerToControl('a')).toEqual(refusal);
      expect(await engine.layers.convertLayerToControl('Convert to control', layer)).toEqual(refusal);
      expect(dispatch).not.toHaveBeenCalled();
      expect(getDocumentLayer(engine.document.getDocument(), 'a')).toBe(layer);
      engine.lifecycle.dispose();
    });
  });

  it('flushing after a prop-only change persists the painted (non-blank) surface', async () => {
    const { bitmapStore, engine, paintCache, raf, setDocument } = await paintOneStroke();
    const clearsBefore = fullClearCount(paintCache);

    const doc = engine.document.getDocument()!;
    const layer = getDocumentLeaves(doc)[0]!;
    setDocument({ ...doc, stacks: stacksFrom([{ ...layer, opacity: 0.25 }]) });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    await engine.lifecycle.flushPendingUploads();

    expect(bitmapStore.flushPendingUploads).toHaveBeenCalled();
    expect(fullClearCount(paintCache)).toBe(clearsBefore);
    expect(paintCache.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('re-rasterizes when the paint layer’s source genuinely changes (swap to a persisted bitmap)', async () => {
    const { engine, raf, resolver, setDocument } = await paintOneStroke();
    expect(resolver).not.toHaveBeenCalled();

    const doc = engine.document.getDocument()!;
    const layer = getDocumentLeaves(doc)[0] as CanvasRasterLayerContractV2;
    setDocument({
      ...doc,
      stacks: stacksFrom([
        {
          ...layer,
          source: {
            bitmap: { contentHash: 'h', height: 100, imageName: 'persisted', width: 100 },
            type: 'paint' as const,
          },
        },
      ]),
    });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(resolver).toHaveBeenCalledWith('persisted', expect.any(AbortSignal));

    engine.lifecycle.dispose();
  });
});

describe('gesture guard: nudge / commitStructural mid-stroke', () => {
  const startOpenStroke = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(30, 30));
    return { dispatch, engine, overlay };
  };

  it('refuses commitStructural as gesture-active while a stroke gesture is open, then commits after it ends', () => {
    const { dispatch, engine, overlay } = startOpenStroke();
    const forward: EngineTestAction = { id: null, type: 'setCanvasSelectedLayer' };
    const inverse: EngineTestAction = { id: 'paint1', type: 'setCanvasSelectedLayer' };

    expect(engine.layers.canCommitStructural()).toBe(false);
    expect(engine.layers.commitStructural('Select', forward, inverse)).toEqual({ status: 'gesture-active' });
    expect(dispatch.mock.calls.some((call) => call[0] === forward)).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);

    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    expect(engine.layers.canCommitStructural()).toBe(true);
    expect(engine.layers.commitStructural('Select', forward, inverse)).toEqual({ status: 'committed' });
    expect(dispatch.mock.calls.some((call) => call[0] === forward)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.lifecycle.dispose();
  });
});

describe('brush cursor ring: live size updates', () => {
  it('invalidates the overlay when the brush size changes without a pointer move', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    overlay.fire('pointermove', pointerAt(30, 30, { buttons: 0 }));
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    engine.tools.stepBrushSize(1);
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    engine.stores.brushOptions.set({ ...engine.stores.brushOptions.get(), size: 123 });
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    engine.stores.brushOptions.set({ ...engine.stores.brushOptions.get(), color: '#123456', opacity: 0.4 });
    engine.stores.eraserOptions.set({ ...engine.stores.eraserOptions.get(), size: 77 });
    expect(raf.pendingCount()).toBe(0);

    engine.lifecycle.dispose();
  });
});

describe('doc-replace mid-gesture: cancels the active tool gesture', () => {
  it('cancels a bbox drag on a document swap, clears the preview, and commits nothing on pointer-up', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { setDocument, store } = createReactiveStore(paintDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('bbox');

    overlay.fire('pointerdown', pointerAt(40, 40));
    overlay.fire('pointermove', pointerAt(60, 60));
    expect(engine.interaction.get('bboxPreview')).toEqual(engine.stores.bboxPreview.get());
    expect(engine.stores.bboxPreview.get()).not.toBeNull();

    setDocument({ ...paintDoc(), height: 200, width: 200 });

    expect(engine.stores.bboxPreview.get()).toBeNull();

    overlay.fire('pointerup', pointerAt(60, 60, { buttons: 0 }));
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'setCanvasBbox')).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.lifecycle.dispose();
  });
});

describe('genuine tool switch mid-gesture', () => {
  it('cancels the stroke through the outgoing brush and routes none of the drag tail to the new tool', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const strokes: StrokeCommittedEvent[] = [];
    engine.tools.onStrokeCommitted((event) => strokes.push(event));
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    engine.tools.setTool('move');

    expect(engine.tools.canTargetLayerFromContextMenu()).toBe(true);
    const dispatchesAfterSwitch = dispatch.mock.calls.length;
    overlay.fire('pointermove', pointerAt(80, 60));
    overlay.fire('pointerup', pointerAt(80, 60, { buttons: 0 }));

    expect(dispatch.mock.calls.length).toBe(dispatchesAfterSwitch);
    expect(strokes).toEqual([]);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.lifecycle.dispose();
  });
});

describe('zoom-cost: overlay-only frames skip the document composite', () => {
  const createFireableSurfaceCanvas = (
    width = 100,
    height = 100
  ): {
    element: HTMLCanvasElement;
    fire: (type: string, event: Partial<PointerEvent>) => void;
    surface: StubRasterSurface;
  } => {
    const surface = createTestStubRasterBackend().createSurface(width, height);
    const listeners = new Map<string, Set<(event: Event) => void>>();
    const element = {
      addEventListener: (type: string, handler: (event: Event) => void) => {
        const set = listeners.get(type) ?? new Set();
        set.add(handler);
        listeners.set(type, set);
      },
      getBoundingClientRect: () => ({ bottom: height, height, left: 0, right: width, top: 0, width, x: 0, y: 0 }),
      getContext: () => surface.ctx,
      height,
      releasePointerCapture: () => {},
      removeEventListener: (type: string, handler: (event: Event) => void) => {
        listeners.get(type)?.delete(handler);
      },
      setPointerCapture: () => {},
      width,
    } as unknown as HTMLCanvasElement;
    const fire = (type: string, event: Partial<PointerEvent>): void => {
      for (const handler of listeners.get(type) ?? []) {
        handler({ preventDefault: () => {}, ...event } as unknown as Event);
      }
    };
    return { element, fire, surface };
  };

  const compositeOps = (surface: StubRasterSurface): RasterCallLogEntry[] =>
    surface.callLog.filter((e) => e.op === 'drawImage' || e.op === 'clearRect' || e.op === 'fillRect');

  it('a hover pointermove redraws only the overlay, never the screen composite', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFireableSurfaceCanvas();
    const overlay = createFireableSurfaceCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('brush');

    raf.flush();
    await flushMicrotasks();
    raf.flush();

    screen.surface.callLog.length = 0;
    overlay.surface.callLog.length = 0;

    overlay.fire('pointermove', pointerAt(40, 40, { buttons: 0 }));
    raf.flush();

    expect(compositeOps(screen.surface)).toHaveLength(0);
    expect(overlay.surface.callLog.some((e) => e.op === 'clearRect')).toBe(true);
    expect(overlay.surface.callLog.some((e) => e.op === 'arc')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('composites with smoothing OFF when zoomed in and ON when zoomed out', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const findSet = (surface: StubRasterSurface, prop: string): unknown[] =>
      surface.callLog.filter((e) => e.op === 'set' && e.args[0] === prop).map((e) => e.args[1]);

    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    screen.surface.callLog.length = 0;
    engine.viewport.getViewport().zoomAtPoint(20, { x: 0, y: 0 });
    raf.flush();
    expect(findSet(screen.surface, 'imageSmoothingEnabled')).toContain(false);

    screen.surface.callLog.length = 0;
    engine.viewport.getViewport().zoomAtPoint(0.5, { x: 0, y: 0 });
    raf.flush();
    expect(findSet(screen.surface, 'imageSmoothingEnabled')).toContain(true);

    engine.lifecycle.dispose();
  });
});

const fillStyleSets = (surface: StubRasterSurface): unknown[] =>
  surface.callLog.filter((e) => e.op === 'set' && e.args[0] === 'fillStyle').map((e) => e.args[1]);

describe('checker colors: fed-token tile rebuild', () => {
  it('builds the fallback-colored tile when never fed, then rebuilds with fed colors and recomposites', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const base = createTestStubRasterBackend();
    const tiles: StubRasterSurface[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        if (w === 16 && h === 16) {
          tiles.push(surface);
        }
        return surface;
      },
    };

    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend,
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();

    expect(tiles).toHaveLength(1);
    expect(fillStyleSets(tiles[0]!)).toEqual([DEFAULT_CHECKER_COLORS.a, DEFAULT_CHECKER_COLORS.b]);

    engine.stores.checkerColors.set({ a: '#010101', b: '#020202' });
    raf.flush();
    await flushMicrotasks();

    expect(tiles).toHaveLength(2);
    expect(fillStyleSets(tiles[1]!)).toEqual(['#010101', '#020202']);

    engine.lifecycle.dispose();
  });
});

const noLayerDoc = (overrides: Partial<CanvasDocumentContractV3> = {}): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 1000,
  stacks: stacksFrom([]),
  selectedLayerId: null,
  version: 3,
  width: 1000,
  ...overrides,
});

describe('fitToView: content ∪ bbox', () => {
  it('unions a renderable layer that lies beyond the bbox into the fit', () => {
    // Content at (900,900) extends the bbox union to 1000px, fitting at about 0.304.
    const layer: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'a',
      isEnabled: true,
      isLocked: false,
      name: 'a',
      opacity: 1,
      source: { image: { height: 100, imageName: 'a', width: 100 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 900, y: 900 },
      type: 'raster',
    };
    const { store } = createReactiveStore(noLayerDoc({ stacks: stacksFrom([layer]) }));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.surface.resize(400, 400, 1);

    engine.viewport.fitToView();

    // Union spans (0,0)..(1000,1000) = 1000px; avail 304 → 0.304.
    expect(engine.viewport.getViewport().getZoom()).toBeCloseTo(0.304, 2);
    engine.lifecycle.dispose();
  });

  it('fits on the first show only, so a re-shown canvas keeps its zoom and pan', () => {
    const { store } = createReactiveStore(noLayerDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.surface.resize(400, 400, 1);

    engine.viewport.fitToViewOnFirstShow();
    const fittedZoom = engine.viewport.getViewport().getZoom();
    engine.viewport.getViewport().zoomAtPoint(2, { x: 200, y: 200 });
    engine.viewport.getViewport().panBy({ x: 30, y: -20 });
    const pannedView = engine.viewport.getViewport().getState();

    engine.viewport.fitToViewOnFirstShow();

    expect(fittedZoom).not.toBeCloseTo(2, 2);
    expect(engine.viewport.getViewport().getZoom()).toBeCloseTo(2, 2);
    expect(engine.viewport.getViewport().getState()).toEqual(pannedView);

    engine.viewport.fitToView();

    expect(engine.viewport.getViewport().getZoom()).toBeCloseTo(fittedZoom, 2);
    engine.lifecycle.dispose();
  });
});

describe('overview: composites the fit-to-view extent', () => {
  const paintAt = (id: string, x: number, y: number, isEnabled = true): CanvasLayerContract => ({
    ...(rasterLayer(id) as CanvasRasterLayerContractV2),
    isEnabled,
    source: { bitmap: { height: 100, imageName: `${id}-pixels`, width: 100 }, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y },
  });
  const overviewTarget = (): HTMLCanvasElement =>
    ({
      getContext: () => createTestStubRasterBackend().createSurface(1, 1).ctx,
      height: 0,
      width: 0,
    }) as unknown as HTMLCanvasElement;

  it('sizes, translates and reports the frame ∪ enabled content, leaving disabled layers out', () => {
    const composite = vi.spyOn(compositor, 'compositeDocument');
    // Frame (0,0)..(100,100) ∪ enabled paint (-200,-50)..(-100,50) = (-200,-50) 300×150; the disabled layer at
    // (500,500) would stretch it to 800×650.
    const { store } = createReactiveStore(
      noLayerDoc({ stacks: stacksFrom([paintAt('outside', -200, -50), paintAt('hidden', 500, 500, false)]) })
    );
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const target = overviewTarget();

    const shown = engine.previews.drawDocumentOverview(target, 150);

    expect(shown).toEqual({ height: 150, width: 300, x: -200, y: -50 });
    expect({ height: target.height, width: target.width }).toEqual({ height: 75, width: 150 });
    // Half scale, shifted so document (-200,-50) lands on target pixel (0,0).
    expect(composite.mock.calls.at(-1)?.[3]).toEqual({ a: 0.5, b: 0, c: 0, d: 0.5, e: 100, f: 25 });
    composite.mockRestore();
    engine.lifecycle.dispose();
  });
});

describe('masks while a result is staged', () => {
  const candidate: CanvasStagingCandidateContract = {
    height: 20,
    imageName: 'staged.png',
    imageUrl: '/staged.png',
    placement: { height: 20, opacity: 1, width: 20, x: 0, y: 0 },
    queuedAt: '2026-07-16T00:00:00.000Z',
    sourceQueueItemId: 'queue',
    thumbnailUrl: '/staged-thumb.png',
    width: 20,
  };
  const MASKED = { drawn: ['a', 'm'], regionOverlays: true };
  const UNMASKED = { drawn: ['a'], regionOverlays: false };

  const setup = (stacks = stacksFrom([maskLayer('m'), rasterLayer('a')])) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const windowListeners = new Map<string, Set<(event: unknown) => void>>();
    vi.stubGlobal('addEventListener', (type: string, listener: (event: unknown) => void) =>
      windowListeners.set(type, (windowListeners.get(type) ?? new Set()).add(listener))
    );
    vi.stubGlobal('removeEventListener', () => {});
    const composite = vi.spyOn(compositor, 'compositeDocument');
    let canvas = makeCanvas({ ...makeDoc(), stacks });
    const listeners = new Set<() => void>();
    const store: EngineStore = {
      dispatch: vi.fn(),
      getState: () => ({ activeProjectId: 'p1', projects: [{ canvas, id: 'p1' }] }) as unknown as WorkbenchState,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };
    const update = (next: Partial<typeof canvas>): void => {
      canvas = { ...canvas, ...next };
      for (const listener of listeners) {
        listener();
      }
    };
    const select = (selectedLayerId: string): void => update({ document: { ...canvas.document, selectedLayerId } });
    const engine = createCanvasEngine({
      backend: {
        ...createTestStubRasterBackend(),
        createImageBitmap: () => Promise.resolve({ close: vi.fn(), height: 20, width: 20 } as unknown as ImageBitmap),
      },
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    overlay.fire('pointerenter', pointerAt(5, 5));
    const frame = () => {
      composite.mockClear();
      raf.flush();
      const options = composite.mock.calls.at(-1)?.[4];
      return {
        drawn: options?.preparation?.leaves.map((leaf) => leaf.id),
        full: composite.mock.calls.filter((call) => call[4]?.damage?.kind === 'full').length,
        regionOverlays: options?.regionOverlays,
      };
    };
    const showPreview = async (): Promise<void> => {
      update({ stagingArea: { ...canvas.stagingArea, isVisible: true, pendingImages: [candidate] } });
      engine.previews.setStagedPreview({ imageName: candidate.imageName, placement: candidate.placement });
      await flushMicrotasks();
    };
    const useTool = (tool: ToolId): void => {
      engine.tools.setTool(tool);
      raf.flush();
    };
    const key = (type: 'keydown' | 'keyup', code: string): void =>
      windowListeners
        .get(type)
        ?.forEach((listener) =>
          listener({ code, key: code === 'Space' ? ' ' : code, preventDefault: () => {}, target: null, type })
        );
    const expectPressHidden = (): void => {
      overlay.fire('pointerdown', pointerAt(5, 5));
      const pressed = frame();
      expect(pressed.drawn ?? []).not.toContain('m');
      expect(pressed.regionOverlays).not.toBe(true);
      overlay.fire('pointerup', pointerAt(5, 5, { buttons: 0 }));
      expect(frame().drawn ?? []).not.toContain('m');
    };
    const dispose = (): void => {
      composite.mockRestore();
      engine.lifecycle.dispose();
    };
    return { dispose, engine, expectPressHidden, frame, key, overlay, select, showPreview, store, update, useTool };
  };

  it('hides masks and region tints under a shown preview, showing them only while a gesture edits a mask', async () => {
    const h = setup();
    const { frame, overlay } = h;
    expect(frame()).toEqual({ ...MASKED, full: 1 });

    h.update({ stagingArea: { ...makeCanvas(makeDoc()).stagingArea, isVisible: false, pendingImages: [candidate] } });
    expect(frame().drawn).toBeUndefined();

    await h.showPreview();
    expect(frame()).toEqual({ ...UNMASKED, full: 1 });

    h.select('m');
    expect(frame()).toMatchObject({ full: 0 });

    h.useTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    expect(frame()).toEqual({ ...MASKED, full: 1 });
    overlay.fire('pointermove', pointerAt(15, 15));
    expect(frame()).toEqual({ ...MASKED, full: 0 });
    overlay.fire('pointerup', pointerAt(15, 15, { buttons: 0 }));
    expect(frame()).toEqual({ ...UNMASKED, full: 1 });

    overlay.fire('pointerdown', pointerAt(5, 5));
    expect(frame()).toEqual({ ...MASKED, full: 1 });
    overlay.fire('pointercancel', pointerAt(5, 5));
    expect(frame()).toEqual({ ...UNMASKED, full: 1 });

    h.useTool('move');
    overlay.fire('pointerdown', pointerAt(5, 5));
    expect(frame()).toMatchObject({ full: 0 });
    overlay.fire('pointermove', pointerAt(6, 6));
    overlay.fire('pointerup', pointerAt(6, 6, { buttons: 0 }));
    expect(frame().drawn ?? UNMASKED.drawn).toEqual(UNMASKED.drawn);

    for (const end of ['pointerup', 'pointercancel'] as const) {
      overlay.fire('pointerdown', pointerAt(5, 5));
      expect(frame()).toMatchObject({ full: 0 });
      overlay.fire('pointermove', pointerAt(15, 15));
      expect(frame()).toEqual({ ...MASKED, full: 1 });
      overlay.fire('pointermove', pointerAt(16, 16));
      expect(frame()).toMatchObject(MASKED);
      overlay.fire(end, pointerAt(16, 16, { buttons: 0 }));
      expect(frame()).toEqual({ ...UNMASKED, full: 1 });
    }

    for (const tool of ['view', 'colorPicker'] as const) {
      h.useTool(tool);
      overlay.fire('pointerdown', pointerAt(5, 5));
      expect(frame()).toMatchObject({ full: 0 });
      overlay.fire('pointermove', pointerAt(15, 15));
      overlay.fire('pointerup', pointerAt(15, 15, { buttons: 0 }));
      const after = frame();
      expect(after.drawn ?? UNMASKED.drawn).toEqual(UNMASKED.drawn);
      expect(after.full).toBe(tool === 'view' ? 1 : 0);
    }

    h.engine.previews.setStagedPreview(null);
    expect(frame()).toEqual({ ...MASKED, full: 1 });
    h.useTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    overlay.fire('pointermove', pointerAt(15, 15));
    expect(frame()).toEqual({ ...MASKED, full: 0 });
    overlay.fire('pointerup', pointerAt(15, 15, { buttons: 0 }));
    expect(frame()).toMatchObject(MASKED);
    h.dispose();
  });
});

describe('beginTransform', () => {
  const setup = (layers: CanvasLayerContract[], selectedLayerId: string | null) => {
    const { dispatch, projectId, store } = createReducerBackedStore({
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      selectedLayerId,
      stacks: stacksFrom(layers),
      version: 3,
      width: 100,
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    return { dispatch, engine };
  };

  it('selects the layer and opens its session now, even with transform already active and no session', () => {
    const { engine } = setup([rasterLayer('a'), rasterLayer('b')], null);
    engine.tools.setTool('transform');
    expect(engine.stores.transformSession.get()).toBeNull();

    expect(engine.layers.beginTransform('b')).toBe('started');
    expect(engine.document.getDocument()?.selectedLayerId).toBe('b');
    expect(engine.stores.activeTool.get()).toBe('transform');
    expect(engine.stores.transformSession.get()?.layerId).toBe('b');
    engine.lifecycle.dispose();
  });

  const strokeThenTransform = (options: { existingLayer?: boolean; uploadImage?: () => Promise<never> } = {}) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const windowListeners = new Map<string, Set<(event: unknown) => void>>();
    vi.stubGlobal('addEventListener', (type: string, listener: (event: unknown) => void) =>
      windowListeners.set(type, (windowListeners.get(type) ?? new Set()).add(listener))
    );
    vi.stubGlobal('removeEventListener', (type: string, listener: (event: unknown) => void) =>
      windowListeners.get(type)?.delete(listener)
    );
    const existing: CanvasLayerContract = {
      ...(rasterLayer('paint') as CanvasRasterLayerContractV2),
      source: { bitmap: null, type: 'paint' },
    };
    const { dispatch, projectId, store } = createReducerBackedStore({
      ...makeDoc(),
      selectedLayerId: options.existingLayer ? existing.id : null,
      stacks: stacksFrom(options.existingLayer ? [existing] : []),
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      ...(options.uploadImage ? { uploadImage: options.uploadImage } : { bitmapStore: createSpyBitmapStore() }),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    const strokes: StrokeCommittedEvent[] = [];
    engine.tools.onStrokeCommitted((event) => strokes.push(event));
    const refusals: CanvasEditRefusal[] = [];
    engine.tools.onEditRefused((refusal) => refusals.push(refusal));
    engine.tools.setTool('brush');
    overlay.fire('pointerenter', pointerAt(20, 20));
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    const stroke = strokes[0]!;
    expect(engine.layers.beginTransform(stroke.layerId)).toBe('started');
    expect(engine.stores.activeTool.get()).toBe('transform');
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.interaction.get('pendingTransformLayerId')).toBe(stroke.layerId);
    const key = (type: 'keydown' | 'keyup', code: string) =>
      windowListeners
        .get(type)
        ?.forEach((listener) =>
          listener({ code, key: code === 'Space' ? ' ' : code, preventDefault: () => {}, target: null, type })
        );
    const persist = () =>
      dispatch({
        id: stroke.layerId,
        source: {
          bitmap: { height: stroke.dirtyRect.height, imageName: 'stroke.png', width: stroke.dirtyRect.width },
          offset: { x: stroke.dirtyRect.x, y: stroke.dirtyRect.y },
          type: 'paint',
        },
        type: 'updateCanvasLayerSource',
      });
    const sessionLayerId = () => engine.stores.transformSession.get()?.layerId ?? null;
    return { engine, key, layerId: stroke.layerId, persist, refusals, sessionLayerId };
  };

  it('keeps waiting through a Space pan held meanwhile, opening when the pixels land under the hold', () => {
    const h = strokeThenTransform();
    h.key('keydown', 'Space');
    expect(h.engine.stores.activeTool.get()).toBe('view');
    h.persist();
    expect(h.sessionLayerId()).toBe(h.layerId);
    h.key('keyup', 'Space');
    expect(h.engine.stores.activeTool.get()).toBe('transform');
    expect(h.sessionLayerId()).toBe(h.layerId);
    h.engine.lifecycle.dispose();
  });

  it.each([
    ['a tool switch', (engine: CanvasEngine) => engine.tools.setTool('brush')],
    ['Escape', (engine: CanvasEngine) => engine.tools.handleEscapePriority({ gestureWasActive: false })],
  ])('stops waiting on %s', (_, end) => {
    const h = strokeThenTransform();
    end(h.engine);
    expect(h.engine.interaction.get('pendingTransformLayerId')).toBeNull();
    h.persist();
    expect(h.sessionLayerId()).toBeNull();
    h.engine.lifecycle.dispose();
  });

  it('stops waiting on an undo, whose replay lands on a document the request never saw', async () => {
    const h = strokeThenTransform({ existingLayer: true });
    expect(h.layerId).toBe('paint');
    await h.engine.history.undo();
    expect(h.engine.interaction.get('pendingTransformLayerId')).toBeNull();
    h.persist();
    expect(h.sessionLayerId()).toBeNull();
    h.engine.lifecycle.dispose();
  });
});

describe('transform session', () => {
  it('image layer Apply commits one structural transform with the exact inverse', async () => {
    const { store } = createReactiveStore(selectedImageDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    engine.tools.setTool('transform');
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 });
    dispatch.mockClear();
    engine.layers.applyTransform();

    const layerDispatches = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'updateCanvasLayer');
    expect(layerDispatches).toHaveLength(1);
    expect(layerDispatches[0]).toEqual({
      id: 'a',
      patch: { transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 } },
      type: 'updateCanvasLayer',
    });
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.stores.canUndo.get()).toBe(true);

    dispatch.mockClear();
    await engine.history.undo();
    const undoDispatch = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'updateCanvasLayer');
    expect(undoDispatch).toEqual({
      id: 'a',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });

    engine.lifecycle.dispose();
  });

  it('handleEscapePriority cancels an open transform session (chain: transform → deselect)', () => {
    const { store } = createReactiveStore(selectedImageDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    engine.tools.setTool('transform');
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 });
    expect(engine.stores.transformSession.get()).not.toBeNull();

    engine.tools.handleEscapePriority({ gestureWasActive: false });
    expect(engine.stores.transformSession.get()).toBeNull();

    engine.lifecycle.dispose();
  });

  it('paint layer Apply bakes pixels through the matrix, resets the transform, and composes one undo entry', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const movedPaint = paintDoc();
    const movedLayer = getDocumentLeaves(movedPaint)[0] as CanvasRasterLayerContractV2;
    movedLayer.source = {
      bitmap: { height: 50, imageName: 'paint1-bmp', width: 50 },
      offset: { x: 0, y: 0 },
      type: 'paint',
    };
    movedLayer.transform = { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 };
    const { store } = createReactiveStore(movedPaint);
    const dispatch = store.dispatch as Mock;
    const bitmapStore = createSpyBitmapStore();
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = base.createSurface(w, h);
        surfaces.push(surface);
        return surface;
      },
    };
    const engine = createCanvasEngine({
      backend,
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();

    const surfacesBeforeApply = surfaces.length;
    const dirtyBefore = bitmapStore.markLayerDirty.mock.calls.length;

    engine.tools.setTool('transform');
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 20 });
    dispatch.mockClear();
    engine.layers.applyTransform();

    // A 50x50 cache scaled by two at (10,20) bakes into a 100x100 surface; subtracting the baked origin leaves zero
    // translation.
    const baked = surfaces[surfacesBeforeApply];
    expect(baked).toBeDefined();
    expect(baked!.width).toBe(100);
    expect(baked!.height).toBe(100);
    const setTransforms = baked!.callLog.filter((entry) => entry.op === 'setTransform');
    expect(
      setTransforms.some(
        (entry) => entry.args[0] === 2 && entry.args[3] === 2 && entry.args[4] === 0 && entry.args[5] === 0
      )
    ).toBe(true);
    expect(baked!.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    expect(
      baked!.callLog.some(
        (entry) => entry.op === 'set' && entry.args[0] === 'imageSmoothingEnabled' && entry.args[1] === true
      )
    ).toBe(true);

    const resetDispatch = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'updateCanvasLayer');
    expect(resetDispatch).toEqual({
      id: 'paint1',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });

    expect(bitmapStore.markLayerDirty.mock.calls.length).toBeGreaterThan(dirtyBefore);
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.stores.canUndo.get()).toBe(true);

    dispatch.mockClear();
    await engine.history.undo();
    const undoDispatch = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'updateCanvasLayer');
    expect(undoDispatch).toEqual({
      id: 'paint1',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 } },
      type: 'updateCanvasLayer',
    });
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.lifecycle.dispose();
  });

  it('tears down the transform session on a document replace', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { setDocument, store } = createReactiveStore(selectedImageDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();

    engine.tools.setTool('transform');
    expect(engine.stores.transformSession.get()).not.toBeNull();

    setDocument({ ...selectedImageDoc(), height: 200, width: 200 }, 1);

    expect(engine.stores.transformSession.get()).toBeNull();

    engine.lifecycle.dispose();
  });

  it('cancels the session (and its preview override) when its layer is deleted mid-session', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const doc = selectedImageDoc();
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();

    engine.tools.setTool('transform');
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 5 });
    expect(engine.stores.transformSession.get()).not.toBeNull();

    setDocument({ ...doc, stacks: stacksFrom([]) });

    expect(engine.stores.transformSession.get()).toBeNull();

    engine.lifecycle.dispose();
  });

  describe('temp-tool switch (space/alt hold)', () => {
    it('preserves the session and its numeric edits, resuming after the hold ends', () => {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

      const { store } = createReactiveStore(selectedImageDoc());
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore: createSpyBitmapStore(),
        imageResolver: () => Promise.resolve(new Blob()),
        projectId: 'p1',
        store,
      });
      const screen = createFakeCanvas();
      const overlay = createFakeCanvas();
      engine.surface.attach(screen.element, overlay.element);
      raf.flush();

      engine.tools.setTool('transform');
      engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 1, x: 5, y: 0 });
      const edited = { rotation: 0, scaleX: 2, scaleY: 1, x: 5, y: 0 };
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      const setTool = engine.tools.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void;

      setTool('view', { temporary: true });
      expect(engine.stores.activeTool.get()).toBe('view');
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      setTool('transform', { temporary: true });
      expect(engine.stores.activeTool.get()).toBe('transform');
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      engine.lifecycle.dispose();
    });
  });
});

describe('engine selection: select all / deselect / invert + hasSelection store', () => {
  it('selectAll sets hasSelection; deselect clears it', () => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const { store } = createFakeStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect(engine.stores.hasSelection.get()).toBe(false);
    engine.selection.selectAll();
    expect(engine.stores.hasSelection.get()).toBe(true);
    engine.selection.deselect();
    expect(engine.stores.hasSelection.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('selection changes are their own history steps, undone and redone through the engine', async () => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const { store } = createFakeStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.selection.selectAll();
    engine.selection.deselect();
    expect(engine.history.getEntries().past).toEqual(['Select all', 'Deselect']);

    await engine.history.undo();
    expect(engine.stores.hasSelection.get()).toBe(true);
    await engine.history.undo();
    expect(engine.stores.hasSelection.get()).toBe(false);
    await engine.history.redo();
    expect(engine.stores.hasSelection.get()).toBe(true);
    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: ['Deselect'], past: ['Select all'] });
    engine.lifecycle.dispose();
  });
});

describe('engine selection: fill / erase', () => {
  const SELECTION_STEPS = new Set(['Select', 'Select all']);
  const editSteps = (engine: CanvasEngine): string[] =>
    engine.history.getEntries().past.filter((label) => !SELECTION_STEPS.has(label));
  const makeEngine = (doc: CanvasDocumentContractV3) => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const { store } = createFakeStore(doc);
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createPixelWritingBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    return { bitmapStore, engine };
  };

  it.each(['fill', 'erase'] as const)('materializes an image control and %ss it as one undo step', async (kind) => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    const before = structuredClone(getDocumentLeaves(h.engine.document.getDocument()!)[0]);
    h.engine.selection.selectAll();
    if (kind === 'fill') {
      h.engine.selection.fillSelection();
    } else {
      h.engine.selection.eraseSelection();
    }
    const after = structuredClone(getDocumentLeaves(h.engine.document.getDocument()!)[0]);
    expect(after).toMatchObject({ id: 'control', source: { type: 'paint' }, type: 'control' });
    await h.engine.history.undo();
    expect(getDocumentLeaves(h.engine.document.getDocument()!)[0]).toEqual(before);
    await h.engine.history.redo();
    expect(getDocumentLeaves(h.engine.document.getDocument()!)[0]).toEqual(after);
    h.engine.lifecycle.dispose();
  });

  it('rolls back a direct control selection edit when masked compositing fails', async () => {
    const h = createControlSelectionHarness({
      source: { bitmap: { height: 10, imageName: 'paint-bitmap', width: 10 }, type: 'paint' },
    });
    await h.publishInitialCache();
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    const originalCtx = beforeCache.surface.ctx;
    const capturedReads: ImageData[] = [];
    const failingCtx = new Proxy(originalCtx, {
      get(target, property, receiver) {
        const value = Reflect.get(target, property, receiver);
        if (property === 'getImageData' && typeof value === 'function') {
          return (...args: unknown[]) => {
            const result = Reflect.apply(value, target, args) as ImageData;
            capturedReads.push(result);
            return result;
          };
        }
        if (property === 'drawImage' && typeof value === 'function') {
          return (...args: unknown[]) => {
            Reflect.apply(value, target, args);
            throw new Error('selection compositing failed');
          };
        }
        return value;
      },
    });
    Object.defineProperty(beforeCache.surface, 'ctx', { configurable: true, value: failingCtx });
    h.engine.tools.setTool('lasso');
    h.overlay.fire('pointerdown', pointerAt(0, 0));
    h.overlay.fire('pointermove', pointerAt(5, 0));
    h.overlay.fire('pointermove', pointerAt(5, 5));
    h.overlay.fire('pointermove', pointerAt(0, 5));
    h.overlay.fire('pointerup', pointerAt(0, 0, { buttons: 0 }));

    expect(() => h.engine.selection.fillSelection()).toThrow('selection compositing failed');

    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(editSteps(h.engine)).toEqual([]);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(capturedReads[0]).toMatchObject({ height: 5, width: 5 });
    expect(
      (beforeCache.surface as StubRasterSurface).callLog.filter((entry) => entry.op === 'putImageData').at(-1)?.args[0]
    ).toBe(capturedReads[0]);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status === 'ok' && restored.guard.cacheVersion).toBe(beforeCache.version);

    Object.defineProperty(beforeCache.surface, 'ctx', { configurable: true, value: originalCtx });
    h.engine.selection.fillSelection();
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('fillSelection on the selected paint layer records one undoable edit + persists', async () => {
    const { bitmapStore, engine } = makeEngine(paintDoc());
    engine.selection.selectAll();
    expect(editSteps(engine)).toEqual([]);
    engine.selection.fillSelection();
    expect(editSteps(engine)).toEqual(['Fill selection']);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledWith('paint1');
    await engine.history.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('clearCaches flushes pending bitmap uploads before invalidating (unflushed strokes survive)', async () => {
    const { bitmapStore, engine } = makeEngine(paintDoc());
    await engine.diagnostics.clearCaches();
    expect(bitmapStore.flushPendingUploads).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });
});

describe('engine selection: marching ants animation + overlay-only', () => {
  it('stops the ants loop (no pending frames) on deselect and on detach', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();

    engine.selection.selectAll();
    raf.flush();
    raf.flush();
    expect(raf.pendingCount()).toBeGreaterThan(0);

    engine.selection.deselect();
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    engine.selection.selectAll();
    raf.flush();
    expect(raf.pendingCount()).toBeGreaterThan(0);
    engine.surface.detach();
    expect(raf.pendingCount()).toBe(0);

    engine.lifecycle.dispose();
  });
});

describe('engine selection: lasso commit through the pipeline', () => {
  it('a lasso drag commits a selection (hasSelection true) without dispatching', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );

    const { store } = createReactiveStore(paintDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('lasso');
    raf.flush();
    dispatch.mockClear();

    overlay.fire('pointerdown', pointerAt(10, 10));
    overlay.fire('pointermove', pointerAt(40, 10));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(10, 40, { buttons: 0 }));

    expect(engine.stores.hasSelection.get()).toBe(true);
    expect(dispatch).not.toHaveBeenCalled();

    engine.lifecycle.dispose();
  });
});

describe('engine selection: document replace clears the selection', () => {
  it('a wholesale document swap deselects', () => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const { setDocument, store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.selection.selectAll();
    expect(engine.stores.hasSelection.get()).toBe(true);
    setDocument(paintDoc(), 1);
    expect(engine.stores.hasSelection.get()).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('text edit session', () => {
  const textSource = (over: Partial<Extract<CanvasLayerSourceContract, { type: 'text' }>> = {}) => ({
    align: 'left' as const,
    color: '#112233',
    content: 'hello',
    fontFamily: 'Inter',
    fontSize: 20,
    fontWeight: 400,
    lineHeight: 1.2,
    type: 'text' as const,
    ...over,
  });

  const textDoc = (over: Partial<CanvasDocumentContractV3> = {}): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom([
      {
        blendMode: 'normal',
        id: 'txt1',
        isEnabled: true,
        isLocked: false,
        name: 'Text 1',
        opacity: 1,
        source: textSource(),
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 5, y: 6 },
        type: 'raster',
      },
    ]),
    selectedLayerId: 'txt1',
    version: 3,
    width: 100,
    ...over,
  });

  const makeEngine = (doc: CanvasDocumentContractV3) => {
    const { setDocument, store } = createReactiveStore(doc);
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const layerActions = () => dispatch.mock.calls.map((call) => call[0] as EngineTestAction);
    return { dispatch, engine, layerActions, setDocument };
  };

  it('create-mode commit dispatches ONE addCanvasLayer with the typed content; undo removes it', async () => {
    const { dispatch, engine, layerActions } = makeEngine(paintDoc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 10, y: 20 });
    expect(engine.stores.textEditSession.get()?.mode).toBe('create');

    dispatch.mockClear();
    engine.layers.commitTextEdit('Typed here');

    const adds = layerActions().filter((a) => a.type === 'addCanvasLayer');
    expect(adds).toHaveLength(1);
    const forward = adds[0];
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster' && forward.layer.source.type === 'text') {
      expect(forward.layer.source.content).toBe('Typed here');
      expect(forward.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 });
    } else {
      throw new Error('expected an addCanvasLayer with a text source');
    }
    expect(engine.stores.textEditSession.get()).toBeNull();

    dispatch.mockClear();
    await engine.history.undo();
    const removes = layerActions().filter((a) => a.type === 'removeCanvasLayers');
    expect(removes).toHaveLength(1);
    engine.lifecycle.dispose();
  });

  it('edit-mode commit dispatches ONE updateCanvasLayerSource with the exact inverse', async () => {
    const { dispatch, engine, layerActions } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()?.mode).toBe('edit');

    dispatch.mockClear();
    engine.layers.commitTextEdit('changed');

    const edits = layerActions().filter((a) => a.type === 'updateCanvasLayerSource');
    expect(edits).toHaveLength(1);
    const forward = edits[0];
    if (forward?.type === 'updateCanvasLayerSource' && forward.source.type === 'text') {
      expect(forward.id).toBe('txt1');
      expect(forward.source.content).toBe('changed');
    } else {
      throw new Error('expected an updateCanvasLayerSource text edit');
    }

    dispatch.mockClear();
    await engine.history.undo();
    const inverse = layerActions().find((a) => a.type === 'updateCanvasLayerSource');
    if (inverse?.type === 'updateCanvasLayerSource' && inverse.source.type === 'text') {
      expect(inverse.source.content).toBe('hello');
    } else {
      throw new Error('expected the inverse to restore the original content');
    }
    engine.lifecycle.dispose();
  });

  it('does not open an edit session on a locked text layer', () => {
    const { engine } = makeEngine(
      textDoc({
        stacks: stacksFrom([
          {
            blendMode: 'normal',
            id: 'txt1',
            isEnabled: true,
            isLocked: true,
            name: 'Text 1',
            opacity: 1,
            source: textSource(),
            transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
            type: 'raster',
          },
        ]),
      })
    );
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('drops the session on dispose', () => {
    const { engine } = makeEngine(paintDoc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 0, y: 0 });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    engine.lifecycle.dispose();
    expect(engine.stores.textEditSession.get()).toBeNull();
  });

  it('handleEscapePriority cancels a defocused-but-open text session (no dispatch)', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    dispatch.mockClear();

    engine.tools.handleEscapePriority({ gestureWasActive: false });
    expect(engine.stores.textEditSession.get()).toBeNull();
    expect(dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });
});

describe('canTargetLayerFromContextMenu (canvas right-click target)', () => {
  const doc = (): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom([rasterLayer('raster')]),
    selectedLayerId: 'raster',
    version: 3,
    width: 100,
  });

  const makeEngine = (document: CanvasDocumentContractV3) => {
    const { store } = createFakeStore(document);
    return createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
  };

  it('refuses while a text-edit session is open (never opens over an in-progress edit)', () => {
    const engine = makeEngine(doc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 5, y: 5 });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    expect(engine.tools.canTargetLayerFromContextMenu()).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('Select Object canvas engine integration', () => {
  const samLayer = (id: string, x = 0): CanvasRasterLayerContractV2 => ({
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { fill: '#fff', height: 100, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 100 },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 },
    type: 'raster',
  });

  const samRectLayer = (id: string, rect: CanvasDocumentContractV3['bbox']): CanvasRasterLayerContractV2 => ({
    ...samLayer(id),
    source: {
      fill: '#fff',
      height: rect.height,
      kind: 'rect',
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: rect.width,
    },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: rect.x, y: rect.y },
  });

  const samDocument = (layers: CanvasLayerContract[] = [samLayer('source')]): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom(layers),
    selectedLayerId: layers[0]?.id ?? null,
    version: 3,
    width: 100,
  });

  const createOpaqueSamBackend = () => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (width, height) => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        Object.defineProperty(surface.ctx, 'getImageData', {
          value: (_x: number, _y: number, readWidth: number, readHeight: number) => {
            const data = new Uint8ClampedArray(readWidth * readHeight * 4);
            for (let index = 0; index < data.length; index += 4) {
              data[index] = 17;
              data[index + 1] = 34;
              data[index + 2] = 51;
              data[index + 3] = 255;
            }
            return { colorSpace: 'srgb', data, height: readHeight, width: readWidth } as ImageData;
          },
        });
        return surface;
      },
    };
    return { backend, surfaces };
  };

  const withSamPreviewBitmapDimensions = (
    backend: StubRasterBackend,
    width = 100,
    height = 100
  ): StubRasterBackend => ({
    ...backend,
    createImageBitmap: async (source) => {
      const bitmap = await backend.createImageBitmap(source);
      return Object.assign(bitmap, { height, width });
    },
  });

  const createSamHarness = async (
    options: {
      backend?: StubRasterBackend;
      bbox?: CanvasDocumentContractV3['bbox'];
      imageResolver?: (imageName: string, signal?: AbortSignal) => Promise<Blob>;
      layers?: CanvasLayerContract[];
      runGraph?: NonNullable<CanvasEngineOptions['selectObjectDeps']>['runGraph'];
      uploadIntermediate?: NonNullable<CanvasEngineOptions['selectObjectDeps']>['uploadIntermediate'];
    } = {}
  ) => {
    const raf = createControllableRaf();
    const windowListeners = new Map<string, Set<(event: Event) => void>>();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('addEventListener', (type: string, listener: (event: Event) => void) => {
      const listeners = windowListeners.get(type) ?? new Set();
      listeners.add(listener);
      windowListeners.set(type, listeners);
    });
    vi.stubGlobal('removeEventListener', (type: string, listener: (event: Event) => void) => {
      windowListeners.get(type)?.delete(listener);
    });
    const bbox = options.bbox ?? { height: 100, width: 100, x: 0, y: 0 };
    const reactive = createReactiveStore({ ...samDocument(options.layers), bbox });
    const engine = createCanvasEngine({
      backend: options.backend ?? {
        ...createTestStubRasterBackend(),
        createImageBitmap: () =>
          Promise.resolve({ close: () => undefined, height: bbox.height, width: bbox.width } as unknown as ImageBitmap),
      },
      bitmapStore: createSpyBitmapStore(),
      imageResolver: options.imageResolver ?? (() => Promise.resolve(new Blob())),
      projectId: 'p1',
      selectObjectDeps: {
        runGraph:
          options.runGraph ??
          (() =>
            Promise.resolve({ height: bbox.height, imageName: 'sam-mask.png', origin: 'test', width: bbox.width })),
        uploadIntermediate:
          options.uploadIntermediate ??
          (() => Promise.resolve({ height: bbox.height, imageName: 'sam-source.png', width: bbox.width })),
      },
      store: reactive.store,
    });
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const fireKey = (type: 'keydown' | 'keyup', code: string, key: string): void => {
      for (const listener of windowListeners.get(type) ?? []) {
        listener({ code, key, preventDefault: vi.fn(), repeat: false, target: null } as unknown as Event);
      }
    };
    return { ...reactive, engine, fireKey, overlay, raf, screen };
  };

  it.each(['document', 'project'] as const)(
    'fully clears Select Object session, store, operation, and tool on %s invalidation',
    async (kind) => {
      const h = await createSamHarness();
      expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');
      const document = h.engine.document.getDocument()!;

      if (kind === 'document') {
        h.setDocument({ ...document }, 1);
      } else {
        h.setActiveProjectId('p2');
      }

      if (kind === 'project') {
        expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
        expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({ status: 'active' });
        expect(h.engine.stores.activeTool.get()).toBe('sam');
      } else {
        expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
        expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
        expect(h.engine.stores.activeTool.get()).toBe('view');
      }
      h.engine.lifecycle.dispose();
    }
  );

  const createCommittingSamHarness = async (
    imageResolver: (imageName: string, signal?: AbortSignal) => Promise<Blob> = () => Promise.resolve(new Blob()),
    backend: StubRasterBackend = createTestStubRasterBackend(),
    bbox: CanvasDocumentContractV3['bbox'] = { height: 100, width: 100, x: 0, y: 0 }
  ) => {
    const document = { ...samDocument([samRectLayer('source', bbox)]), bbox };
    const reactive = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: {
        ...backend,
        createImageBitmap: () =>
          Promise.resolve({ close: () => undefined, height: bbox.height, width: bbox.width } as unknown as ImageBitmap),
      },
      bitmapStore: createSpyBitmapStore(),
      imageResolver,
      projectId: reactive.projectId,
      selectObjectDeps: {
        runGraph: () =>
          Promise.resolve({ height: bbox.height, imageName: 'sam-mask.png', origin: 'test', width: bbox.width }),
        uploadIntermediate: () =>
          Promise.resolve({ height: bbox.height, imageName: 'sam-source.png', width: bbox.width }),
      },
      store: reactive.store,
    });
    await engine.exports.exportLayerPixels('source');
    getCanvasOperations(engine).startSelectObject('source');
    getCanvasOperations(engine).updateSelectObjectSession({
      input: {
        bbox: null,
        excludePoints: [],
        includePoints: [{ x: bbox.x + 5, y: bbox.y + 5 }],
        type: 'visual',
      },
    });
    await getCanvasOperations(engine).processSelectObjectSession();
    return { ...reactive, document, engine };
  };

  it('starts on the selected supported source, activates the real sam tool, and edits overlay-only', async () => {
    const h = await createSamHarness();

    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');
    expect(h.engine.stores.activeTool.get()).toBe('sam');
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object', layerId: 'source', projectId: 'p1' },
      status: 'active',
    });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
      layerName: 'source',
      layerType: 'raster',
      sourceRect: { height: 100, width: 100, x: 0, y: 0 },
    });
    h.raf.flush();
    h.screen.surface.callLog.length = 0;
    h.overlay.surface.callLog.length = 0;

    h.overlay.fire('pointerdown', pointerAt(5, 5));
    h.overlay.fire('pointerup', pointerAt(5, 5, { buttons: 0 }));
    h.raf.flush();

    expect(getCanvasOperations(h.engine).stores.samSession.get()?.input.includePoints).toEqual([{ x: 5, y: 5 }]);
    expect(h.overlay.surface.callLog.some((entry) => entry.op === 'arc')).toBe(true);
    expect(h.screen.surface.callLog.some((entry) => entry.op === 'clearRect' || entry.op === 'drawImage')).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('runs a transformed layer rect through upload, local visual graph, metadata, and decoded rect', async () => {
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createImageBitmap: () =>
        Promise.resolve({ close: () => undefined, height: 20, width: 30 } as unknown as ImageBitmap),
      createSurface: (width, height) => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    };
    const uploadIntermediate = vi.fn(async (source: Blob) => {
      expect(await source.text()).toBe('stub-surface-30x20');
      return { height: 20, imageName: 'layer-source.png', width: 30 };
    });
    const runGraph = vi.fn<NonNullable<CanvasEngineOptions['selectObjectDeps']>['runGraph']>((options) => {
      expect(options.graph.nodes['sam-segment']).toMatchObject({
        bounding_boxes: [{ x_max: 20, x_min: 0, y_max: 18, y_min: 3 }],
        point_lists: [{ points: [{ label: 1, x: 5, y: 8 }] }],
      });
      return Promise.resolve({ height: 20, imageName: 'mask.png', origin: 'test', width: 30 });
    });
    const imageResolver = vi.fn(() => Promise.resolve(new Blob(['mask'])));
    // A 20x20 shape scaled 1.5x horizontally and translated: baked world rect {-6, 9, 30, 20}.
    const scaled: CanvasRasterLayerContractV2 = {
      ...samLayer('scaled'),
      source: { fill: '#fff', height: 20, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 20 },
      transform: { rotation: 0, scaleX: 1.5, scaleY: 1, x: -6, y: 9 },
    };
    const h = await createSamHarness({
      backend,
      imageResolver,
      layers: [scaled, samLayer('other', 40)],
      runGraph,
      uploadIntermediate,
    });
    expect(getCanvasOperations(h.engine).startSelectObject('scaled')).toBe('started');
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.sourceRect).toEqual({
      height: 20,
      width: 30,
      x: -6,
      y: 9,
    });
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: {
        bbox: { height: 15, width: 20, x: -6, y: 12 },
        excludePoints: [],
        includePoints: [{ x: -1, y: 17 }],
        type: 'visual',
      },
    });

    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');
    h.raf.flush();

    expect(uploadIntermediate).toHaveBeenCalledOnce();
    expect(runGraph).toHaveBeenCalledOnce();
    expect(imageResolver).toHaveBeenCalledWith('mask.png', expect.any(AbortSignal));
    expect(surfaces.some((surface) => surface.width === 30 && surface.height === 20)).toBe(true);
    expect(h.overlay.surface.callLog).toContainEqual({
      args: [expect.anything(), -6, 9, 30, 20],
      op: 'drawImage',
    });
    h.engine.lifecycle.dispose();
  });

  it('saves the guarded preview to the selection mask and exits only after success', async () => {
    const { backend, surfaces } = createOpaqueSamBackend();
    const h = await createCommittingSamHarness(() => Promise.resolve(new Blob()), backend);
    const makeDurable = vi.fn(() => Promise.resolve());

    const previewCalls = surfaces.find((surface) =>
      surface.callLog.some(
        (entry) => entry.op === 'set' && entry.args[0] === 'fillStyle' && entry.args[1] === '#38bdf8'
      )
    )?.callLog;
    expect(previewCalls).toEqual(
      expect.arrayContaining([
        { args: ['globalCompositeOperation', 'source-in'], op: 'set' },
        { args: ['fillStyle', '#38bdf8'], op: 'set' },
      ])
    );

    await expect(getCanvasOperations(h.engine).saveSelectObjectSession('selection', makeDurable)).resolves.toEqual({
      status: 'selected',
    });

    expect(makeDurable).not.toHaveBeenCalled();
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(h.engine.stores.hasSelection.get()).toBe(true);
    expect(h.engine.selection.getSelectionBounds()).toEqual(h.document.bbox);
    expect(h.engine.selection.getSelectionMaskRect()).toEqual(h.document.bbox);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(h.engine.stores.activeTool.get()).toBe('view');
    h.engine.lifecycle.dispose();
  });

  it('Apply replaces the source layer content in place with one undoable history entry', async () => {
    const h = await createCommittingSamHarness();
    const makeDurable = vi.fn(() => Promise.resolve());

    await expect(getCanvasOperations(h.engine).applySelectObjectSession(makeDurable)).resolves.toEqual({
      layerId: 'source',
      status: 'committed',
    });

    expect(makeDurable).toHaveBeenCalledOnce();
    expect(makeDurable).toHaveBeenCalledWith('sam-mask.png');
    const replaced = getDocumentLeaves(h.engine.document.getDocument()!).find((layer) => layer.id === 'source')!;
    expect(replaced).toMatchObject({
      source: {
        bitmap: { height: 100, imageName: 'sam-mask.png', width: 100 },
        offset: { x: 0, y: 0 },
        type: 'paint',
      },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(getDocumentLeaves(h.engine.document.getDocument()!)).toHaveLength(1);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(h.engine.stores.activeTool.get()).toBe('view');

    expect(h.engine.stores.canUndo.get()).toBe(true);
    await h.engine.history.undo();
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    await h.engine.history.redo();
    expect(getDocumentLeaves(h.engine.document.getDocument()!).find((layer) => layer.id === 'source')).toEqual(
      replaced
    );
    h.engine.lifecycle.dispose();
  });

  it('publishes a committing phase and mutually excludes Apply and Save', async () => {
    const h = await createCommittingSamHarness();
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => durable.promise);

    expect(getCanvasOperations(h.engine).stores.samSession.get()?.status).toBe('committing');
    await expect(getCanvasOperations(h.engine).applySelectObjectSession(() => Promise.resolve())).resolves.toEqual({
      status: 'busy',
    });
    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('control', () => Promise.resolve())
    ).resolves.toEqual({
      status: 'busy',
    });
    durable.resolve();
    await expect(pending).resolves.toMatchObject({ status: 'committed' });
    h.engine.lifecycle.dispose();
  });

  it('does not report an old durability failure into a replacement session', async () => {
    const document = samDocument([samLayer('first'), samLayer('second', 30)]);
    const reactive = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: withSamPreviewBitmapDimensions(createTestStubRasterBackend()),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reactive.projectId,
      selectObjectDeps: {
        runGraph: () => Promise.resolve({ height: 100, imageName: 'sam-mask.png', origin: 'test', width: 100 }),
        uploadIntermediate: () => Promise.resolve({ height: 100, imageName: 'sam-source.png', width: 100 }),
      },
      store: reactive.store,
    });
    await engine.exports.exportLayerPixels('first');
    await engine.exports.exportLayerPixels('second');
    getCanvasOperations(engine).startSelectObject('first');
    getCanvasOperations(engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    await getCanvasOperations(engine).processSelectObjectSession();
    let rejectDurability!: (error: Error) => void;
    const durability = new Promise<void>((_resolve, reject) => {
      rejectDurability = reject;
    });
    const oldSave = getCanvasOperations(engine).saveSelectObjectSession('raster', () => durability);

    expect(getCanvasOperations(engine).startSelectObject('second')).toBe('started');
    rejectDurability(new Error('old failure'));

    await expect(oldSave).resolves.toMatchObject({ status: 'failed' });
    expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({ error: null });
    engine.lifecycle.dispose();
  });

  it('keeps the session and exact preview retryable when durability fails', async () => {
    const h = await createCommittingSamHarness();
    const previewInput = getCanvasOperations(h.engine).stores.samSession.get()?.input;

    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('raster', () =>
        Promise.reject(new Error('promotion failed'))
      )
    ).resolves.toEqual({ message: 'promotion failed', status: 'failed' });

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: { code: 'commit-failed', detail: 'promotion failed' },
      hasPreview: true,
    });
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.input).toEqual(previewInput);
    expect(h.engine.document.getDocument()).toEqual(h.document);
    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => Promise.resolve())
    ).resolves.toMatchObject({
      status: 'committed',
    });
    h.engine.lifecycle.dispose();
  });

  it.each(['source', 'project', 'document'] as const)('aborts a pending save on %s invalidation', async (kind) => {
    const h = await createSamHarness();
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    await getCanvasOperations(h.engine).processSelectObjectSession();
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('inpaint_mask', () => durable.promise);
    const document = h.engine.document.getDocument()!;

    if (kind === 'source') {
      h.setDocument({ ...document, stacks: stacksFrom([{ ...getDocumentLeaves(document)[0]!, opacity: 0.5 }]) });
    } else if (kind === 'project') {
      h.setActiveProjectId('p2');
    } else {
      h.setDocument({ ...document }, 1);
    }
    durable.resolve();

    if (kind === 'project') {
      await expect(pending).resolves.toMatchObject({ status: 'committed' });
    } else {
      await expect(pending).resolves.not.toMatchObject({ status: 'committed' });
    }
    if (kind === 'project') {
      expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
    } else {
      expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    }
    h.engine.lifecycle.dispose();
  });

  it('validates the source layer on start: missing, unsupported, disabled, and locked', async () => {
    const mask: CanvasInpaintMaskLayerContract = {
      ...samLayer('mask'),
      mask: { bitmap: null, fill: { color: '#fff', style: 'solid' } },
      type: 'inpaint_mask',
    };
    const h = await createSamHarness({
      layers: [
        mask,
        { ...samLayer('disabled'), isEnabled: false },
        { ...samLayer('locked'), isLocked: true },
        samLayer('ready'),
      ],
    });

    expect(getCanvasOperations(h.engine).startSelectObject('missing')).toBe('missing');
    expect(getCanvasOperations(h.engine).startSelectObject('mask')).toBe('unsupported');
    expect(getCanvasOperations(h.engine).startSelectObject('disabled')).toBe('disabled');
    expect(getCanvasOperations(h.engine).startSelectObject('locked')).toBe('locked');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(h.engine.stores.activeTool.get()).toBe('view');

    expect(getCanvasOperations(h.engine).startSelectObject('ready')).toBe('started');
    expect(h.engine.stores.activeTool.get()).toBe('sam');
    h.engine.lifecycle.dispose();
  });

  it('cancels the owned session and decoded preview on a real tool switch', async () => {
    const h = await createSamHarness();
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.hasPreview).toBe(true);

    h.engine.tools.setTool('view');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    h.engine.lifecycle.dispose();
  });

  it('suppresses a late preview decode after cancellation and closes its bitmap', async () => {
    const image = createDeferred<Blob>();
    const bitmap = createDeferred<ImageBitmap>();
    const close = vi.fn();
    const base = createTestStubRasterBackend();
    const backend = { ...base, createImageBitmap: vi.fn(() => bitmap.promise) };
    const h = await createSamHarness({ backend, imageResolver: () => image.promise });
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    const pending = getCanvasOperations(h.engine).processSelectObjectSession();
    await vi.waitFor(() =>
      expect(getCanvasOperations(h.engine).stores.samSession.get()?.status).toBe('rendering-preview')
    );
    image.resolve(new Blob());
    await vi.waitFor(() => expect(backend.createImageBitmap).toHaveBeenCalledOnce());

    getCanvasOperations(h.engine).cancelSelectObjectSession();
    bitmap.resolve({ close, height: 20, width: 20 } as unknown as ImageBitmap);

    await expect(pending).resolves.toBe('stale');
    expect(close).toHaveBeenCalledOnce();
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    h.engine.lifecycle.dispose();
  });

  it('isolates only the source layer in the compositor while a preview is published', async () => {
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(recordingBitmap('sam-mask', 100, 100)));
    const h = await createSamHarness({ backend, layers: [samLayer('source'), samLayer('other', 30)] });
    h.engine.stores.checkerboard.set(false);
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    await getCanvasOperations(h.engine).processSelectObjectSession();
    h.screen.surface.callLog.length = 0;
    h.raf.flush();

    expect(h.screen.surface.callLog.filter((entry) => entry.op === 'drawImage')).toHaveLength(1);
    expect(h.screen.surface.callLog).toEqual(
      expect.arrayContaining([
        { args: [0, 0, 100, 100], op: 'rect' },
        { args: [], op: 'clip' },
      ])
    );
    expect(getDocumentLeaves(h.engine.document.getDocument()!).every((layer) => layer.isEnabled)).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('owns only the latest session and suppresses in-flight work from the replaced session', async () => {
    const output = createDeferred<{ height: number; imageName: string; origin: string; width: number }>();
    const layers = [samLayer('first'), samLayer('second', 30)];
    const h = await createSamHarness({ layers, runGraph: () => output.promise });
    getCanvasOperations(h.engine).startSelectObject('first');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    const pending = getCanvasOperations(h.engine).processSelectObjectSession();
    await vi.waitFor(() =>
      expect(getCanvasOperations(h.engine).stores.samSession.get()?.status).toBe('processing-sam')
    );

    expect(getCanvasOperations(h.engine).startSelectObject('second')).toBe('started');
    expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
    output.resolve({ height: 100, imageName: 'late.png', origin: 'late', width: 100 });

    await expect(pending).resolves.toBe('stale');
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({ hasPreview: false });
    h.engine.lifecycle.dispose();
  });
});

describe('history entries and stepBy', () => {
  it('enumerates the stroke entry and replays it with clamped stepBy', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const layer = { ...rasterLayer('a'), source: { bitmap: null, type: 'paint' as const } };
    const document = { ...makeDoc(), stacks: stacksFrom([layer]), selectedLayerId: layer.id };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: [], past: [] });

    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    overlay.fire('pointermove', pointerAt(12, 12));
    overlay.fire('pointerup', pointerAt(12, 12, { buttons: 0 }));

    const afterStroke = engine.history.getEntries();
    expect(afterStroke.past).toHaveLength(1);
    expect(afterStroke.future).toHaveLength(0);

    await engine.history.stepBy(-1);
    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: afterStroke.past, past: [] });

    await engine.history.stepBy(5);
    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: [], past: afterStroke.past });

    engine.lifecycle.dispose();
  });
});

describe('pixel undo onto a cache that is no longer trustworthy', () => {
  const strokeThenInvalidate = async (source: CanvasLayerSourceContract, imageResolver: () => Promise<Blob>) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const document = paintDoc();
    const layer = { ...getDocumentLeaves(document)[0]!, source } as CanvasLayerContract;
    const { store } = createReactiveStore({ ...document, stacks: stacksFrom([layer]) });
    const base = createTestStubRasterBackend();
    const surfaces: { width: number; height: number }[] = [];
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: {
        ...base,
        createSurface: (width: number, height: number) => {
          surfaces.push({ height, width });
          return base.createSurface(width, height);
        },
      },
      bitmapStore,
      imageResolver,
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    expect(engine.history.getEntries().past).toEqual(['Brush stroke']);
    await engine.diagnostics.clearCaches();
    bitmapStore.markLayerDirty.mockClear();
    return { bitmapStore, engine, surfaces };
  };
  const persisted: CanvasLayerSourceContract = {
    bitmap: { height: 100, imageName: 'persisted', width: 100 },
    type: 'paint',
  };

  it('rasterizes the layer from its source before writing the patch', async () => {
    const imageResolver = vi.fn(() => Promise.resolve(new Blob()));
    const { engine } = await strokeThenInvalidate(persisted, imageResolver);
    const fetchesBeforeUndo = imageResolver.mock.calls.length;

    expect(await engine.history.undo()).toBe('applied');
    expect(imageResolver.mock.calls.length).toBeGreaterThan(fetchesBeforeUndo);
    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: ['Brush stroke'], past: [] });
    engine.lifecycle.dispose();
  });

  it('keeps the step in place, unpersisted, when the pixels cannot be rasterized', async () => {
    let fail = false;
    const imageResolver = vi.fn(() => (fail ? Promise.reject(new Error('offline')) : Promise.resolve(new Blob())));
    const { bitmapStore, engine } = await strokeThenInvalidate(persisted, imageResolver);
    fail = true;

    await expectReplayFailure(engine.history.undo(), /could not be restored/);
    expect(engine.history.getEntries()).toEqual({ evicted: 0, future: [], past: ['Brush stroke'] });
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });
});

describe('refused strokes', () => {
  it('reports a stroke refused before its first pixel and leaves the layer untouched', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const refusals: CanvasEditRefusal[] = [];
    engine.tools.onEditRefused((refusal) => refusals.push(refusal));
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    engine.tools.setTool('brush');
    engine.stores.documentEditingLocked.set(true);

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    expect(refusals).toEqual(['busy']);
    expect(engine.history.getEntries().past).toEqual([]);
    expect(engine.document.getDocument()).toBe(store.getState().projects[0]!.canvas.document);
    engine.lifecycle.dispose();
  });
});

describe('floating selection lifecycle', () => {
  const liftFloat = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const reactive = createReactiveStore(paintDoc());
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    engine.selection.selectAll();
    engine.tools.setTool('move');
    overlay.fire('pointerdown', pointerAt(30, 30));
    overlay.fire('pointermove', pointerAt(45, 35));
    overlay.fire('pointerup', pointerAt(45, 35, { buttons: 0 }));
    expect(engine.stores.hasFloatingSelection.get()).toBe(true);
    expect(bitmapStore.suspendLayer).toHaveBeenCalledWith('paint1');
    bitmapStore.markLayerDirty.mockClear();
    return { bitmapStore, engine, ...reactive };
  };

  it('banks a float on cooldown, so the flush that follows persists the landed pixels', async () => {
    const { bitmapStore, engine } = liftFloat();

    const cooled = engine.lifecycle.beginCooldown();

    expect(engine.stores.hasFloatingSelection.get()).toBe(false);
    expect(engine.history.getEntries().past.at(-1)).toBe('Move selection');
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledWith('paint1');
    expect(bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    await expect(cooled).resolves.toBe('cooled');
    engine.lifecycle.dispose();
  });

  it('drops a float without recording it when the document is replaced', () => {
    const { bitmapStore, engine, setDocument } = liftFloat();

    setDocument({ ...paintDoc(), height: 200, width: 200 }, 1);

    expect(engine.stores.hasFloatingSelection.get()).toBe(false);
    expect(engine.history.getEntries().past).not.toContain('Move selection');
    expect(bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    engine.lifecycle.dispose();
  });

  it('lands a live float as its own step before an accepted result goes on top of the stack', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const reducer = createReducerBackedStore(paintDoc());
    const candidate: CanvasStagingCandidateContract = {
      height: 20,
      imageName: 'staged.png',
      imageUrl: '/staged.png',
      placement: { height: 20, opacity: 1, width: 20, x: 0, y: 0 },
      queuedAt: '2026-07-16T00:00:00.000Z',
      sourceQueueItemId: 'queue',
      thumbnailUrl: '/staged-thumb.png',
      width: 20,
    };
    reducer.store.dispatch({ candidate, projectId: reducer.projectId, type: 'appendCanvasStagingCandidate' });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    engine.selection.selectAll();
    engine.tools.setTool('move');
    overlay.fire('pointerdown', pointerAt(30, 30));
    overlay.fire('pointermove', pointerAt(45, 35));
    overlay.fire('pointerup', pointerAt(45, 35, { buttons: 0 }));
    expect(engine.stores.hasFloatingSelection.get()).toBe(true);

    expect(engine.layers.commitStagedImage({ candidate, selectedImageIndex: 0 }).status).toBe('committed');

    expect(engine.stores.hasFloatingSelection.get()).toBe(false);
    expect(engine.history.getEntries().past.slice(-2)).toEqual(['Move selection', 'Accept staged image']);
    const { document } = reducer.store.getState().projects[0]!.canvas;
    expect(document.stacks.raster[0]).toMatchObject({ source: { image: { imageName: 'staged.png' } } });
    engine.lifecycle.dispose();
  });
});

describe('layer quick-switch memory', () => {
  it('remembers the previously selected layer across selection changes, keeping it through a deselect', () => {
    const document = { ...makeDoc(), stacks: stacksFrom([rasterLayer('a'), rasterLayer('b')]), selectedLayerId: 'a' };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect(engine.interaction.get('layerQuickSwitch')).toEqual({ bookmarkedLayerId: null, previousLayerId: null });

    setDocument({ ...document, selectedLayerId: 'b' });
    expect(engine.interaction.get('layerQuickSwitch').previousLayerId).toBe('a');
    setDocument({ ...document, selectedLayerId: null });
    expect(engine.interaction.get('layerQuickSwitch').previousLayerId).toBe('a');
    setDocument({ ...document, selectedLayerId: 'b' });
    expect(engine.interaction.get('layerQuickSwitch').previousLayerId).toBe('a');
    setDocument({ ...document, selectedLayerId: null });
    setDocument({ ...document, selectedLayerId: 'a' });
    expect(engine.interaction.get('layerQuickSwitch').previousLayerId).toBe('b');

    engine.interaction.set('layerQuickSwitch', {
      ...engine.interaction.get('layerQuickSwitch'),
      bookmarkedLayerId: 'b',
    });
    setDocument({ ...document, selectedLayerId: 'b' });
    expect(engine.interaction.get('layerQuickSwitch')).toEqual({ bookmarkedLayerId: 'b', previousLayerId: 'a' });
    expect(engine.history.getEntries().past).toEqual([]);
    engine.lifecycle.dispose();
  });
});

describe('document epoch', () => {
  it('advances on each document change, once the engine model already reflects it', () => {
    const { setDocument, store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const seen: (number | undefined)[] = [];
    engine.interaction.subscribe('documentEpoch', () =>
      seen.push(
        engine.document
          .model()
          ?.compileLeaves()
          .find((leaf) => leaf.id === 'paint1')?.layer.opacity
      )
    );
    const before = engine.interaction.get('documentEpoch');
    const dimmed = paintDoc();
    const [layer] = getDocumentLeaves(dimmed);

    setDocument({ ...dimmed, stacks: stacksFrom([{ ...layer!, opacity: 0.25 }]) });

    expect(engine.interaction.get('documentEpoch')).toBe(before + 1);
    expect(seen).toEqual([0.25]);
    engine.lifecycle.dispose();
  });
});

describe('color picker loupe', () => {
  const attachPicker = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const { store } = createReactiveStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    const frame = (): { drawn: boolean; loupe: { x: number; y: number } | null; rings: number } => {
      overlay.surface.callLog.length = 0;
      raf.flush();
      const arcs = overlay.surface.callLog.filter((op) => op.op === 'arc');
      const loupe = arcs.find((op) => op.args[2] === 60);
      return {
        drawn: overlay.surface.callLog.length > 0,
        loupe: loupe ? { x: loupe.args[0] as number, y: loupe.args[1] as number } : null,
        rings: arcs.filter((op) => op.args[2] !== 60).length,
      };
    };
    return { engine, frame, overlay, screen };
  };

  it('follows the hovering pointer and leaves the canvas with it', () => {
    const { engine, frame, overlay } = attachPicker();
    engine.tools.setTool('colorPicker');
    overlay.fire('pointerenter', pointerAt(20, 20, { buttons: 0 }));
    overlay.fire('pointermove', pointerAt(20, 20, { buttons: 0 }));
    expect(frame().loupe).toEqual({ x: 20, y: 20 });

    overlay.fire('pointerleave', pointerAt(120, 20, { buttons: 0 }));

    expect(frame()).toMatchObject({ drawn: true, loupe: null });
    engine.lifecycle.dispose();
  });

  it('removes on detach every listener attach added, across re-attaching', () => {
    const { engine, overlay, screen } = attachPicker();

    engine.surface.detach();
    engine.surface.attach(screen.element, overlay.element);
    engine.surface.detach();

    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointerleave', 'pointerenter', 'wheel']) {
      expect(overlay.listenerCount(type), type).toBe(0);
    }
    engine.lifecycle.dispose();
  });
});

describe('control derivation composition', () => {
  it('commits control-layer patches without the operation lease, so a running operation refuses them', async () => {
    const control: CanvasControlLayerContract = {
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'normal',
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control Layer 1',
      opacity: 1,
      source: { image: { height: 10, imageName: 'photo', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const { store } = createReactiveStore({ ...makeDoc(), stacks: stacksFrom([control]) });
    const { applicationHost, engine } = createCoreCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createTestMutationPort(store, 'p1'),
      projectId: 'p1',
      reportError: () => undefined,
      uploadImage: () => Promise.resolve({ height: 1, imageName: 'durable', width: 1 }),
      uploadIntermediateImage: () => Promise.resolve({ height: 1, imageName: 'transient', width: 1 }),
    });
    const patch = { expected: control, label: 'Set control type', next: { ...control, controlType: 'depth' as const } };

    engine.stores.documentEditingLocked.set(true);
    expect(applicationHost.commitControlLayerPatch(patch)).toEqual({ status: 'busy' });
    engine.stores.documentEditingLocked.set(false);
    expect(applicationHost.commitControlLayerPatch(patch)).toEqual({ status: 'committed' });
    expect(applicationHost.captureControlDetectionGuard('control')).toMatchObject({
      documentRevision: 0,
      layerId: 'control',
      projectId: 'p1',
    });
    await expect(applicationHost.flushLayerUploads('control', {})).resolves.toBe('clean');
    engine.lifecycle.dispose();
  });
});

describe('edit feedback on the surface', () => {
  const attachEngine = (document: CanvasDocumentContractV3) => {
    vi.stubGlobal(
      'Path2D',
      class FakePath2D {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const windowListeners = new Map<string, Set<(event: unknown) => void>>();
    vi.stubGlobal('addEventListener', (type: string, listener: (event: unknown) => void) =>
      windowListeners.set(type, (windowListeners.get(type) ?? new Set()).add(listener))
    );
    vi.stubGlobal('removeEventListener', (type: string, listener: (event: unknown) => void) =>
      windowListeners.get(type)?.delete(listener)
    );
    const { dispatch, projectId, store } = createReducerBackedStore(document);
    const stubBackend = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const engine = createCanvasEngine({
      backend: {
        ...stubBackend,
        createSurface: (width: number, height: number) => {
          const surface = stubBackend.createSurface(width, height);
          surfaces.push(surface);
          return surface;
        },
      },
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    const refusals: CanvasEditRefusal[] = [];
    engine.tools.onEditRefused((refusal) => refusals.push(refusal));
    const key = (type: 'keydown' | 'keyup', code: string) =>
      windowListeners
        .get(type)
        ?.forEach((listener) => listener({ code, key: code, preventDefault: () => {}, target: null, type }));
    return { dispatch, engine, key, overlay, refusals, surfaces };
  };

  it('says why a stroke on a locked layer did nothing, and shows not-allowed until the layer is unlocked', () => {
    const locked: CanvasLayerContract = {
      ...(rasterLayer('paint') as CanvasRasterLayerContractV2),
      isLocked: true,
      name: 'Sky',
      source: { bitmap: null, type: 'paint' },
    };
    const h = attachEngine({ ...makeDoc(), selectedLayerId: 'paint', stacks: stacksFrom([locked]) });
    h.engine.tools.setTool('brush');
    expect(h.engine.stores.cursor.get()).toBe('not-allowed');

    h.overlay.fire('pointerenter', pointerAt(20, 20));
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointerup', pointerAt(20, 20, { buttons: 0 }));
    expect(h.refusals).toEqual([{ layerId: 'paint', layerName: 'Sky', status: 'layer-locked' }]);
    expect(getDocumentLeaves(h.engine.document.getDocument())).toHaveLength(1);

    h.dispatch({ id: 'paint', patch: { isLocked: false }, type: 'updateCanvasLayer' });
    expect(h.engine.stores.cursor.get()).toBe('crosshair');
    h.engine.lifecycle.dispose();
  });

  describe('shape tool through the pointer pipeline', () => {
    const paintDoc = (): CanvasDocumentContractV3 => ({
      ...makeDoc(),
      selectedLayerId: 'paint',
      stacks: stacksFrom([
        { ...(rasterLayer('paint') as CanvasRasterLayerContractV2), source: { bitmap: null, type: 'paint' as const } },
      ]),
    });
    const maskDoc = (): CanvasDocumentContractV3 => ({
      ...makeDoc(),
      selectedLayerId: 'mask',
      stacks: stacksFrom([
        {
          blendMode: 'normal',
          id: 'mask',
          isEnabled: true,
          isLocked: false,
          mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
          name: 'Mask',
          opacity: 1,
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
          type: 'inpaint_mask',
        },
      ]),
    });
    const at = (x: number, y: number, opts: { buttons?: number; mod?: boolean; timeStamp?: number } = {}) => ({
      ...pointerAt(x, y, { buttons: opts.buttons }),
      ctrlKey: opts.mod ?? false,
      metaKey: opts.mod ?? false,
      timeStamp: opts.timeStamp ?? 0,
    });
    const shapeEngine = (document: CanvasDocumentContractV3, kind: 'rect' | 'polygon') => {
      const h = attachEngine(document);
      const strokes: StrokeCommittedEvent[] = [];
      h.engine.tools.onStrokeCommitted((event) => strokes.push(event));
      h.engine.tools.setTool('shape');
      h.engine.stores.shapeOptions.set({ ...h.engine.stores.shapeOptions.get(), kind });
      h.overlay.fire('pointerenter', at(0, 0));
      const click = (x: number, y: number, opts: { mod?: boolean; timeStamp?: number } = {}) => {
        h.overlay.fire('pointerdown', at(x, y, opts));
        h.overlay.fire('pointerup', at(x, y, { ...opts, buttons: 0 }));
      };
      const drag = (from: number, to: number, opts: { mod?: boolean } = {}) => {
        h.overlay.fire('pointerdown', at(from, from));
        h.overlay.fire('pointermove', at(to, to));
        h.overlay.fire('pointerup', at(to, to, { ...opts, buttons: 0 }));
      };
      const placeTriangle = (start: number) => {
        click(10, 10, { timeStamp: start });
        click(50, 10, { timeStamp: start + 1000 });
        click(50, 40, { timeStamp: start + 2000 });
      };
      const leaves = () => getDocumentLeaves(h.engine.document.getDocument());
      return { ...h, click, drag, leaves, placeTriangle, strokes };
    };

    it('adds a polygon closed by a first-vertex click or a double-click as a new layer, one undo step each', async () => {
      const h = shapeEngine({ ...makeDoc(), stacks: stacksFrom([]) }, 'polygon');
      h.engine.stores.shapeOptions.set({ ...h.engine.stores.shapeOptions.get(), target: 'new' });

      h.placeTriangle(0);
      h.click(11, 11, { timeStamp: 3000 });
      expect(h.leaves()).toEqual([
        expect.objectContaining({ source: expect.objectContaining({ height: 30, kind: 'polygon', width: 40 }) }),
      ]);

      h.placeTriangle(4000);
      h.click(50, 40, { timeStamp: 6100 });
      expect(h.leaves()).toHaveLength(2);
      expect(h.refusals).toEqual([]);

      await h.engine.history.undo();
      await h.engine.history.undo();
      expect(h.leaves()).toHaveLength(0);
      h.engine.lifecycle.dispose();
    });

    it.each([
      ['paint layer', paintDoc, 'paint'],
      ['mask', maskDoc, 'mask'],
    ] as const)(
      'cuts a shape out of the selected %s with the mod key, and undo puts the pixels back',
      async (_l, doc, id) => {
        const h = shapeEngine(doc(), 'rect');
        h.drag(10, 60);
        h.drag(20, 40, { mod: true });

        expect(h.leaves()).toHaveLength(1);
        expect(h.strokes.map((stroke) => [stroke.layerId, stroke.tool])).toEqual([
          [id, 'shape'],
          [id, 'shape-cutout'],
        ]);
        const cutout = h.strokes[1]!;
        expect(cutout.dirtyRect).toEqual({ height: 20, width: 20, x: 20, y: 20 });

        await h.engine.history.undo();
        expect(h.engine.stores.canRedo.get()).toBe(true);
        expect(putImageDataCalls(h.surfaces)).toContainEqual({ image: cutout.beforeImageData, x: 10, y: 10 });
        expect(h.leaves()).toHaveLength(1);
        h.engine.lifecycle.dispose();
      }
    );

    describe('Escape', () => {
      const polygonEngine = (tool: 'shape' | 'lasso') => {
        const h = shapeEngine({ ...makeDoc(), stacks: stacksFrom([]) }, 'polygon');
        h.engine.stores.lassoOptions.set({ ...h.engine.stores.lassoOptions.get(), shape: 'polygon' });
        h.engine.tools.setTool(tool);
        h.engine.selection.selectAll();
        const preview = () => h.engine.stores.lassoPreview.get();
        return { ...h, preview };
      };

      it.each(['shape', 'lasso'] as const)(
        'drops a %s polygon between clicks, keeping the selection and adding nothing',
        (tool) => {
          const h = polygonEngine(tool);
          h.placeTriangle(0);
          expect(h.preview()).toMatchObject({ points: [{}, {}, {}] });

          h.key('keydown', 'Escape');
          expect(h.preview()).toBeNull();
          expect(h.engine.stores.hasSelection.get()).toBe(true);
          expect(h.leaves()).toHaveLength(0);

          h.click(70, 70, { timeStamp: 5000 });
          expect(h.preview()).toMatchObject({ points: [{ x: 70, y: 70 }] });
          h.engine.lifecycle.dispose();
        }
      );
    });
  });
});

it('reactivates Spline Edit on the same selected vector after engine cooldown', async () => {
  const vector = {
    ...createVectorLayer({ id: 'v', name: 'Vector' }),
    transform: { x: 0, y: 0, rotation: 0, scaleX: 1, scaleY: 1 },
    paths: [],
  };
  const doc = { ...makeDoc(), stacks: stacksFrom([vector]), selectedLayerId: 'v' };
  const { store } = createFakeStore(doc);
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    projectId: 'p1',
    store,
  });
  try {
    engine.tools.setTool('spline');
    expect(engine.interaction.get('vectorEditSession')?.layerId).toBe('v');
    await engine.lifecycle.beginCooldown();
    expect(engine.interaction.get('vectorEditSession')).toBeNull();
    engine.lifecycle.activate();
    expect(engine.interaction.get('activeTool')).toBe('spline');
    expect(engine.interaction.get('vectorEditSession')).toMatchObject({
      layerId: 'v',
      activePathId: null,
      selectedPoints: [],
    });
  } finally {
    engine.lifecycle.dispose();
  }
});

it('aborts pending vector region work on Escape and document changes without creating raster output', async () => {
  const vector = {
    ...createVectorLayer({ id: 'v', name: 'Vector' }),
    fillEnabled: true,
    strokeEnabled: false,
    transform: { x: 0, y: 0, rotation: 0, scaleX: 1, scaleY: 1 },
    paths: [
      {
        id: 'p',
        name: null,
        isClosed: true,
        points: [
          { x: 0, y: 0 },
          { x: 100, y: 0 },
          { x: 0, y: 100 },
        ].map((anchor) => ({ anchor, inHandle: null, outHandle: null, type: 'corner' as const })),
      },
    ],
  };
  const doc = { ...makeDoc(), stacks: stacksFrom([vector]), selectedLayerId: 'v' };
  const h = createReactiveStore(doc);
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    projectId: 'p1',
    store: h.store,
  });
  try {
    engine.tools.setTool('spline');
    const first = engine.layers.traceFillVectorLayer('v');
    engine.tools.handleEscapePriority({ gestureWasActive: false });
    expect(await first).toEqual({ status: 'aborted' });
    const second = engine.layers.traceFillVectorLayer('v');
    h.setDocument({ ...doc, selectedLayerId: null });
    expect(await second).toEqual({ status: 'aborted' });
    expect(h.store.dispatch).not.toHaveBeenCalled();
  } finally {
    engine.lifecycle.dispose();
  }
});

it('keeps pending vector rendering independent of brush color and size changes', async () => {
  vi.stubGlobal(
    'Path2D',
    class {
      moveTo() {}
      lineTo() {}
      closePath() {}
      quadraticCurveTo() {}
      bezierCurveTo() {}
    }
  );
  const vector = {
    ...createVectorLayer({ id: 'v', name: 'Vector' }),
    fillEnabled: true,
    fillColor: '#00ff00',
    strokeColor: '#ff0000',
    strokeWidth: 8,
    paths: [
      {
        id: 'p',
        name: null,
        isClosed: true,
        points: [
          { x: 0, y: 0 },
          { x: 100, y: 0 },
          { x: 100, y: 100 },
        ].map((anchor) => ({ anchor, inHandle: null, outHandle: null, type: 'corner' as const })),
      },
    ],
  };
  const h = createReducerBackedStore({ ...makeDoc(), stacks: stacksFrom([vector]), selectedLayerId: 'v' });
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    projectId: h.projectId,
    store: h.store,
  });
  try {
    const pending = engine.layers.traceFillVectorLayer('v');
    engine.interaction.set('brushOptions', { ...engine.interaction.get('brushOptions'), color: '#0000ff', size: 99 });
    expect(await pending).toEqual({ status: 'committed' });
    expect(getDocumentLayer(h.store.getState().projects[0]!.canvas.document, 'v')).toMatchObject({
      strokeColor: '#ff0000',
      strokeWidth: 8,
      fillColor: '#00ff00',
    });
  } finally {
    engine.lifecycle.dispose();
  }
});

it('selects the first vector path immediately after engine creation', () => {
  const vector = createVectorLayer({ id: 'v', name: 'Vector' });
  vector.paths = [
    {
      id: 'p',
      name: null,
      isClosed: false,
      points: [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 100, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
      ],
    },
  ];
  const { store } = createFakeStore({ ...makeDoc(), stacks: stacksFrom([vector]), selectedLayerId: 'v' });
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    projectId: 'p1',
    store,
  });
  try {
    engine.tools.setTool('spline');
    expect(engine.interaction.get('activeTool')).toBe('spline');
    expect(engine.layers.vectorEdit.targetContextMenu({ x: 50, y: 0 })).toBe(true);
    expect(engine.interaction.get('vectorEditSession')?.activePathId).toBe('p');
  } finally {
    engine.lifecycle.dispose();
  }
});
