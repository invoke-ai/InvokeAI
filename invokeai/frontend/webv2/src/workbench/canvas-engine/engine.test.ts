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
import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
  CanvasStateContractV2,
  Project,
  WorkbenchState,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import {
  applyCanvasProjectMutation,
  isCanvasProjectMutation,
  type CanvasProjectMutation,
} from '@workbench/canvasProjectMutations';

type EngineTestAction = WorkbenchAction | CanvasProjectMutation;

import { DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/render/compositor';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { canvasApplicationPort } from '@workbench/canvas-operations/applicationPort';
import {
  createCanvasEngine as createApplicationCanvasEngine,
  getCanvasOperations,
  type CanvasEngine,
  type CanvasEngineOptions,
} from '@workbench/canvas-operations/createCanvasEngine';
import { FILTER_AUTO_PROCESS_DEBOUNCE_MS } from '@workbench/canvas-operations/filterOperationSession';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState';
import { afterEach, describe, expect, it, type Mock, vi } from 'vitest';

import type { BitmapStore } from './document/bitmapStore';
import type { StrokeCommittedEvent } from './tools/tool';

import { createBitmapStore } from './document/bitmapStore';
import { mergeDownMatrix } from './document/mergeDown';

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

type TestCanvasEngineOptions = Omit<CanvasEngineOptions, 'getMainModelBase' | 'mutationPort' | 'reportError'> & {
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
    ...options,
    getMainModelBase:
      getMainModelBase ?? (() => canvasApplicationPort.getSelectedModelBase(store.getState(), projectId)),
    mutationPort: mutationPort ?? createTestMutationPort(store, projectId),
    projectId,
    reportError: reportError ?? (() => undefined),
  });

// Records adjusted-surface cache access without exposing it on the engine. The
// factory wraps the real implementation, preserving all behaviour.
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
    createAdjustedSurfaceCache: (backend: Parameters<typeof actual.createAdjustedSurfaceCache>[0]) => {
      const cache = actual.createAdjustedSurfaceCache(backend);
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

const makeDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [rasterLayer('a')],
  selectedLayerId: null,
  version: 2,
  width: 100,
});

const makeCanvas = (document: CanvasDocumentContractV2, documentRevision = 0): CanvasStateContractV2 => ({
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
  version: 2,
});

const createFakeStore = (
  document: CanvasDocumentContractV2
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

// ---- Reactive store + render-loop harness -----------------------------
//
// The tests above never `attach()`, so the render loop (and thus
// `ensureLayerCaches`) never runs. The tests below exercise the document
// mirror wired into a live engine, so they need: a store that actually
// notifies subscribers on change, a controllable `requestAnimationFrame`
// pair to drive the (otherwise real) scheduler deterministically, and fake
// canvases whose `getContext('2d')` returns a recording stub context.

/** A reactive fake store: `setDocument` notifies subscribers, unlike `createFakeStore` above. */
const createReactiveStore = (
  document: CanvasDocumentContractV2
): {
  listenerCount: () => number;
  setActiveProjectId: (projectId: string) => void;
  setDocument: (next: CanvasDocumentContractV2, documentRevision?: number) => void;
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

/** A controllable `requestAnimationFrame`/`cancelAnimationFrame` pair for driving the scheduler. */
const createControllableRaf = () => {
  let nextHandle = 1;
  const callbacks = new Map<number, FrameRequestCallback>();
  return {
    cancelFrame: (handle: number): void => {
      callbacks.delete(handle);
    },
    /** Runs every currently-queued frame callback, like the browser firing a frame. */
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

/** A minimal fake `HTMLCanvasElement`: a recording 2D context, no-op listeners. */
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

/** A deferred promise, so a test can resolve a rasterize step on demand. */
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

/**
 * Stateful raster backend for publication-race tests. Every scratch/cache
 * surface gets a stable id, as do the fake bitmaps supplied by each test. This
 * lets a test distinguish a decode drawn into an isolated scratch surface from
 * a scratch surface published into the live cache without exposing cache
 * internals from the production engine.
 */
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

/** One-shot allocation/draw faults, armed after a deterministic number of successful calls. */
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

const createInvalidateDuringBitmapDrawBackend = (bitmap: ImageBitmap, onDraw: () => void): StubRasterBackend => {
  const backend = createTestStubRasterBackend();
  let invalidated = false;
  return {
    ...backend,
    createImageBitmap: vi.fn(() => Promise.resolve(bitmap)),
    createSurface: (width, height) => {
      const surface = backend.createSurface(width, height);
      const ctx = new Proxy(surface.ctx, {
        get: (target, property, receiver) => {
          const value = Reflect.get(target, property, receiver);
          if (property !== 'drawImage' || typeof value !== 'function') {
            return value;
          }
          return (...args: unknown[]) => {
            const result = Reflect.apply(value, target, args);
            if (!invalidated && args[0] === bitmap) {
              invalidated = true;
              onDraw();
            }
            return result;
          };
        },
      });
      Object.defineProperty(surface, 'ctx', { value: ctx });
      return surface;
    },
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

/** Recursively follows recording-surface draw edges to a bitmap/surface id. */
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

/** Flushes pending microtasks (promise chains) without depending on fake timers. */
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
      layers: [{ ...document.layers[0]!, opacity: 0.5 }],
    });
    pending.resolve(new Blob(['pixels']));

    await expect(capture).resolves.toEqual({ status: 'stale' });
    engine.lifecycle.dispose();
  });

  it('returns stale when direct paint changes cache pixels after the document snapshot', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const layer = {
      ...rasterLayer('a'),
      source: { bitmap: { height: 20, imageName: 'paint-pixels', width: 20 }, type: 'paint' as const },
    };
    const document = { ...makeDoc(), layers: [layer], selectedLayerId: layer.id };
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
    const initial = await engine.exports.exportLayerPixels(layer.id);
    expect(initial.status).toBe('ok');
    if (initial.status !== 'ok') {
      throw new Error('Expected initial pixels');
    }
    const initialGuard = initial.guard;
    initial.release();
    const documentSnapshot = engine.document.captureSnapshot();
    const capture = engine.exports.captureRasterSnapshot(documentSnapshot!, [layer.id]);

    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    overlay.fire('pointermove', pointerAt(10, 10));
    overlay.fire('pointerup', pointerAt(10, 10, { buttons: 0 }));

    expect(engine.exports.isLayerExportGuardCurrent(initialGuard)).toBe(false);
    await expect(capture).resolves.toEqual({ status: 'stale' });
    engine.lifecycle.dispose();
  });

  it('returns stale when cooldown changes lifecycle generation during raster detachment', async () => {
    const pending = createDeferred<Blob>();
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });
    const documentSnapshot = engine.document.captureSnapshot();
    const capture = engine.exports.captureRasterSnapshot(documentSnapshot!, ['a']);

    const cooldown = engine.lifecycle.beginCooldown();
    pending.resolve(new Blob(['pixels']));

    await expect(capture).resolves.toEqual({ status: 'stale' });
    await cooldown;
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

  it('releases snapshot reservations and pins promptly after abort so a large retry can proceed', async () => {
    const imageSize = 5_300;
    const largeLayer: CanvasRasterLayerContractV2 = {
      blendMode: 'normal',
      id: 'large',
      isEnabled: true,
      isLocked: false,
      name: 'large',
      opacity: 1,
      source: {
        image: { height: imageSize, imageName: 'large', width: imageSize },
        type: 'image',
      },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const document: CanvasDocumentContractV2 = {
      ...makeDoc(),
      height: imageSize,
      layers: [largeLayer],
      width: imageSize,
    };
    const { requests, resolver } = createAbortableImageResolver();
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    const controller = new AbortController();
    const firstSnapshot = engine.document.captureSnapshot();
    const firstCapture = engine.exports.captureRasterSnapshot(firstSnapshot!, ['large'], {
      signal: controller.signal,
    });
    await drainMicrotasksUntil(() => requests.has('large'));

    controller.abort(new DOMException('invoke cancelled', 'AbortError'));
    await expect(firstCapture).resolves.toEqual({ status: 'aborted' });

    requests.delete('large');
    const retrySnapshot = engine.document.captureSnapshot();
    const retryCapture = engine.exports.captureRasterSnapshot(retrySnapshot!, ['large']);
    await drainMicrotasksUntil(() => requests.has('large'));
    requests.get('large')?.deferred.resolve(new Blob(['retry pixels']));

    const retry = await retryCapture;
    expect(retry.status).toBe('ok');
    if (retry.status === 'ok') {
      retry.snapshot.release();
    }
    engine.lifecycle.dispose();
  });

  it('returns typed not-ready when asynchronous raster cache preparation fails', async () => {
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.reject(new Error('decode unavailable')),
      projectId: 'p1',
      reportError: vi.fn(),
      store,
    });
    const documentSnapshot = engine.document.captureSnapshot();

    await expect(engine.exports.captureRasterSnapshot(documentSnapshot!, ['a'])).resolves.toEqual({
      status: 'not-ready',
    });
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

  it('captures valid pixels while identifying a genuinely empty paint layer for callers to skip', async () => {
    const blank = { ...rasterLayer('blank'), source: { bitmap: null, type: 'paint' } as const };
    const { store } = createFakeStore({ ...makeDoc(), layers: [rasterLayer('valid'), blank] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob(['valid'])),
      projectId: 'p1',
      store,
    });
    const documentSnapshot = engine.document.captureSnapshot();

    const result = await engine.exports.captureRasterSnapshot(documentSnapshot!, ['valid', 'blank']);

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('Expected mixed raster snapshot');
    }
    expect([...result.snapshot.layerSurfaces.keys()]).toEqual(['valid']);
    expect([...result.snapshot.emptyLayerIds]).toEqual(['blank']);
    result.snapshot.release();
    engine.lifecycle.dispose();
  });

  it('returns a successful empty raster snapshot for a blank-only document', async () => {
    const blank = { ...rasterLayer('blank'), source: { bitmap: null, type: 'paint' } as const };
    const { store } = createFakeStore({ ...makeDoc(), layers: [blank] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const documentSnapshot = engine.document.captureSnapshot();

    const result = await engine.exports.captureRasterSnapshot(documentSnapshot!, ['blank']);

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('Expected empty raster snapshot');
    }
    expect(result.snapshot.layerSurfaces.size).toBe(0);
    expect([...result.snapshot.emptyLayerIds]).toEqual(['blank']);
    result.snapshot.release();
    engine.lifecycle.dispose();
  });

  it('keeps caller-owned raster snapshot surfaces alive across cooldown', async () => {
    const { engine } = createEngine();
    const documentSnapshot = engine.document.captureSnapshot();
    const result = await engine.exports.captureRasterSnapshot(documentSnapshot!, ['a']);
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('raster snapshot was not captured');
    }

    await engine.lifecycle.beginCooldown();

    expect(result.snapshot.layerSurfaces.size).toBe(1);
    result.snapshot.release();
    expect(result.snapshot.layerSurfaces.size).toBe(0);
    engine.lifecycle.dispose();
  });

  it('refuses a raster snapshot using the actual live surface bytes before cloning', async () => {
    const { engine } = createEngine();
    const live = await engine.exports.exportLayerPixels('a');
    expect(live.status).toBe('ok');
    if (live.status !== 'ok') {
      throw new Error('layer cache was not rasterized');
    }
    // Model a live paint cache that has grown beyond its persisted source bounds.
    // The stub resize records dimensions without allocating a real 276 MiB buffer.
    // One live surface fits the 512 MiB limit; cloning a second one does not.
    live.surface.resize(8_500, 8_500);
    const documentSnapshot = engine.document.captureSnapshot();

    await expect(engine.exports.captureRasterSnapshot(documentSnapshot!, ['a'])).resolves.toEqual({
      status: 'over-budget',
    });

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
  it('mirrors the reducer-owned document on creation', () => {
    const { doc, engine } = createEngine();
    expect(engine.document.getDocument()).toBe(doc);
    engine.lifecycle.dispose();
  });

  it('exposes an initial viewport and stores', () => {
    const { engine } = createEngine();
    expect(engine.viewport.getViewport().getZoom()).toBe(1);
    expect(engine.stores.activeTool.get()).toBe('view');
    expect(engine.stores.viewportReady.get()).toBe(false);
    expect(engine.surface.attach).toBe(engine.surface.attach);
    expect(engine.viewport.getViewport).toBe(engine.viewport.getViewport);
    expect(engine.tools.setTool).toBeTypeOf('function');
    expect(engine.history.undo).toBe(engine.history.undo);
    expect(engine.lifecycle.beginCooldown).toBe(engine.lifecycle.beginCooldown);
    expect(engine.edits.tryAcquire).toBeTypeOf('function');
    engine.lifecycle.dispose();
  });

  it('exposes grouped capabilities without the deprecated flat facade', () => {
    const { engine } = createEngine();
    const deprecatedFlatMethods = [
      'attach',
      'beginCooldown',
      'commitStructural',
      'dispose',
      'exportLayerPixels',
      'fitToView',
      'getDocument',
      'processFilterOperation',
      'setTool',
      'startSelectObject',
      'undo',
    ] as const;

    for (const method of deprecatedFlatMethods) {
      expect(method in engine).toBe(false);
    }
    expect(engine.surface.attach).toBeTypeOf('function');
    expect(engine.layers.commitStructural).toBeTypeOf('function');
    expect(getCanvasOperations(engine).startSelectObject).toBeTypeOf('function');
    expect(engine.exports.exportLayerPixels).toBeTypeOf('function');
    engine.lifecycle.dispose();
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

  it('retries persistence after a dirty cooldown result', async () => {
    const doc = makeDoc();
    const { store } = createFakeStore(doc);
    const bitmapStore = createSpyBitmapStore();
    vi.mocked(bitmapStore.flushPendingUploads)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(undefined);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    await expect(engine.lifecycle.beginCooldown()).resolves.toBe('dirty');
    await expect(engine.lifecycle.beginCooldown()).resolves.toBe('cooled');

    expect(bitmapStore.flushPendingUploads).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('keeps an in-flight invocation reservation accounted through cooldown', async () => {
    const { engine } = createEngine();
    const deps = engine.exports.getCompositeExecutorDeps();
    const reservation = deps.reserve?.(300 * 1024 * 1024);
    expect(reservation?.status).toBe('ok');

    await engine.lifecycle.beginCooldown();

    const competing = deps.reserve?.(300 * 1024 * 1024);
    expect(competing?.status).toBe('over-budget');

    if (reservation?.status === 'ok') {
      reservation.lease.release();
    }
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

  it('exportLayerPixels refuses hidden layers unless includeDisabled is set', async () => {
    const hidden = { ...rasterLayer('hidden'), isEnabled: false };
    const { store } = createFakeStore({ ...makeDoc(), layers: [hidden] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.exports.exportLayerPixels('hidden')).toEqual({ status: 'disabled' });
    expect((await engine.exports.exportLayerPixels('hidden', { includeDisabled: true })).status).toBe('ok');
    engine.lifecycle.dispose();
  });

  it('exportLayerPixels can bake raster adjustments in layer-local space without baking presentation props', async () => {
    const layer: CanvasRasterLayerContractV2 = {
      ...(rasterLayer('a') as CanvasRasterLayerContractV2),
      adjustments: { brightness: 0.25, contrast: 0, saturation: 0 },
      blendMode: 'multiply',
      opacity: 0.4,
      source: {
        bitmap: { height: 10, imageName: 'local-paint', width: 10 },
        offset: { x: -4, y: 7 },
        type: 'paint',
      },
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 5, y: 6 },
    };
    const { store } = createFakeStore({ ...makeDoc(), layers: [layer] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    const raw = await engine.exports.exportLayerPixels('a');
    const adjusted = await engine.exports.exportLayerPixels('a', { applyAdjustments: true });

    expect(raw.status).toBe('ok');
    expect(adjusted.status).toBe('ok');
    if (raw.status === 'ok' && adjusted.status === 'ok') {
      expect(adjusted.rect).toEqual({ height: 10, width: 10, x: -4, y: 7 });
      expect(adjusted.surface).not.toBe(raw.surface);
      expect(adjusted.surface).toMatchObject({ height: 10, width: 10 });
      expect(engine.exports.isLayerExportGuardCurrent(adjusted.guard)).toBe(true);
      expect((raw.surface as StubRasterSurface).callLog.some((entry) => entry.op === 'putImageData')).toBe(false);
      expect((adjusted.surface as StubRasterSurface).callLog).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ op: 'drawImage' }),
          { args: [0, 0, 10, 10], op: 'getImageData' },
          expect.objectContaining({ op: 'putImageData' }),
        ])
      );
      expect(
        (adjusted.surface as StubRasterSurface).callLog.some(
          (entry) =>
            entry.op === 'set' && (entry.args[0] === 'globalAlpha' || entry.args[0] === 'globalCompositeOperation')
        )
      ).toBe(false);
    }
    engine.lifecycle.dispose();
  });

  it('exportLayerPixels shares the scheduled rasterization already running for the same source', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const pendingResolve = createDeferred<Blob>();
    const imageResolver = vi.fn(() => pendingResolve.promise);
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver,
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();

    engine.surface.attach(screen.element, overlay.element);
    raf.flush();

    const exported = engine.exports.exportLayerPixels('a');
    expect(imageResolver).toHaveBeenCalledTimes(1);

    pendingResolve.resolve(new Blob());
    expect((await exported).status).toBe('ok');
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
    const document = { ...makeDoc(), layers: [rasterLayer('L', { imageName: 'A' })] };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({ backend, imageResolver, projectId: 'p1', store });

    const exportA = engine.exports.exportLayerPixels('L');
    setDocument({ ...document, layers: [rasterLayer('L', { imageName: 'B' })] });
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

  it('does not publish a rasterization from a replaced document that reuses the layer id', async () => {
    const first = createDeferred<Blob>();
    const second = createDeferred<Blob>();
    const blobA = new Blob(['document-A']);
    const blobB = new Blob(['document-B']);
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn((blob) =>
      Promise.resolve(blob === blobA ? recordingBitmap('document-A') : recordingBitmap('document-B'))
    );
    const imageResolver = vi.fn((imageName: string) => (imageName === 'A' ? first.promise : second.promise));
    const document = { ...makeDoc(), layers: [rasterLayer('L', { imageName: 'A' })] };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({ backend, imageResolver, projectId: 'p1', store });

    const oldExport = engine.exports.exportLayerPixels('L');
    setDocument({ ...document, layers: [rasterLayer('L', { imageName: 'B' })] }, 1);
    const newExport = engine.exports.exportLayerPixels('L');

    second.resolve(blobB);
    const newResult = await newExport;
    expect(newResult.status).toBe('ok');
    if (newResult.status !== 'ok') {
      throw new Error('replacement rasterization did not publish');
    }
    const publicationAfterReplacement = backend.drawSourcesFor(newResult.surface as StubRasterSurface);

    first.resolve(blobA);
    expect(await oldExport).toEqual({ status: 'not-ready' });
    expect(backend.drawSourcesFor(newResult.surface as StubRasterSurface)).toEqual(publicationAfterReplacement);
    engine.lifecycle.dispose();
  });

  it('shares one rasterization between concurrent exports of the same version', async () => {
    const blob = new Blob(['shared']);
    const imageResolver = vi.fn(() => Promise.resolve(blob));
    const { store } = createFakeStore({ ...makeDoc(), layers: [rasterLayer('L')] });
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

  it('does not return a default export when the layer becomes disabled during rasterization', async () => {
    const pending = createDeferred<Blob>();
    const layer = rasterLayer('L');
    const document = { ...makeDoc(), layers: [layer] };
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });

    const exported = engine.exports.exportLayerPixels('L');
    setDocument({ ...document, layers: [{ ...layer, isEnabled: false }] });
    pending.resolve(new Blob());

    expect(await exported).toEqual({ status: 'disabled' });
    engine.lifecycle.dispose();
  });

  it('publishes an empty paint cache without drawing a zero-sized scratch canvas', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const empty = { ...rasterLayer('empty'), source: { bitmap: null, type: 'paint' } as const };
    const { store } = createReactiveStore({ ...makeDoc(), layers: [empty] });
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
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();

    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();

    const zeroSizedDraws = surfaces.flatMap((surface) =>
      surface.callLog.filter((entry) => {
        if (entry.op !== 'drawImage') {
          return false;
        }
        const source = entry.args[0] as { height?: number; width?: number };
        return source.width === 0 || source.height === 0;
      })
    );
    expect(zeroSizedDraws).toEqual([]);
    expect(await engine.exports.exportLayerPixels('empty')).toEqual({ status: 'empty' });
    engine.lifecycle.dispose();
  });

  it('retries a same-version rasterization after a one-time synchronous rasterizer failure', async () => {
    const shape: CanvasLayerSourceContract = {
      fill: '#fff',
      height: 10,
      kind: 'rect',
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: 10,
    };
    const layer = { ...rasterLayer('shape'), source: shape };
    const { store } = createFakeStore({ ...makeDoc(), layers: [layer] });
    const base = createTestStubRasterBackend();
    let shouldThrow = true;
    const engine = createCanvasEngine({
      backend: {
        ...base,
        createSurface: (width, height) => {
          const surface = base.createSurface(width, height);
          const ctx = new Proxy(surface.ctx, {
            get(target, property, receiver) {
              if (property === 'fill') {
                return (...args: unknown[]) => {
                  if (shouldThrow) {
                    shouldThrow = false;
                    throw new Error('one-time fill failure');
                  }
                  return Reflect.apply(target.fill, target, args);
                };
              }
              return Reflect.get(target, property, receiver);
            },
          });
          Object.defineProperty(surface, 'ctx', { value: ctx });
          return surface;
        },
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.exports.exportLayerPixels('shape')).toEqual({ status: 'not-ready' });
    expect((await engine.exports.exportLayerPixels('shape')).status).toBe('ok');
    engine.lifecycle.dispose();
  });

  it('exportBakedLayerPixels draws the cache through the layer transform into a document-space surface', async () => {
    const layer = {
      ...rasterLayer('a'),
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 5, y: 6 },
    };
    const { store } = createFakeStore({ ...makeDoc(), layers: [layer] });
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

  it('exportBakedLayerPixels bakes raster adjustments into the exported surface only', async () => {
    const layer = {
      ...rasterLayer('a'),
      adjustments: { brightness: 0.25, contrast: 0, saturation: 0 },
    };
    const { store } = createFakeStore({ ...makeDoc(), layers: [layer] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    const raw = await engine.exports.exportLayerPixels('a');
    const baked = await engine.exports.exportBakedLayerPixels('a');

    expect(raw.status).toBe('ok');
    expect(baked.status).toBe('ok');
    if (raw.status === 'ok' && baked.status === 'ok') {
      expect(baked.surface).not.toBe(raw.surface);
      expect((raw.surface as StubRasterSurface).callLog.some((entry) => entry.op === 'putImageData')).toBe(false);
      expect((baked.surface as StubRasterSurface).callLog).toEqual(
        expect.arrayContaining([
          { args: [0, 0, 10, 10], op: 'getImageData' },
          expect.objectContaining({ op: 'putImageData' }),
        ])
      );
    }
    engine.lifecycle.dispose();
  });

  it('exportBakedLayerPixels applies adjustments exactly once when explicitly requested', async () => {
    const layer = {
      ...rasterLayer('a'),
      adjustments: { brightness: 0.25, contrast: 0, saturation: 0 },
    };
    const { store } = createFakeStore({ ...makeDoc(), layers: [layer] });
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
      projectId: 'p1',
      store,
    });

    expect((await engine.exports.exportBakedLayerPixels('a', { applyAdjustments: true })).status).toBe('ok');

    expect(surfaces.flatMap((surface) => surface.callLog).filter((entry) => entry.op === 'getImageData')).toHaveLength(
      1
    );
    engine.lifecycle.dispose();
  });

  it('exportBakedLayerBlob encodes the baked layer surface as PNG', async () => {
    const { engine } = createEngine();

    const result = await engine.exports.exportBakedLayerBlob('a');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.rect).toEqual({ height: 10, width: 10, x: 0, y: 0 });
      expect(result.blob.type).toBe('image/png');
      expect(await result.blob.text()).toBe('stub-surface-10x10');
    }
    engine.lifecycle.dispose();
  });

  it('returns a current LayerExportGuard from local, baked-pixel, and baked-blob exports', async () => {
    const { doc, engine } = createEngine();

    const local = await engine.exports.exportLayerPixels('a');
    const baked = await engine.exports.exportBakedLayerPixels('a');
    const blob = await engine.exports.exportBakedLayerBlob('a');

    expect(local.status).toBe('ok');
    expect(baked.status).toBe('ok');
    expect(blob.status).toBe('ok');
    if (local.status !== 'ok' || baked.status !== 'ok' || blob.status !== 'ok') {
      throw new Error('expected successful exports');
    }
    for (const result of [local, baked, blob]) {
      expect(result.guard).toMatchObject({ layer: doc.layers[0], layerId: 'a', projectId: 'p1' });
      expect(engine.exports.isLayerExportGuardCurrent(result.guard)).toBe(true);
    }
    engine.lifecycle.dispose();
  });

  it('captures the current layer export guard synchronously after its cache is ready', async () => {
    const { engine } = createEngine();
    await engine.exports.exportLayerPixels('a');

    const guard = engine.exports.captureLayerExportGuard('a');

    expect(guard).not.toBeNull();
    expect(guard).toMatchObject({ layerId: 'a', projectId: 'p1' });
    expect(engine.exports.isLayerExportGuardCurrent(guard!)).toBe(true);
    expect(engine.exports.captureLayerExportGuard('missing')).toBeNull();
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

    setDocument({ ...document, layers: [{ ...document.layers[0]!, opacity: 0.5 }] });

    expect(engine.exports.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    engine.lifecycle.dispose();
  });

  it('invalidates a canvas operation when its layer source changes or its layer is removed', async () => {
    for (const nextLayers of [
      [
        {
          ...makeDoc().layers[0]!,
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

      setDocument({ ...document, layers: nextLayers });

      expect(cleanupPreview).toHaveBeenCalledOnce();
      expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
      engine.lifecycle.dispose();
    }
  });

  it('invalidates a LayerExportGuard when the document is replaced with the same layer object', async () => {
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

    setDocument({ ...document, layers: document.layers }, 1);

    expect(engine.exports.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    engine.lifecycle.dispose();
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

  it('does not return an encoded layer blob after its export guard becomes stale', async () => {
    const encoded = createDeferred<Blob>();
    const base = createTestStubRasterBackend();
    const encodeSurface = vi.fn(() => encoded.promise);
    const document = makeDoc();
    const { setDocument, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: { ...base, encodeSurface },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    const pending = engine.exports.exportBakedLayerBlob('a');
    await vi.waitFor(() => expect(encodeSurface).toHaveBeenCalledOnce());
    setDocument({ ...document, layers: [{ ...document.layers[0]!, opacity: 0.5 }] });
    encoded.resolve(new Blob(['stale'], { type: 'image/png' }));

    expect(await pending).toEqual({ status: 'not-ready' });
    engine.lifecycle.dispose();
  });

  it('invalidates a LayerExportGuard when live cache pixels are edited', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const document = paintDoc();
    const layer = document.layers[0];
    if (!layer || layer.type !== 'raster') {
      throw new Error('expected paint layer');
    }
    const persisted: CanvasDocumentContractV2 = {
      ...document,
      layers: [
        {
          ...layer,
          source: { bitmap: { height: 20, imageName: 'paint-pixels', width: 20 }, type: 'paint' },
        },
      ],
    };
    const { projectId, store } = createReducerBackedStore(persisted);
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
    const exported = await engine.exports.exportLayerPixels(layer.id);
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }

    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    overlay.fire('pointermove', pointerAt(10, 10));
    overlay.fire('pointerup', pointerAt(10, 10, { buttons: 0 }));

    expect(engine.exports.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox crops only the bbox overlap and restores exact contracts/cache snapshots on undo/redo', async () => {
    const layer = {
      ...rasterLayer('a'),
      adjustments: { brightness: 0.2, contrast: -0.1, saturation: 0.3 },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 5, y: 5 },
    };
    const document = { ...makeDoc(), bbox: { height: 8, width: 10, x: 8, y: 2 }, layers: [layer] };
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
    const beforeContract = structuredClone(engine.document.getDocument()!.layers[0]!);
    const beforeExport = await engine.exports.exportLayerPixels('a');
    expect(beforeExport.status).toBe('ok');
    if (beforeExport.status !== 'ok') {
      throw new Error('expected original cache pixels');
    }
    const thumbnailListener = vi.fn();
    engine.stores.thumbnailVersion.subscribeKey('a', thumbnailListener);
    bitmapStore.markLayerDirty.mockClear();

    expect(await engine.layers.cropLayerToBbox('a')).toEqual({ status: 'cropped' });

    const afterContract = engine.document.getDocument()!.layers[0]!;
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

    engine.history.undo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(beforeContract);
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

    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(afterContract);
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

  it('cropLayerToBbox preserves control adapter settings while removing the baked filter', async () => {
    const control: CanvasLayerContract = {
      adapter: {
        beginEndStepPct: [0.1, 0.9],
        controlMode: 'more_control',
        kind: 'controlnet',
        model: 'm',
        weight: 0.7,
      },
      blendMode: 'screen',
      filter: { settings: { low: 10 }, type: 'canny' },
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control',
      opacity: 0.6,
      source: { image: { height: 10, imageName: 'control', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const document = { ...makeDoc(), bbox: { height: 7, width: 6, x: 2, y: 3 }, layers: [control] };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.layers.cropLayerToBbox('control')).toEqual({ status: 'cropped' });
    const cropped = engine.document.getDocument()!.layers[0];
    expect(cropped).toMatchObject({
      adapter: control.adapter,
      source: { bitmap: null, offset: { x: 2, y: 3 }, type: 'paint' },
      type: 'control',
      withTransparencyEffect: true,
    });
    expect(cropped && 'filter' in cropped).toBe(false);
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox preserves mask fill/noise/denoise configuration while replacing pixels and offset', async () => {
    const layer: CanvasInpaintMaskLayerContract = {
      ...maskLayer('mask'),
      denoiseLimit: 0.45,
      mask: {
        bitmap: { height: 10, imageName: 'mask-img', width: 10 },
        fill: { color: '#123456', style: 'crosshatch' },
        offset: { x: 0, y: 0 },
      },
      noiseLevel: 0.25,
    };
    const document = { ...makeDoc(), bbox: { height: 7, width: 6, x: 2, y: 3 }, layers: [layer] };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.layers.cropLayerToBbox('mask')).toEqual({ status: 'cropped' });
    expect(engine.document.getDocument()!.layers[0]).toMatchObject({
      denoiseLimit: 0.45,
      mask: { bitmap: null, fill: layer.mask.fill, offset: { x: 2, y: 3 } },
      noiseLevel: 0.25,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    });
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox preserves regional prompts and reference images while replacing mask pixels', async () => {
    const layer: CanvasLayerContract = {
      autoNegative: false,
      blendMode: 'normal',
      id: 'region',
      isEnabled: true,
      isLocked: false,
      mask: {
        bitmap: { height: 10, imageName: 'region-mask', width: 10 },
        fill: { color: '#abcdef', style: 'vertical' },
      },
      name: 'Region',
      negativePrompt: 'negative',
      opacity: 0.8,
      positivePrompt: 'positive',
      referenceImages: [{ config: { image: null, type: 'flux2_reference_image' }, id: 'reference', isEnabled: true }],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'regional_guidance',
    };
    const document = { ...makeDoc(), bbox: { height: 7, width: 6, x: 2, y: 3 }, layers: [layer] };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.layers.cropLayerToBbox('region')).toEqual({ status: 'cropped' });
    expect(engine.document.getDocument()!.layers[0]).toMatchObject({
      autoNegative: false,
      mask: { bitmap: null, fill: layer.mask.fill, offset: { x: 2, y: 3 } },
      negativePrompt: 'negative',
      positivePrompt: 'positive',
      referenceImages: layer.referenceImages,
      type: 'regional_guidance',
    });
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox returns empty without mutation when the layer and bbox do not overlap', async () => {
    const document = { ...makeDoc(), bbox: { height: 5, width: 5, x: 50, y: 50 } };
    const { projectId, store } = createReducerBackedStore(document);
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const before = engine.document.getDocument();

    expect(await engine.layers.cropLayerToBbox('a')).toEqual({ status: 'empty' });
    expect(engine.document.getDocument()).toBe(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox reports missing, locked, unsupported, and not-ready layers explicitly', async () => {
    const polygon: CanvasLayerSourceContract = {
      fill: '#000',
      height: 10,
      kind: 'polygon',
      points: [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 5, y: 10 },
      ],
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: 10,
    };
    const layers = [
      { ...rasterLayer('locked'), isLocked: true },
      { ...rasterLayer('polygon'), source: polygon },
      rasterLayer('pending'),
    ];
    const document = { ...makeDoc(), layers };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.reject(new Error('decode failed')),
      projectId,
      store,
    });

    expect(await engine.layers.cropLayerToBbox('missing')).toEqual({ status: 'missing' });
    expect(await engine.layers.cropLayerToBbox('locked')).toEqual({ status: 'locked' });
    expect(await engine.layers.cropLayerToBbox('polygon')).toEqual({ status: 'unsupported' });
    expect(await engine.layers.cropLayerToBbox('pending')).toEqual({ status: 'not-ready' });
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox returns not-ready without mutation when the source changes during rasterization', async () => {
    const pending = createDeferred<Blob>();
    const document = { ...makeDoc(), layers: [rasterLayer('a', { imageName: 'A' })] };
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
    setDocument({ ...document, layers: [rasterLayer('a', { imageName: 'B' })] });
    pending.resolve(new Blob());

    expect(await crop).toEqual({ status: 'not-ready' });
    expect(store.dispatch).not.toHaveBeenCalled();
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox aborts when an operation starts during its deferred export', async () => {
    const pending = createDeferred<Blob>();
    const document = { ...makeDoc(), layers: [rasterLayer('operation'), rasterLayer('a')] };
    const { projectId, store } = createReducerBackedStore(document);
    const bitmapStore = createSpyBitmapStore();
    const resolver = vi.fn((imageName: string) => (imageName === 'a' ? pending.promise : Promise.resolve(new Blob())));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: resolver,
      projectId,
      store,
    });
    const operationExport = await engine.exports.exportLayerPixels('operation');
    if (operationExport.status !== 'ok') {
      throw new Error('expected an operation guard');
    }
    (store.dispatch as Mock).mockClear();
    bitmapStore.markLayerDirty.mockClear();

    const crop = engine.layers.cropLayerToBbox('a');
    await vi.waitFor(() => expect(resolver).toHaveBeenCalledWith('a', expect.any(AbortSignal)));
    const cleanupPreview = vi.fn();
    getCanvasOperations(engine).controller.start({
      cleanupPreview,
      guard: operationExport.guard,
      identity: { kind: 'filter', layerId: 'operation', projectId },
    });
    pending.resolve(new Blob());

    await expect(crop).resolves.toEqual({ status: 'busy' });
    expect(engine.document.getDocument()).toEqual(document);
    expect(store.dispatch).not.toHaveBeenCalled();
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(cleanupPreview).not.toHaveBeenCalled();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter', layerId: 'operation' },
      status: 'active',
    });
    engine.lifecycle.dispose();
  });

  it('cropLayerToBbox reports busy during an open gesture', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const paint = {
      ...rasterLayer('paint'),
      source: { bitmap: { height: 10, imageName: 'paint', width: 10 }, type: 'paint' as const },
    };
    const busyHarness = createReducerBackedStore({ ...makeDoc(), layers: [paint], selectedLayerId: 'paint' });
    const busyEngine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: busyHarness.projectId,
      store: busyHarness.store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    busyEngine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    busyEngine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(2, 2));
    expect(await busyEngine.layers.cropLayerToBbox('paint')).toEqual({ status: 'busy' });
    overlay.fire('pointerup', pointerAt(2, 2, { buttons: 0 }));
    busyEngine.lifecycle.dispose();
  });

  it('cropLayerToBbox reports failed for unexpected raster errors', async () => {
    const base = createTestStubRasterBackend();
    const failedHarness = createReducerBackedStore({
      ...makeDoc(),
      bbox: { height: 7, width: 6, x: 2, y: 3 },
    });
    const failedEngine = createCanvasEngine({
      backend: {
        ...base,
        createSurface: (width, height) => {
          if (width === 6 && height === 7) {
            throw new Error('crop allocation failed');
          }
          return base.createSurface(width, height);
        },
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: failedHarness.projectId,
      store: failedHarness.store,
    });
    expect(await failedEngine.layers.cropLayerToBbox('a')).toEqual({
      message: 'crop allocation failed',
      status: 'failed',
    });
    failedEngine.lifecycle.dispose();
  });

  it('copyLayerToRaster finishes in its bound project after another project becomes active', async () => {
    const pending = createDeferred<Blob>();
    const document = makeDoc();
    const { setActiveProjectId, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });

    const copy = engine.layers.copyLayerToRaster('a');
    setActiveProjectId('p2');
    pending.resolve(new Blob());

    expect(await copy).not.toBeNull();
    expect(store.dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: 'applyCanvasLayerStackMutation' }));
    engine.lifecycle.dispose();
  });

  it('copyLayerToRaster adds a baked paint copy directly above the source layer', async () => {
    const doc = { ...makeDoc(), layers: [rasterLayer('top'), rasterLayer('a')] };
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

    const newId = await engine.layers.copyLayerToRaster('a');

    expect(newId).toMatch(/^layer-/);
    const mutation = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'applyCanvasLayerStackMutation');
    expect(mutation).toBeDefined();
    const added = mutation?.type === 'applyCanvasLayerStackMutation' ? mutation.add : undefined;
    const addedLayer = added?.layers[0];
    if (added && addedLayer?.type === 'raster') {
      expect(added.index).toBe(1);
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

  it('copyLayerToRaster returns null for empty layers', async () => {
    const empty = { ...rasterLayer('empty'), source: { bitmap: null, type: 'paint' } as const };
    const { store } = createFakeStore({ ...makeDoc(), layers: [empty] });
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.layers.copyLayerToRaster('empty')).toBeNull();
    expect(dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('copyLayerToRaster records mask copies in engine history', async () => {
    const mask: CanvasInpaintMaskLayerContract = {
      ...maskLayer('mask'),
      mask: {
        bitmap: { height: 10, imageName: 'mask-image', width: 10 },
        fill: { color: '#e07575', style: 'diagonal' },
      },
    };
    const { projectId, store } = createReducerBackedStore({ ...makeDoc(), layers: [mask] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    const newId = await engine.layers.copyLayerToRaster('mask');
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === newId)).toBe(true);

    engine.history.undo();
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === newId)).toBe(false);
    engine.history.redo();
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === newId)).toBe(true);
    engine.lifecycle.dispose();
  });

  it('copyLayerToRaster redo cache failure leaves the copy absent and retryable', async () => {
    const { projectId, store } = createReducerBackedStore(makeDoc());
    const base = createTestStubRasterBackend();
    let failNextAllocation = false;
    const engine = createCanvasEngine({
      backend: {
        ...base,
        createSurface: (width, height) => {
          if (failNextAllocation) {
            failNextAllocation = false;
            throw new Error('copy replay allocation failed');
          }
          return base.createSurface(width, height);
        },
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    const newId = await engine.layers.copyLayerToRaster('a');
    expect(newId).not.toBeNull();
    engine.history.undo();
    expect(engine.document.getDocument()!.layers.filter((layer) => layer.id === newId)).toHaveLength(0);

    failNextAllocation = true;
    expect(() => engine.history.redo()).toThrow('copy replay allocation failed');

    expect(engine.document.getDocument()!.layers.filter((layer) => layer.id === newId)).toHaveLength(0);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.history.redo();
    expect(engine.document.getDocument()!.layers.filter((layer) => layer.id === newId)).toHaveLength(1);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.history.undo();
    expect(engine.document.getDocument()!.layers.filter((layer) => layer.id === newId)).toHaveLength(0);
    engine.lifecycle.dispose();
  });

  it('setTool switches the active tool and updates the store', () => {
    const { engine } = createEngine();
    const listener = vi.fn();
    engine.stores.activeTool.subscribe(listener);

    engine.tools.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');
    expect(listener).toHaveBeenCalledTimes(1);

    // Setting the same tool again is a no-op.
    engine.tools.setTool('brush');
    expect(listener).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('interaction lock forces view and refuses non-view tools until unlocked', () => {
    const { engine } = createEngine();

    engine.tools.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');

    engine.tools.setInteractionLocked(true);
    expect(engine.stores.activeTool.get()).toBe('view');

    engine.tools.setTool('bbox');
    expect(engine.stores.activeTool.get()).toBe('view');

    engine.tools.setTool('colorPicker');
    expect(engine.stores.activeTool.get()).toBe('view');

    engine.tools.setInteractionLocked(false);
    engine.tools.setTool('bbox');
    expect(engine.stores.activeTool.get()).toBe('bbox');

    engine.lifecycle.dispose();
  });

  it('registers the brush and eraser tools', () => {
    const { engine } = createEngine();
    engine.tools.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');
    engine.tools.setTool('eraser');
    expect(engine.stores.activeTool.get()).toBe('eraser');
    engine.lifecycle.dispose();
  });

  it('onStrokeCommitted returns an unsubscribe and never fires without a stroke', () => {
    const { engine } = createEngine();
    const listener = vi.fn();
    const unsubscribe = engine.tools.onStrokeCommitted(listener);
    expect(typeof unsubscribe).toBe('function');
    unsubscribe();
    expect(listener).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('resize records the viewport size and clamps the device-pixel ratio', () => {
    const { engine } = createEngine();
    engine.surface.resize(800, 600, 4);
    expect(engine.viewport.getViewport().getViewportSize()).toEqual({ height: 600, width: 800 });
    expect(engine.viewport.getViewport().getDpr()).toBe(2);
    engine.lifecycle.dispose();
  });

  it('dispose removes both store subscriptions', () => {
    const { engine, unsubscribe } = createEngine();
    expect(unsubscribe).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
    expect(unsubscribe).toHaveBeenCalledOnce();
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
    const doc: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [a, b],
      selectedLayerId: null,
      version: 2,
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

    // Initial frame: dispatches (and, after a microtask flush, completes) the
    // rasterize for both layers.
    raf.flush();
    await flushMicrotasks();
    raf.flush(); // the follow-up frame each rasterize's resolve scheduled
    expect(resolver).toHaveBeenCalledTimes(2);

    // Isolate the reorder-triggered frame's draw log.
    screen.surface.callLog.length = 0;

    // Pure reorder: new array reference, same layer object references, swapped order.
    setDocument({ ...doc, layers: [b, a] });
    raf.flush();

    // No layer content changed, so nothing should be re-rasterized.
    expect(resolver).toHaveBeenCalledTimes(2);

    // The compositor draws bottom-to-top; `a` (alpha 0.25) is now the bottom
    // layer and `b` (alpha 0.75) the top, so alpha is applied in that order.
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

    const doc: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [rasterLayer('a'), rasterLayer('b')],
      selectedLayerId: null,
      version: 2,
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

    // Still present: nothing removed yet.
    expect(adjustedSurfaceCacheDeletes).not.toContain('a');

    // Remove layer 'a' via an ordinary layer-array edit (onLayersChanged → dropLayer).
    setDocument({ ...doc, layers: [rasterLayer('b')] });
    raf.flush();

    // The removed layer's adjusted-surface slot is dropped alongside its layer cache.
    expect(adjustedSurfaceCacheDeletes).toContain('a');

    engine.lifecycle.dispose();
  });
});

describe('resize: synchronous composite (no flash/strobe on panel drag)', () => {
  it('composites in the SAME task as the backing-store resize, scheduling no deferred frame', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const doc = makeDoc();
    const { store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);

    // Run the initial attach frame so the surface holds a composited frame, then
    // isolate what the resize alone draws.
    raf.flush();
    screen.surface.callLog.length = 0;

    // A single resize event (as a ResizeObserver fires mid panel-drag). Sizing a
    // `<canvas>` backing store CLEARS it; the fix recomposites synchronously so the
    // cleared surface is repainted before the browser paints (no blank frame).
    // No `raf.flush()` follows — so any draw seen here happened IN-TASK.
    engine.surface.resize(800, 600, 1);

    // Backing store was resized in the same call...
    expect(screen.element.width).toBe(800);
    expect(screen.element.height).toBe(600);
    // ...and the composite ran SYNCHRONOUSLY (drew into the surface without a rAF
    // flush). `compositeDocument` always clears the target, so a `clearRect` here
    // proves the recomposite happened in-task rather than being deferred to rAF.
    const composited = screen.surface.callLog.some((entry) => entry.op === 'clearRect');
    expect(composited).toBe(true);

    engine.lifecycle.dispose();
  });

  it('schedules NO deferred frame after the synchronous composite (exactly one composite per resize)', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const doc = makeDoc();
    const { store } = createReactiveStore(doc);
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
    screen.surface.callLog.length = 0;

    engine.surface.resize(800, 600, 1);

    // The synchronous composite ran...
    expect(screen.surface.callLog.some((entry) => entry.op === 'clearRect')).toBe(true);
    // ...and setViewportSize's viewport-subscription `{ view: true }` invalidate was
    // suppressed, so NO rAF frame is queued to recomposite the identical content.
    expect(raf.pendingCount()).toBe(0);

    // Draining any frame anyway must not produce a second composite.
    screen.surface.callLog.length = 0;
    raf.flush();
    expect(screen.surface.callLog.some((entry) => entry.op === 'clearRect')).toBe(false);

    engine.lifecycle.dispose();
  });

  it('marks the composite dirty flag so the synchronous path is not gated out (T22)', () => {
    // A resize must force a full recomposite even though no layer pixels changed:
    // if it only marked `overlay`, the T22 `needsComposite` gate would skip the
    // composite and the just-cleared screen surface would stay blank.
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const doc = makeDoc();
    const { store } = createReactiveStore(doc);
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
    screen.surface.callLog.length = 0;

    engine.surface.resize(320, 240, 2);

    // The screen (document composite) surface — not just the overlay — was redrawn.
    expect(screen.surface.callLog.some((entry) => entry.op === 'clearRect')).toBe(true);
    engine.lifecycle.dispose();
  });
});

describe('ensureLayerCaches: edit-during-rasterize race', () => {
  it('re-rasterizes with the newest source when an edit lands while a rasterize is in flight', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const doc = makeDoc(); // single layer 'a', imageName 'a'
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

    // Subscribe to thumbnail-version notifications for layer 'a' *before* either
    // rasterize settles. A bare `.get()` check at the end can't distinguish
    // "notified once, at the wrong time, with stale pixels on the surface" from
    // "notified once, at the right time, with fresh pixels on the surface" --
    // both leave the same final version number. Tracking call count/timing does.
    const thumbnailListener = vi.fn();
    const unsubscribe = engine.stores.thumbnailVersion.subscribeKey('a', thumbnailListener);

    // Frame 1: dispatches the first rasterize for imageName 'a'; it stays in flight
    // (the deferred is never auto-resolved).
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'a', expect.any(AbortSignal));

    // An edit lands mid-flight: same layer id, new object reference, new source.
    setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'a-v2' })] });

    // A frame runs while the first rasterize is still in flight. The source no
    // longer matches that job, so a fresh isolated job starts immediately.
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'a-v2', expect.any(AbortSignal));

    // The newer rasterize wins and publishes while the old decode is pending.
    deferreds.get('a-v2')!.resolve(new Blob());
    await flushMicrotasks();
    raf.flush();
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    expect(engine.stores.thumbnailVersion.get('a')).toBe(2);

    // The older decode resolves afterwards. It may finish its isolated scratch
    // draw, but it must neither publish nor notify subscribers.
    deferreds.get('a')!.resolve(new Blob());
    await flushMicrotasks();
    raf.flush();
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenCalledTimes(2);

    unsubscribe();
    engine.lifecycle.dispose();
  });
});

// ---- Engine-owned history (P2.3) --------------------------------------
//
// Drives a real brush stroke end-to-end through the engine (attach → dispatch
// pointer events → commit) and asserts the history wiring: strokeCommitted pushes
// an image patch, undo/redo write the before/after pixels back into the layer's
// cache surface AND re-mark the layer dirty for persistence, and the canUndo /
// canRedo stores track the stacks.

const paintDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [
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
  ],
  selectedLayerId: 'paint1',
  version: 2,
  width: 100,
});

type ControlPaintHarnessOverrides = Partial<Omit<CanvasControlLayerContract, 'source' | 'type'>> & {
  source: CanvasControlLayerContract['source'];
};

interface ControlPaintHarnessOptions {
  bitmapStoreFactory?: (deps: {
    backend: StubRasterBackend;
    projectId: string;
    store: EngineStore;
  }) => ReturnType<typeof createSpyBitmapStore>;
  pixelWrites?: { enabled: boolean };
}

const createControlPaintHarness = (
  overrides: ControlPaintHarnessOverrides,
  options: ControlPaintHarnessOptions = {}
) => {
  const raf = createControllableRaf();
  vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
  vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
  vi.stubGlobal('Path2D', class FakePath2D {});

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
  const document: CanvasDocumentContractV2 = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers: [layer],
    selectedLayerId: layer.id,
    version: 2,
    width: 100,
  };
  const { projectId, store } = createReducerBackedStore(document);
  const baseBackend = createTestStubRasterBackend();
  const pixelWrites = options.pixelWrites ?? { enabled: true };
  const backend: StubRasterBackend = {
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

const createControlSelectionHarness = (overrides: ControlPaintHarnessOverrides) => {
  const selectionPixelWrites = { enabled: true };
  return {
    ...createControlPaintHarness(overrides, { pixelWrites: selectionPixelWrites }),
    setSelectionPixelWrites: (enabled: boolean) => {
      selectionPixelWrites.enabled = enabled;
    },
  };
};

/** A minimal in-memory bitmap store: records dirty-marks, never touches the network. */
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
            const layer = store
              .getState()
              .projects.find((project) => project.id === projectId)
              ?.canvas.document.layers.find((candidate) => candidate.id === layerId);
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

/** A fake canvas that also lets a test fire pointer events at the engine's listeners. */
const createInputCanvas = (
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

/** Every `putImageData` recorded across all surfaces the backend created. */
const putImageDataCalls = (surfaces: StubRasterSurface[]): { image: unknown; x: unknown; y: unknown }[] =>
  surfaces.flatMap((surface) =>
    surface.callLog
      .filter((entry) => entry.op === 'putImageData')
      .map((entry) => ({ image: entry.args[0], x: entry.args[1], y: entry.args[2] }))
  );

describe('engine-owned control pixel editing', () => {
  it('brushes an empty control in place and never creates a raster layer', () => {
    const h = createControlPaintHarness({ source: { bitmap: null, type: 'paint' } });
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(40, 40));
    h.overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    expect(h.engine.document.getDocument()!.layers).toHaveLength(1);
    expect(h.engine.document.getDocument()!.layers[0]).toMatchObject({ id: 'control', type: 'control' });
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

  it('restores the exact direct cache after a cancelled stroke grows it and permits a subsequent edit', async () => {
    const h = createControlPaintHarness({
      source: { bitmap: { height: 10, imageName: 'control-paint', width: 10 }, type: 'paint' },
    });
    await h.publishInitialCache();
    const before = await snapshotLayerCache(h.engine, 'control');
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(80, 80));
    h.overlay.fire('pointermove', pointerAt(90, 90));
    h.overlay.fire('pointercancel', pointerAt(90, 90, { buttons: 0 }));

    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(before.surface);
      expect(restored.rect).toEqual(before.rect);
      expect(restored.guard.cacheVersion).toBe(before.version);
      const finalWrite = (restored.surface as StubRasterSurface).callLog
        .filter((entry) => entry.op === 'putImageData')
        .at(-1);
      expect(finalWrite?.args[0]).toMatchObject({ height: before.rect.height, width: before.rect.width });
    }
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.strokes).toHaveLength(0);

    h.overlay.fire('pointerdown', pointerAt(20, 20));
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledTimes(2);
    h.overlay.fire('pointercancel', pointerAt(20, 20, { buttons: 0 }));
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledTimes(2);
    h.engine.lifecycle.dispose();
  });

  it.each(['brush', 'eraser'] as const)(
    'treats a byte-identical direct %s stroke as an exact rollback',
    async (tool) => {
      const h = createControlPaintHarness(
        { source: { bitmap: { height: 10, imageName: 'control-paint', width: 10 }, type: 'paint' } },
        { pixelWrites: { enabled: false } }
      );
      await h.publishInitialCache();
      const before = await snapshotLayerCache(h.engine, 'control');
      h.engine.tools.setTool(tool);
      h.overlay.fire('pointerdown', pointerAt(5, 5));
      h.overlay.fire('pointermove', pointerAt(8, 8));
      h.overlay.fire('pointerup', pointerAt(8, 8, { buttons: 0 }));

      const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
      expect(restored.status).toBe('ok');
      if (restored.status === 'ok') {
        expect(restored.surface).toBe(before.surface);
        expect(restored.rect).toEqual(before.rect);
        expect(restored.guard.cacheVersion).toBe(before.version);
      }
      expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
      expect(h.engine.stores.canUndo.get()).toBe(false);
      expect(h.strokes).toHaveLength(0);
      h.engine.lifecycle.dispose();
    }
  );

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

  it('invalidates an in-flight direct-control upload on begin and releases the barrier after cancel', async () => {
    const uploads = [
      createDeferred<{ height: number; imageName: string; width: number }>(),
      createDeferred<{ height: number; imageName: string; width: number }>(),
    ];
    let uploadIndex = 0;
    const uploadImage = vi.fn(() => uploads[uploadIndex++]!.promise);
    const h = await createRealControlPersistenceHarness(uploadImage);
    h.bitmapStore.markLayerDirty('control');
    const barrier = h.bitmapStore.flushPendingUploads();
    for (let tick = 0; tick < 50 && uploadImage.mock.calls.length < 1; tick += 1) {
      await Promise.resolve();
    }
    expect(uploadImage).toHaveBeenCalledOnce();

    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(30, 30));
    let settled = false;
    void barrier.then(() => {
      settled = true;
    });
    uploads[0]!.resolve({ height: 10, imageName: 'obsolete-control', width: 10 });
    await Promise.resolve();
    await Promise.resolve();

    expect(settled).toBe(false);
    expect(uploadImage).toHaveBeenCalledOnce();

    h.overlay.fire('pointercancel', pointerAt(30, 30, { buttons: 0 }));
    for (let tick = 0; tick < 50 && uploadImage.mock.calls.length < 2; tick += 1) {
      await Promise.resolve();
    }
    expect(uploadImage).toHaveBeenCalledTimes(2);
    uploads[1]!.resolve({ height: 10, imageName: 'resumed-control', width: 10 });
    await drainMicrotasksUntil(() => settled);

    expect(settled).toBe(true);
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.strokes).toHaveLength(0);
    h.engine.lifecycle.dispose();
  });

  it.each([
    ['image control', { source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' } }],
    [
      'transformed empty control',
      {
        source: { bitmap: null, type: 'paint' },
        transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 7, y: 11 },
      },
    ],
  ] as const)('treats byte-identical brush and eraser strokes on a %s as rollback', async (_scenario, overrides) => {
    for (const tool of ['brush', 'eraser'] as const) {
      const h = createControlPaintHarness(overrides, { pixelWrites: { enabled: false } });
      if (overrides.source.type === 'image') {
        await h.publishInitialCache();
      }
      const beforeDocument = structuredClone(h.engine.document.getDocument());
      const beforeCache = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
      h.engine.tools.setTool(tool);
      h.overlay.fire('pointerdown', pointerAt(5, 5));
      h.overlay.fire('pointermove', pointerAt(8, 8));
      h.overlay.fire('pointerup', pointerAt(8, 8, { buttons: 0 }));

      expect(h.engine.document.getDocument()).toEqual(beforeDocument);
      const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
      expect(restored.status).toBe(beforeCache.status);
      if (restored.status === 'ok' && beforeCache.status === 'ok') {
        expect(restored.surface).toBe(beforeCache.surface);
        expect(restored.rect).toEqual(beforeCache.rect);
        expect(restored.guard.cacheVersion).toBe(beforeCache.guard.cacheVersion);
      }
      expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
      expect(h.engine.stores.canUndo.get()).toBe(false);
      expect(h.strokes).toHaveLength(0);
      h.engine.lifecycle.dispose();
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

  it('materializes an image control plus brush stroke as one reversible edit', async () => {
    const h = createControlPaintHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 20 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 7, y: 11 },
    });
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument()!.layers[0]);

    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(25, 25));
    h.overlay.fire('pointerup', pointerAt(25, 25, { buttons: 0 }));

    const after = structuredClone(h.engine.document.getDocument()!.layers[0]);
    expect(after).toMatchObject({
      adapter: before && 'adapter' in before ? before.adapter : undefined,
      filter: before && 'filter' in before ? before.filter : undefined,
      id: 'control',
      source: { type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    });
    expect(h.engine.stores.canUndo.get()).toBe(true);
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledWith('control');
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.markLayerDirty.mock.invocationCallOrder[0]).toBeLessThan(
      h.bitmapStore.releaseSuspendedLayer.mock.invocationCallOrder[0]!
    );

    h.engine.history.undo();
    expect(h.engine.document.getDocument()!.layers[0]).toEqual(before);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.engine.stores.canRedo.get()).toBe(true);

    h.engine.history.redo();
    expect(h.engine.document.getDocument()!.layers[0]).toEqual(after);
    h.engine.lifecycle.dispose();
  });

  it('restores a materialized control when layer-snapshot history preparation fails', async () => {
    const h = createControlPaintHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 20 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 7, y: 11 },
    });
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(25, 25));
    historyPreparationFaults.layerSnapshot = true;

    expect(() => h.overlay.fire('pointerup', pointerAt(25, 25, { buttons: 0 }))).toThrow(
      'layer snapshot preparation failed'
    );

    expect(h.engine.document.getDocument()).toEqual(before);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.strokes).toHaveLength(0);
    h.engine.lifecycle.dispose();
  });

  it('releases materialized persistence and edit ownership when rollback cleanup throws', async () => {
    const h = createControlPaintHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 20 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 7, y: 11 },
    });
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument());
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointermove', pointerAt(25, 25));
    historyPreparationFaults.layerSnapshot = true;
    adjustedSurfaceCacheDeleteFaults.add('control');

    expect(() => h.overlay.fire('pointerup', pointerAt(25, 25, { buttons: 0 }))).toThrow(
      'adjusted surface cache delete failed'
    );

    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.strokes).toHaveLength(0);

    historyPreparationFaults.layerSnapshot = false;
    adjustedSurfaceCacheDeleteFaults.clear();
    h.overlay.fire('pointerdown', pointerAt(30, 30));
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledTimes(2);
    h.overlay.fire('pointercancel', pointerAt(30, 30, { buttons: 0 }));
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledTimes(2);
    h.engine.lifecycle.dispose();
  });

  it.each(['eraser', 'brush'] as const)(
    'rolls back a prepared image control when a %s gesture is cancelled',
    async (tool) => {
      const h = createControlPaintHarness({
        source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      });
      await h.publishInitialCache();
      const before = structuredClone(h.engine.document.getDocument());
      h.engine.tools.setTool(tool);
      h.overlay.fire('pointerdown', pointerAt(5, 5));
      h.overlay.fire('pointercancel', pointerAt(5, 5, { buttons: 0 }));
      expect(h.engine.document.getDocument()).toEqual(before);
      expect(h.engine.stores.canUndo.get()).toBe(false);
      expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(h.bitmapStore.suspendLayer).toHaveBeenCalledWith('control');
      expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
      h.engine.lifecycle.dispose();
    }
  );

  it.each([
    ['locked', { isLocked: true }],
    ['disabled', { isEnabled: false }],
  ] as const)('does not mutate or auto-create over a %s control', (_scenario, patch) => {
    const h = createControlPaintHarness({ source: { bitmap: null, type: 'paint' }, ...patch });
    const before = structuredClone(h.engine.document.getDocument());
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(20, 20));
    h.overlay.fire('pointerup', pointerAt(20, 20, { buttons: 0 }));
    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('normalizes a transformed empty control before its first brush stroke', () => {
    const h = createControlPaintHarness({
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 30, y: 40 },
    });
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(35, 45));
    h.overlay.fire('pointerup', pointerAt(35, 45, { buttons: 0 }));
    expect(h.engine.document.getDocument()!.layers[0]).toMatchObject({
      id: 'control',
      source: { type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
    });
    h.engine.history.undo();
    expect(h.engine.document.getDocument()!.layers[0]).toMatchObject({
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 30, y: 40 },
    });
    h.engine.lifecycle.dispose();
  });

  it('does not edit or auto-create when an image control cache is not ready', () => {
    const h = createControlPaintHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
    });
    h.raf.flush();
    const before = structuredClone(h.engine.document.getDocument());

    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(5, 5));
    h.overlay.fire('pointerup', pointerAt(5, 5, { buttons: 0 }));

    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.engine.document.getDocument()!.layers).toHaveLength(1);
    expect(h.strokes).toHaveLength(0);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.discardLayer).not.toHaveBeenCalled();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.engine.stores.canRedo.get()).toBe(false);
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

  it('finishes affected-layer removal cleanup after direct gesture rollback throws', async () => {
    const h = createControlPaintHarness({
      source: { bitmap: { height: 10, imageName: 'control-paint', width: 10 }, type: 'paint' },
    });
    await h.publishInitialCache();
    h.engine.tools.setTool('brush');
    h.overlay.fire('pointerdown', pointerAt(8, 8));
    h.overlay.fire('pointermove', pointerAt(12, 12));
    adjustedSurfaceCacheDeleteFaults.add('control');

    expect(() => h.store.dispatch({ ids: ['control'], type: 'removeCanvasLayers' })).toThrow(
      'adjusted surface cache delete failed'
    );

    expect(h.engine.document.getDocument()!.layers).toHaveLength(0);
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.discardLayer).toHaveBeenCalledWith('control');
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
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // A brush gesture entirely inside the 100x100 document (identity viewport).
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    return { bitmapStore, engine, strokes, surfaces };
  };

  it('records a stroke and restores before/after pixels on undo/redo', () => {
    const { bitmapStore, engine, strokes, surfaces } = drawStroke();

    // One stroke committed and one history entry recorded.
    expect(strokes).toHaveLength(1);
    const event = strokes[0]!;
    expect(event.layerId).toBe('paint1');
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    // The commit marked the layer dirty for persistence.
    const dirtyAfterStroke = bitmapStore.markLayerDirty.mock.calls.length;
    expect(dirtyAfterStroke).toBeGreaterThanOrEqual(1);

    // Undo: the layer's cache surface receives putImageData(before), and the layer
    // is re-marked dirty (convergence path). Content-sized: the paint layer started
    // empty and grew to exactly the stroke's dirty rect, so the cache-local origin
    // equals the dirty-rect origin and the patch lands at surface (0, 0).
    engine.history.undo();
    const undoPut = putImageDataCalls(surfaces).find((call) => call.image === event.beforeImageData);
    expect(undoPut).toBeDefined();
    expect(undoPut!.x).toBe(0);
    expect(undoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    expect(bitmapStore.markLayerDirty.mock.calls.length).toBeGreaterThan(dirtyAfterStroke);

    // Redo: putImageData(after) restores the post-stroke pixels.
    engine.history.redo();
    const redoPut = putImageDataCalls(surfaces).find((call) => call.image === event.afterImageData);
    expect(redoPut).toBeDefined();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('round-trips pixels exactly across a cache-growing stroke → undo → redo', () => {
    // Integrated case: a multi-move stroke that GROWS the (initially empty) paint
    // cache across several pointer batches, then undo/redo restore the exact
    // before/after ImageData over the FULL grown extent. The piecewise pieces
    // (growth, patch application) are covered elsewhere; this pins them together.
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // Drag rightward across the document in several batches so the content-sized
    // cache grows (and reallocates) as the stroke extends.
    overlay.fire('pointerdown', pointerAt(10, 50));
    overlay.fire('pointermove', pointerAt(30, 50));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointermove', pointerAt(70, 50));
    overlay.fire('pointermove', pointerAt(90, 50));
    overlay.fire('pointerup', pointerAt(90, 50, { buttons: 0 }));

    expect(strokes).toHaveLength(1);
    const event = strokes[0]!;
    // The stroke grew the cache well beyond a single dab: the dirty rect spans most
    // of the drag width, and the captured before/after cover that full extent.
    expect(event.dirtyRect.width).toBeGreaterThan(60);
    expect(event.beforeImageData.width).toBe(event.dirtyRect.width);
    expect(event.beforeImageData.height).toBe(event.dirtyRect.height);
    expect(event.afterImageData.width).toBe(event.dirtyRect.width);
    expect(event.afterImageData.height).toBe(event.dirtyRect.height);

    // Undo writes the EXACT pre-stroke ImageData back into the cache. The cache
    // grew to exactly the (chunk-padded) dirty rect, so its local origin equals the
    // dirty-rect origin and the patch lands at surface (0, 0).
    engine.history.undo();
    const undoPut = putImageDataCalls(surfaces).find((call) => call.image === event.beforeImageData);
    expect(undoPut).toBeDefined();
    expect(undoPut!.x).toBe(0);
    expect(undoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    // Redo writes the EXACT post-stroke ImageData back — a lossless round-trip.
    engine.history.redo();
    const redoPut = putImageDataCalls(surfaces).find((call) => call.image === event.afterImageData);
    expect(redoPut).toBeDefined();
    expect(redoPut!.x).toBe(0);
    expect(redoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('clears history when the document is replaced', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // A dims change triggers onDocumentReplaced → history.clear().
    const replaced = { ...paintDoc(), height: 200, width: 200 };
    setDocument(replaced);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('clears history on a same-dimension snapshot restore (documentRevision bump)', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // A snapshot restore reuses the same dims AND layer ids (structuredClone), so
    // only the bumped documentRevision distinguishes it from an ordinary edit. It
    // must still clear history: a subsequent undo would otherwise put pre-restore
    // pixels over the restored content.
    setDocument(paintDoc(), 1);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('re-rasterizes and resets persistence bookkeeping on a revision-bump swap that reuses a layer id with a different source', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const docV1: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [rasterLayer('a', { imageName: 'src-v1' })],
      selectedLayerId: 'a',
      version: 2,
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

    // Initial rasterize of the v1 source.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'src-v1', expect.any(AbortSignal));

    // A revision bump that REUSES layer id 'a' with a DIFFERENT source: the
    // mirror routes this through onDocumentReplaced (a full swap), which a
    // reference diff alone could not tell from an ordinary edit. The engine must
    // invalidate the surviving cache entry so it re-rasterizes the new source
    // (a stale cache entry would keep rendering the v1 pixels).
    setDocument({ ...docV1, layers: [rasterLayer('a', { imageName: 'src-v2' })] }, 1);

    // Persistence bookkeeping for the outgoing document was dropped so a reused
    // layer id can't have its next legit persistence dispatch suppressed.
    expect(bitmapStore.reset).toHaveBeenCalledTimes(1);

    // The invalidated cache re-rasterizes, this time from the v2 source.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'src-v2', expect.any(AbortSignal));

    engine.lifecycle.dispose();
  });
});

describe('engine-owned history: undo/redo guarded during an active gesture', () => {
  it('no-ops undo/redo mid-stroke, then works after the gesture ends', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // First, record one committed stroke so there is something to undo.
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    expect(engine.stores.canUndo.get()).toBe(true);

    // Start a SECOND stroke and leave it open (pointer down + move, no up).
    // (The live session itself may draw, so snapshot the put count after it opens.)
    overlay.fire('pointerdown', pointerAt(50, 50));
    overlay.fire('pointermove', pointerAt(60, 60));
    const putsMidGesture = putImageDataCalls(surfaces).length;

    // Mid-gesture undo/redo must be no-ops: no history pop and no putImageData.
    engine.history.undo();
    engine.history.redo();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(putImageDataCalls(surfaces).length).toBe(putsMidGesture);
    // In particular, the first stroke's before pixels were never injected.
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.beforeImageData)).toBe(false);

    // End the gesture; now undo works and writes the newest stroke's before pixels.
    overlay.fire('pointerup', pointerAt(60, 60, { buttons: 0 }));
    expect(strokes).toHaveLength(2);
    engine.history.undo();
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[1]!.beforeImageData)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.lifecycle.dispose();
  });
});

// ---- commitStructural: UI-initiated structural edits on the canvas history ----

describe('commitStructural', () => {
  const forward: EngineTestAction = { id: 'a', type: 'setCanvasSelectedLayer' };
  const inverse: EngineTestAction = { id: null, type: 'setCanvasSelectedLayer' };

  it('dispatches forward immediately and records a reversible history entry', () => {
    const { store } = createFakeStore(makeDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(engine.layers.canCommitStructural()).toBe(true);
    expect(engine.layers.commitStructural('Select layer', forward, inverse)).toBe(true);
    // Forward dispatched once; the edit is now undoable but not redoable.
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenNthCalledWith(1, forward);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    // Undo dispatches the inverse and flips the stacks.
    engine.history.undo();
    expect(dispatch).toHaveBeenNthCalledWith(2, inverse);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    // Redo re-dispatches the forward.
    engine.history.redo();
    expect(dispatch).toHaveBeenNthCalledWith(3, forward);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.lifecycle.dispose();
    expect(engine.layers.canCommitStructural()).toBe(false);
    expect(engine.layers.commitStructural('Select layer', forward, inverse)).toBe(false);
  });
});

// ---- drawLayerThumbnail: cache-backed layer previews --------------------

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
      adjustments: { brightness: 0.25, contrast: -0.1, saturation: 0.3 },
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

const createSeededThumbnailBackend = (rgba: [number, number, number, number]): StubRasterBackend => {
  const backend = createTestStubRasterBackend();
  const createSurface = backend.createSurface.bind(backend);
  backend.createSurface = (width, height) => {
    const surface = createSurface(width, height);
    const originalCtx = surface.ctx;
    const ctx = new Proxy(originalCtx, {
      get(target, property, receiver) {
        if (property === 'getImageData') {
          return (x: number, y: number, imageWidth: number, imageHeight: number): ImageData => {
            target.getImageData(x, y, imageWidth, imageHeight);
            const data = new Uint8ClampedArray(imageWidth * imageHeight * 4);
            for (let index = 0; index < data.length; index += 4) {
              data.set(rgba, index);
            }
            return { colorSpace: 'srgb', data, height: imageHeight, width: imageWidth } as ImageData;
          };
        }
        return Reflect.get(target, property, receiver);
      },
    });
    Object.defineProperty(surface, 'ctx', { value: ctx });
    return surface;
  };
  return backend;
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
  it('returns false and draws nothing when the layer has no cache', () => {
    const { engine } = createEngine();
    const { calls, target } = createThumbnailTarget();
    expect(engine.previews.drawLayerThumbnail('missing', target, 96)).toBe(false);
    expect(calls).toHaveLength(0);
    engine.lifecycle.dispose();
  });

  it('scales the layer cache into the target and reports success', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(makeDoc()); // one 10x10 image layer 'a'
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    // One frame creates the layer cache entry (10x10).
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const { calls, target } = createThumbnailTarget();
    expect(engine.previews.drawLayerThumbnail('a', target, 96)).toBe(true);
    // 10x10 never upscales, so the target keeps the source dimensions.
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

  it('draws adjusted raster pixels over the checkerboard', async () => {
    const layer = {
      ...rasterLayer('a'),
      adjustments: { brightness: 0.2, contrast: -0.1, saturation: 0.3 },
      opacity: 0.4,
    } as CanvasRasterLayerContractV2;
    const backend = createSeededThumbnailBackend([10, 20, 30, 255]);
    const createSurface = vi.spyOn(backend, 'createSurface');
    const { store } = createReactiveStore({ ...makeDoc(), layers: [layer] });
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    await engine.previews.requestLayerThumbnail('a');
    const surfaceCountBeforeDraw = createSurface.mock.calls.length;
    const adjustedGetCountBeforeDraw = adjustedSurfaceCacheGets.length;

    const { calls, target } = createThumbnailTarget();
    expect(engine.previews.drawLayerThumbnail('a', target, 96)).toBe(true);
    expect(adjustedSurfaceCacheGets).toHaveLength(adjustedGetCountBeforeDraw);
    expect(createSurface.mock.calls.length).toBeGreaterThan(surfaceCountBeforeDraw);
    const createdSurfaces = createSurface.mock.results
      .slice(surfaceCountBeforeDraw)
      .map((result) => result.value as StubRasterSurface);
    expect(createdSurfaces.some((surface) => surface.callLog.some((call) => call.op === 'getImageData'))).toBe(true);
    const adjustedPixels = createdSurfaces
      .flatMap((surface) => surface.callLog)
      .find((call) => call.op === 'putImageData')?.args[0] as ImageData;
    expect(Array.from(adjustedPixels.data.slice(0, 4))).toEqual([66, 77, 89, 255]);
    const operations = calls.map((call) => call.op);
    expect(operations.indexOf('fillRect')).toBeLessThan(operations.lastIndexOf('drawImage'));
    expect(calls).toContainEqual({ args: [0.4], op: 'globalAlpha' });
    engine.lifecycle.dispose();
  });

  it.each([
    ['control transparency', { ...controlLayerForThumbnail('a'), withTransparencyEffect: true }, 'getImageData'],
    ['inpaint mask fill', thumbnailMaskLayer('a', 'inpaint_mask'), 'source-in'],
    ['regional mask fill', thumbnailMaskLayer('a', 'regional_guidance'), 'source-in'],
  ])('renders %s pixels over the checkerboard', async (_name, layer, expectedEffect) => {
    const backend =
      expectedEffect === 'getImageData'
        ? createSeededThumbnailBackend([100, 50, 200, 255])
        : createTestStubRasterBackend();
    const createSurface = vi.spyOn(backend, 'createSurface');
    const { store } = createReactiveStore({ ...makeDoc(), layers: [layer as CanvasLayerContract] });
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    await engine.previews.requestLayerThumbnail('a');
    const surfaceCountBeforeDraw = createSurface.mock.calls.length;

    const { calls, target } = createThumbnailTarget();
    expect(engine.previews.drawLayerThumbnail('a', target, 96)).toBe(true);
    expect(createSurface.mock.calls.length).toBeGreaterThan(surfaceCountBeforeDraw);
    const createdSurfaces = createSurface.mock.results
      .slice(surfaceCountBeforeDraw)
      .map((result) => result.value as StubRasterSurface);
    expect(
      createdSurfaces.some((surface) =>
        surface.callLog.some(
          (call) => call.op === expectedEffect || (call.op === 'set' && call.args[1] === expectedEffect)
        )
      )
    ).toBe(true);
    if (expectedEffect === 'getImageData') {
      const effectedPixels = createdSurfaces
        .flatMap((surface) => surface.callLog)
        .find((call) => call.op === 'putImageData')?.args[0] as ImageData;
      expect(Array.from(effectedPixels.data.slice(0, 4))).toEqual([100, 50, 200, 125]);
    } else {
      expect(
        createdSurfaces.some((surface) =>
          surface.callLog.some(
            (call) =>
              call.op === 'set' &&
              (call.args[0] === 'fillStyle' || call.args[0] === 'strokeStyle') &&
              call.args[1] === '#e07575'
          )
        )
      ).toBe(true);
    }
    expect(calls.map((call) => call.op)).toContain('fillRect');
    expect(calls.at(-1)?.op).toBe('drawImage');
    engine.lifecycle.dispose();
  });

  it.each(['raster', 'control', 'inpaint_mask', 'regional_guidance'] as const)(
    'bounds every %s thumbnail effect allocation before processing a large source',
    async (kind) => {
      const layer = largeThumbnailLayer(kind);
      const backend = createTestStubRasterBackend();
      const createSurface = vi.spyOn(backend, 'createSurface');
      const { store } = createReactiveStore({ ...makeDoc(), layers: [layer] });
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

  it.each([
    [
      'raster adjustments',
      rasterLayer('a'),
      (layer: CanvasLayerContract) => ({
        ...layer,
        adjustments: { brightness: 0.25, contrast: 0, saturation: 0 },
      }),
    ],
    [
      'control transparency',
      controlLayerForThumbnail('a'),
      (layer: CanvasLayerContract) => ({ ...layer, withTransparencyEffect: false }),
    ],
    [
      'inpaint mask fill',
      thumbnailMaskLayer('a', 'inpaint_mask'),
      (layer: CanvasLayerContract) => ({
        ...layer,
        mask: { ...('mask' in layer ? layer.mask : {}), fill: { color: '#00ff00', style: 'solid' } },
      }),
    ],
    [
      'regional mask fill',
      thumbnailMaskLayer('a', 'regional_guidance'),
      (layer: CanvasLayerContract) => ({
        ...layer,
        mask: { ...('mask' in layer ? layer.mask : {}), fill: { color: '#00ff00', style: 'solid' } },
      }),
    ],
    ['opacity', rasterLayer('a'), (layer: CanvasLayerContract) => ({ ...layer, opacity: 0.5 })],
  ])(
    'invalidates the keyed thumbnail version when %s changes without changing the source',
    async (_name, layer, edit) => {
      const doc = { ...makeDoc(), layers: [layer as CanvasLayerContract] };
      const { setDocument, store } = createReactiveStore(doc);
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        imageResolver: () => Promise.resolve(new Blob()),
        projectId: 'p1',
        store,
      });
      await engine.previews.requestLayerThumbnail('a');
      const listener = vi.fn();
      const unsubscribe = engine.stores.thumbnailVersion.subscribeKey('a', listener);

      setDocument({ ...doc, layers: [edit(layer as CanvasLayerContract) as CanvasLayerContract] });

      expect(listener).toHaveBeenCalledTimes(1);
      unsubscribe();
      engine.lifecycle.dispose();
    }
  );

  it('does not let display invalidation suppress the next cache-version publication', async () => {
    const layer = rasterLayer('a');
    const doc = { ...makeDoc(), layers: [layer] };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    await engine.previews.requestLayerThumbnail('a');
    const listener = vi.fn();
    const unsubscribe = engine.stores.thumbnailVersion.subscribeKey('a', listener);

    const adjusted = {
      ...layer,
      adjustments: { brightness: 0.25, contrast: 0, saturation: 0 },
    } as CanvasLayerContract;
    setDocument({ ...doc, layers: [adjusted] });
    setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'a-v2' })] });
    await engine.previews.requestLayerThumbnail('a');

    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
    engine.lifecycle.dispose();
  });

  it('refuses a newly allocated cache until pixels have been published', () => {
    const pending = createDeferred<Blob>();
    const { store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });
    const { calls, target } = createThumbnailTarget();

    void engine.previews.requestLayerThumbnail('a');

    expect(engine.previews.drawLayerThumbnail('a', target, 96)).toBe(false);
    expect(calls).toHaveLength(0);
    engine.lifecycle.dispose();
  });
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

  it('rasterizes disabled layers on explicit request', async () => {
    const layer = { ...makeDoc().layers[0]!, isEnabled: false };
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const { store } = createReactiveStore({ ...makeDoc(), layers: [layer] });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    expect(resolver).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases a detached thumbnail bitmap when its layer source is replaced', async () => {
    const bitmap = recordingBitmap('detached-replaced');
    const backend = createTestStubRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
    const doc = { ...makeDoc(), layers: [rasterLayer('a', { imageName: 'old' })] };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'new' })] });

    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases a detached thumbnail bitmap when its layer is deleted', async () => {
    const bitmap = recordingBitmap('detached-deleted');
    const backend = createTestStubRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
    const doc = { ...makeDoc(), layers: [rasterLayer('a', { imageName: 'old' })] };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    setDocument({ ...doc, layers: [] });

    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases each sequential shared-source decode after its thumbnail consumer settles', async () => {
    const bitmap = recordingBitmap('shared-deleted');
    const backend = createTestStubRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
    const first = rasterLayer('a', { imageName: 'shared' });
    const second = rasterLayer('b', { imageName: 'shared' });
    const doc = { ...makeDoc(), layers: [first, second] };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    expect(await engine.previews.requestLayerThumbnail('b')).toBe('ready');
    expect(bitmap.close).toHaveBeenCalledTimes(2);

    setDocument({ ...doc, layers: [second] });
    setDocument({ ...doc, layers: [] });

    expect(bitmap.close).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('does not retain settled shared-source decodes until layers change source', async () => {
    const bitmap = recordingBitmap('shared-replaced');
    const backend = createTestStubRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
    const first = rasterLayer('a', { imageName: 'shared' });
    const second = rasterLayer('b', { imageName: 'shared' });
    const doc = { ...makeDoc(), layers: [first, second] };
    const { setDocument, store } = createReactiveStore(doc);
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect(await engine.previews.requestLayerThumbnail('a')).toBe('ready');
    expect(await engine.previews.requestLayerThumbnail('b')).toBe('ready');
    expect(bitmap.close).toHaveBeenCalledTimes(2);

    const firstChanged = rasterLayer('a', { imageName: 'first-new' });
    setDocument({ ...doc, layers: [firstChanged, second] });
    setDocument({ ...doc, layers: [firstChanged, rasterLayer('b', { imageName: 'second-new' })] });

    expect(bitmap.close).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('releases a decoded bitmap when invalidation lands after cache insertion but before publication', async () => {
    const bitmap = recordingBitmap('stale-before-publication');
    const original = rasterLayer('a', { imageName: 'old' });
    const doc = { ...makeDoc(), layers: [original] };
    const { setDocument, store } = createReactiveStore(doc);
    const backend = createInvalidateDuringBitmapDrawBackend(bitmap, () => {
      setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'new' })] });
    });
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('stale');
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases a stale decode immediately even when another layer references its source', async () => {
    const bitmap = recordingBitmap('stale-shared-before-publication');
    const first = rasterLayer('a', { imageName: 'shared' });
    const second = rasterLayer('b', { imageName: 'shared' });
    const doc = { ...makeDoc(), layers: [first, second] };
    const { setDocument, store } = createReactiveStore(doc);
    const backend = createInvalidateDuringBitmapDrawBackend(bitmap, () => {
      setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'new' }), second] });
    });
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('stale');
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'new' })] });
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases a stale decode without waiting for a matching paint source to change', async () => {
    const bitmap = recordingBitmap('stale-shared-paint');
    const first = rasterLayer('a', { imageName: 'shared' });
    const second: CanvasRasterLayerContractV2 = {
      ...(rasterLayer('b') as CanvasRasterLayerContractV2),
      source: { bitmap: { height: 10, imageName: 'shared', width: 10 }, type: 'paint' },
    };
    const doc = { ...makeDoc(), layers: [first, second] };
    const { setDocument, store } = createReactiveStore(doc);
    const backend = createInvalidateDuringBitmapDrawBackend(bitmap, () => {
      setDocument({ ...doc, layers: [second] });
    });
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('stale');
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    setDocument({
      ...doc,
      layers: [{ ...second, source: { bitmap: { height: 10, imageName: 'new', width: 10 }, type: 'paint' } }],
    });
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases a stale decode without waiting for a matching mask bitmap to change', async () => {
    const bitmap = recordingBitmap('stale-shared-mask');
    const first = rasterLayer('a', { imageName: 'shared' });
    const second: CanvasInpaintMaskLayerContract = {
      ...maskLayer('b'),
      mask: {
        ...maskLayer('b').mask,
        bitmap: { height: 10, imageName: 'shared', width: 10 },
      },
    };
    const doc = { ...makeDoc(), layers: [first, second] };
    const { setDocument, store } = createReactiveStore(doc);
    const backend = createInvalidateDuringBitmapDrawBackend(bitmap, () => {
      setDocument({ ...doc, layers: [second] });
    });
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    expect(await engine.previews.requestLayerThumbnail('a')).toBe('stale');
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    setDocument({
      ...doc,
      layers: [
        {
          ...second,
          mask: { ...second.mask, bitmap: { height: 10, imageName: 'new', width: 10 } },
        },
      ],
    });
    expect(bitmap.close).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });

  it('releases a finished unpublished decode while another request is still in flight', async () => {
    const bitmap = recordingBitmap('stale-shared-in-flight');
    const secondResolve = createDeferred<Blob>();
    let resolveCount = 0;
    const resolver = vi.fn(() => {
      resolveCount += 1;
      return resolveCount === 1 ? Promise.resolve(new Blob()) : secondResolve.promise;
    });
    const first = rasterLayer('a', { imageName: 'shared' });
    const second = rasterLayer('b', { imageName: 'shared' });
    const doc = { ...makeDoc(), layers: [first, second] };
    const { setDocument, store } = createReactiveStore(doc);
    const backend = createInvalidateDuringBitmapDrawBackend(bitmap, () => {
      setDocument({
        ...doc,
        layers: [rasterLayer('a', { imageName: 'first-new' }), rasterLayer('b', { imageName: 'second-new' })],
      });
    });
    const engine = createCanvasEngine({ backend, imageResolver: resolver, projectId: 'p1', store });

    const firstRequest = engine.previews.requestLayerThumbnail('a');
    const secondRequest = engine.previews.requestLayerThumbnail('b');
    expect(await firstRequest).toBe('stale');
    expect(bitmap.close).toHaveBeenCalledTimes(1);

    secondResolve.resolve(new Blob());
    expect(await secondRequest).toBe('stale');
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
    setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'a-v2' })] });
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
        context: expect.objectContaining({ error: 'decode failed', layerId: 'a' }),
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
    setDocument({ ...doc, layers: [] });
    expect(requests.get('a')?.signal?.aborted).toBe(true);
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();

    expect(await request).toBe('stale');
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();
    expect(engine.previews.drawLayerThumbnail('a', createThumbnailTarget().target, 96)).toBe(false);
    engine.lifecycle.dispose();
  });

  it('clears state when the document is replaced', async () => {
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
    setDocument({ ...doc, width: doc.width + 1 }, 1);

    expect(requests.get('a')?.signal?.aborted).toBe(true);
    expect(await request).toBe('stale');
    expect(engine.stores.thumbnailStatus.get('a')).toBeUndefined();
    engine.lifecycle.dispose();
  });

  it('keeps an in-flight thumbnail request after another project becomes active', async () => {
    const pending = createDeferred<Blob>();
    let signal: AbortSignal | undefined;
    const resolver = vi.fn((_imageName: string, nextSignal?: AbortSignal) => {
      signal = nextSignal;
      return pending.promise;
    });
    const { setActiveProjectId, store } = createReactiveStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });

    const request = engine.previews.requestLayerThumbnail('a');
    setActiveProjectId('p2');
    expect(signal?.aborted).toBe(false);
    expect(engine.stores.thumbnailStatus.get('a')).toBe('loading');

    pending.resolve(new Blob());
    expect(await request).toBe('ready');
    expect(engine.stores.thumbnailStatus.get('a')).toBe('ready');
    expect(engine.previews.drawLayerThumbnail('a', createThumbnailTarget().target, 96)).toBe(true);
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

// ---- mergeLayerDown: composites the upper cache into the below local space ----

const twoPaintDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [
    {
      blendMode: 'multiply',
      id: 'upper',
      isEnabled: true,
      isLocked: false,
      name: 'upper',
      opacity: 0.5,
      // Content-sized: a persisted bitmap gives the cache a non-empty content rect.
      // The upper (40×40) sits fully within the below (60×60) in below-local space,
      // so the merge union stays 60×60 and the warped upper transform is non-trivial.
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
      // Offset so the merge matrix is non-trivial (below-local origin shifts).
      source: { bitmap: { height: 60, imageName: 'below-bmp', width: 60 }, offset: { x: 0, y: 0 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
      type: 'raster',
    },
  ],
  selectedLayerId: 'upper',
  version: 2,
  width: 100,
});

/** An empty inpaint mask layer, for the merge-down mask-rejection tests below. */
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
    // One frame builds both layer caches; await the async bitmap decode so both
    // caches are READY (merge refuses stale/in-flight caches — finding 20).
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const surfacesBeforeMerge = surfaces.length;
    expect(engine.layers.mergeLayerDown('upper')).toBe(true);

    // The reducer is asked to collapse the two layers into a paint layer.
    const mergeCall = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'mergeCanvasLayersDown');
    expect(mergeCall).toEqual({
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
      type: 'mergeCanvasLayersDown',
      upperLayerId: 'upper',
    });

    // A fresh union-sized surface was allocated for the merged pixels. Content-
    // sized: below's rect {0,0,60,60} unioned with upper's rect warped into
    // below-local space {20,20,40,40} is {0,0,60,60}, so the merged surface stays
    // 60×60 with origin (0,0).
    const merged = surfaces[surfacesBeforeMerge];
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

    // Below is blitted before the upper transform; the upper after it.
    const drawIndices = log.reduce<number[]>((acc, entry, index) => {
      if (entry.op === 'drawImage') {
        acc.push(index);
      }
      return acc;
    }, []);
    expect(drawIndices).toHaveLength(2);
    expect(drawIndices[0]).toBeLessThan(matrixIndex);
    expect(drawIndices[1]).toBeGreaterThan(matrixIndex);

    // The upper layer's opacity and blend mode are baked into the merge.
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

  it('is a no-op for the bottom-most layer (nothing below)', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(twoPaintDoc());
    const dispatch = store.dispatch as Mock;
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

    expect(engine.layers.mergeLayerDown('below')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );

    engine.lifecycle.dispose();
  });

  it.each(['upper', 'below'] as const)(
    "is a no-op when the %s layer is locked (merge is not undoable; must mirror the paint tool's locked-target refusal)",
    (lockedId) => {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

      const doc = twoPaintDoc();
      const locked = {
        ...doc,
        layers: doc.layers.map((layer) => (layer.id === lockedId ? { ...layer, isLocked: true } : layer)),
      };
      const { store } = createReactiveStore(locked);
      const dispatch = store.dispatch as Mock;
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

      expect(engine.layers.mergeLayerDown('upper')).toBe(false);
      expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
        false
      );

      engine.lifecycle.dispose();
    }
  );

  // A two-paint doc where exactly ONE layer is content-empty (bitmap: null → a 0×0
  // cache surface). Merging must NOT `drawImage` the zero-dimension operand (which
  // throws in browsers) but must still dispatch the collapse and composite the
  // non-empty operand.
  const oneEmptyPaintDoc = (emptyId: 'upper' | 'below'): CanvasDocumentContractV2 => {
    const doc = twoPaintDoc();
    return {
      ...doc,
      layers: doc.layers.map((layer) =>
        layer.id === emptyId ? { ...layer, source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' } } : layer
      ),
    };
  };

  it.each([
    { emptyId: 'upper', keptId: 'below' },
    { emptyId: 'below', keptId: 'upper' },
  ] as const)(
    'merges when only the $emptyId layer is empty: dispatches, and never draws a 0×0 surface',
    async ({ emptyId }) => {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

      const { store } = createReactiveStore(oneEmptyPaintDoc(emptyId));
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
      // Await the kept layer's bitmap decode so its cache is READY before merge
      // (the empty operand needs no decode; merge refuses stale/in-flight caches).
      raf.flush();
      await flushMicrotasks();
      raf.flush();

      const surfacesBeforeMerge = surfaces.length;
      expect(engine.layers.mergeLayerDown('upper')).toBe(true);

      // Single-dispatch collapse still happens (undo semantics unchanged).
      expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
        true
      );

      // The merged surface only composites the NON-empty operand: exactly one
      // drawImage, and it never draws a zero-dimension source canvas.
      const merged = surfaces[surfacesBeforeMerge];
      expect(merged).toBeDefined();
      const draws = merged!.callLog.filter((entry) => entry.op === 'drawImage');
      expect(draws).toHaveLength(1);
      for (const draw of draws) {
        const src = draw.args[0] as { width: number; height: number };
        expect(src.width).toBeGreaterThan(0);
        expect(src.height).toBeGreaterThan(0);
      }

      engine.lifecycle.dispose();
    }
  );

  // A both-empty pair must fold trivially (delete the upper, below stays empty)
  // rather than silently no-op — otherwise merge-visible stalls on such a run (F4).
  it('folds a both-empty pair trivially: dispatches the collapse and allocates no merged surface (F4)', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const base = twoPaintDoc();
    const doc: CanvasDocumentContractV2 = {
      ...base,
      layers: base.layers.map((layer) => ({
        ...layer,
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' as const },
      })),
    };
    const { store } = createReactiveStore(doc);
    const dispatch = store.dispatch as Mock;
    const backendBase = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend = {
      ...backendBase,
      createSurface: (w: number, h: number): StubRasterSurface => {
        const surface = backendBase.createSurface(w, h);
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
    expect(engine.layers.mergeLayerDown('upper')).toBe(true);

    // The collapse still dispatches (the upper layer is removed).
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
      true
    );
    // No merged surface was allocated: a 0×0 union surface would throw, and there
    // are no pixels to composite.
    expect(surfaces.length).toBe(surfacesBeforeMerge);

    engine.lifecycle.dispose();
  });

  // Regression: `mergeLayerDown` used to gate on `isRenderableLayer`, which masks
  // satisfy (a mask rasterizes to a stencil whenever enabled). That let a mask
  // reach the merge path, whose reducer unconditionally produces a `type: 'raster'`
  // result — merging a mask blitted its stencil into the layer below and/or
  // clobbered a mask below into a raster layer, destroying its config, with no
  // undo. The guard must reject a mask on EITHER side, mirroring the layers
  // panel's `isMergeableRasterLayer`/`canMergeLayerDown` enablement exactly.
  it.each([
    { label: 'mask above a raster layer', maskId: 'upper' as const },
    { label: 'a raster layer above a mask', maskId: 'below' as const },
  ])('is a no-op for $label (document unchanged, no dispatch)', ({ maskId }) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const base = twoPaintDoc();
    const doc: CanvasDocumentContractV2 = {
      ...base,
      layers: base.layers.map((layer) => (layer.id === maskId ? maskLayer(maskId) : layer)),
    };
    const { store } = createReactiveStore(doc);
    const dispatch = store.dispatch as Mock;
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

    expect(engine.layers.mergeLayerDown('upper')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );
    // Document unchanged: still two layers, each with its original type — no
    // mask was blitted into, and no mask was clobbered into a raster layer.
    expect(engine.document.getDocument()!.layers.map((l) => l.type)).toEqual(doc.layers.map((l) => l.type));

    engine.lifecycle.dispose();
  });

  it('is a no-op for mask-above-mask (document unchanged, no dispatch)', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const base = twoPaintDoc();
    const doc: CanvasDocumentContractV2 = {
      ...base,
      layers: [maskLayer('upper'), maskLayer('below')],
    };
    const { store } = createReactiveStore(doc);
    const dispatch = store.dispatch as Mock;
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

    expect(engine.layers.mergeLayerDown('upper')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );
    expect(engine.document.getDocument()!.layers.map((l) => l.type)).toEqual(['inpaint_mask', 'inpaint_mask']);

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
    const result = merged.layers.find((layer) => layer.id !== 'upper' && layer.id !== 'below');
    expect(result).toMatchObject({
      isEnabled: true,
      source: { bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(merged.layers.find((layer) => layer.id === 'upper')?.isEnabled).toBe(false);
    expect(merged.layers.find((layer) => layer.id === 'below')?.isEnabled).toBe(false);

    engine.history.undo();
    expect(engine.document.getDocument()!.layers.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['upper', true],
      ['below', true],
    ]);

    engine.history.redo();
    expect(engine.document.getDocument()!.layers.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      [result!.id, true],
      ['upper', false],
      ['below', false],
    ]);
    engine.lifecycle.dispose();
  });

  it('publishes a boolean merge into its bound project after another project becomes active', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { setActiveProjectId, store } = createReactiveStore(twoPaintDoc());
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

    const merge = engine.layers.booleanMergeRasterLayers('upper', 'intersect');
    setActiveProjectId('p2');

    expect(await merge).toBe('merged');
    expect(store.dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: 'applyCanvasLayerStackMutation' }));
    engine.lifecycle.dispose();
  });

  it('refuses the operation until both raster caches are ready', async () => {
    const { engine, raf } = setup();
    raf.flush();

    expect(await engine.layers.booleanMergeRasterLayers('upper', 'intersect')).toBe('not-ready');
    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'below']);
    engine.lifecycle.dispose();
  });

  it('rejects unsupported and missing layer pairs without modifying the document', async () => {
    const doc = twoPaintDoc();
    doc.layers[1] = maskLayer('below');
    const { projectId, store } = createReducerBackedStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.layers.booleanMergeRasterLayers('upper', 'exclude')).toBe('unsupported');
    expect(await engine.layers.booleanMergeRasterLayers('missing', 'exclude')).toBe('missing');
    expect(engine.document.getDocument()!.layers).toEqual(doc.layers);
    engine.lifecycle.dispose();
  });

  it('merges two live unflushed paint caches whose contract bitmaps are still null', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const emptyPaint = (id: string): CanvasRasterLayerContractV2 => ({
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
    const upper = emptyPaint('upper');
    const below = emptyPaint('below');
    const doc = { ...makeDoc(), layers: [upper, below], selectedLayerId: 'upper' };
    const { projectId, store } = createReducerBackedStore(doc);
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
    engine.tools.setTool('brush');
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    overlay.fire('pointerdown', pointerAt(10, 10));
    overlay.fire('pointermove', pointerAt(30, 30));
    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    store.dispatch({ id: 'below', type: 'setCanvasSelectedLayer' });
    overlay.fire('pointerdown', pointerAt(15, 15));
    overlay.fire('pointermove', pointerAt(35, 35));
    overlay.fire('pointerup', pointerAt(35, 35, { buttons: 0 }));

    expect(await engine.layers.booleanMergeRasterLayers('upper', 'intersect')).toBe('merged');
    engine.lifecycle.dispose();
  });
});

describe('extract masked canvas area', () => {
  const maskedDoc = (): CanvasDocumentContractV2 => {
    const doc = twoPaintDoc();
    return {
      ...doc,
      layers: [
        {
          ...maskLayer('mask'),
          mask: {
            bitmap: { height: 20, imageName: 'mask-bitmap', width: 20 },
            fill: { color: '#e07575', style: 'diagonal' },
            offset: { x: 15, y: 25 },
          },
        },
        ...doc.layers,
      ],
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

    const extracted = engine.document.getDocument()!.layers.find((layer) => layer.id === result.layerId);
    expect(extracted).toMatchObject({
      isEnabled: true,
      source: { bitmap: null, offset: { x: 15, y: 25 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(engine.document.getDocument()!.layers.find((layer) => layer.id === 'mask')).toEqual(maskedDoc().layers[0]);
    expect(engine.document.getDocument()!.layers.find((layer) => layer.id === 'upper')?.isEnabled).toBe(true);
    expect(engine.document.getDocument()!.layers.find((layer) => layer.id === 'below')?.isEnabled).toBe(true);

    engine.history.undo();
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(false);
    engine.history.redo();
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(true);
    engine.lifecycle.dispose();
  });

  it('extracts enabled raster layers but never control layers', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const raster = rasterLayer('raster');
    const control: CanvasLayerContract = {
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'normal',
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control',
      opacity: 1,
      source: { image: { height: 10, imageName: 'control', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: false,
    };
    const mask: CanvasInpaintMaskLayerContract = {
      ...maskLayer('mask'),
      mask: {
        ...maskLayer('mask').mask,
        bitmap: { height: 10, imageName: 'mask', width: 10 },
      },
    };
    const document = { ...makeDoc(), layers: [mask, raster, control], selectedLayerId: 'mask' };
    const { projectId, store } = createReducerBackedStore(document);
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
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const rasterExport = await engine.exports.exportLayerPixels(raster.id);
    const controlExport = await engine.exports.exportLayerPixels(control.id);
    expect(rasterExport.status).toBe('ok');
    expect(controlExport.status).toBe('ok');
    if (rasterExport.status !== 'ok' || controlExport.status !== 'ok') {
      throw new Error('fixture layers did not rasterize');
    }

    expect((await engine.exports.extractMaskedArea(mask.id)).status).toBe('extracted');

    const extractedPixels = surfaces.find((surface) =>
      surface.callLog.some(
        (entry) =>
          entry.op === 'set' && entry.args[0] === 'globalCompositeOperation' && entry.args[1] === 'destination-in'
      )
    );
    const compositeSources = extractedPixels?.callLog
      .filter((entry) => entry.op === 'drawImage')
      .map((entry) => entry.args[0]);
    expect(compositeSources).toContain(rasterExport.surface.canvas);
    expect(compositeSources).not.toContain(controlExport.surface.canvas);
    engine.lifecycle.dispose();
  });

  it('skips enabled raster layers that have no persisted or live pixels', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const document = maskedDoc();
    const blank: CanvasRasterLayerContractV2 = {
      blendMode: 'normal',
      id: 'blank',
      isEnabled: true,
      isLocked: false,
      name: 'Blank',
      opacity: 1,
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    document.layers.splice(2, 0, blank);
    const { projectId, store } = createReducerBackedStore(document);
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

    expect((await engine.exports.extractMaskedArea('mask')).status).toBe('extracted');
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
      layers: document.layers.map((layer) =>
        layer.id === 'upper' && layer.type === 'raster'
          ? {
              ...layer,
              source: {
                bitmap: { height: 40, imageName: 'upper-v2', width: 40 },
                offset: { x: 0, y: 0 },
                type: 'paint',
              },
            }
          : layer
      ),
    });

    expect(await extraction).toEqual({ status: 'not-ready' });
    expect(store.dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('refuses extraction from a locked mask', async () => {
    const document = maskedDoc();
    document.layers[0] = { ...document.layers[0]!, isLocked: true };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });

    expect(await engine.exports.extractMaskedArea('mask')).toEqual({ status: 'unsupported' });
    engine.lifecycle.dispose();
  });

  it('refuses extraction until the mask and contributor caches are ready', async () => {
    const { engine, raf } = setup();
    raf.flush();

    expect(await engine.exports.extractMaskedArea('mask')).toEqual({ status: 'not-ready' });
    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual(['mask', 'upper', 'below']);
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
    expect(engine.document.getDocument()!.layers).toEqual(doc.layers);
    engine.lifecycle.dispose();

    const emptyDoc = maskedDoc();
    emptyDoc.layers[0] = maskLayer('mask');
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

// ---- mergeVisibleRasterLayers: guarded non-destructive composite ----

/**
 * An EngineStore backed by the REAL workbench reducer, so the operation's
 * prepared stack mutation advances the document and mirror atomically — the
 * no-op mock store cannot exercise that transaction.
 */
const createReducerBackedStore = (
  document: CanvasDocumentContractV2,
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

  it('commits through the project-bound engine and preserves layer identity and staging semantics across undo/redo', () => {
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
    const acceptedLayer = projectAfterCommit.canvas.document.layers[0]!;
    const acceptedEvent = projectAfterCommit.events[0]!;
    expect(acceptedLayer).toMatchObject({
      id: result.layerId,
      opacity: 0.75,
      source: { image: { imageName: 'staged-result.png' }, type: 'image' },
      transform: { scaleX: 2, scaleY: 2, x: 7, y: 9 },
    });
    expect(projectAfterCommit.canvas.stagingArea.pendingImages).toEqual([]);
    expect(acceptedEvent.type).toBe('canvas-layer-accepted');

    engine.history.undo();
    const projectAfterUndo = reducer.store.getState().projects[0]!;
    expect(projectAfterUndo.canvas.document.layers).not.toContainEqual(expect.objectContaining({ id: result.layerId }));
    expect(projectAfterUndo.canvas.document.selectedLayerId).toBe('a');
    expect(projectAfterUndo.canvas.stagingArea.pendingImages).toEqual([]);

    engine.history.redo();
    const projectAfterRedo = reducer.store.getState().projects[0]!;
    expect(projectAfterRedo.canvas.document.layers[0]).toBe(acceptedLayer);
    expect(projectAfterRedo.canvas.stagingArea.pendingImages).toEqual([]);
    expect(projectAfterRedo.events).toContain(acceptedEvent);
    expect(projectAfterRedo.events.filter((event) => event.id === acceptedEvent.id)).toHaveLength(1);

    engine.history.undo();
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

  it('returns stale without changing staging or history when the selected candidate key changes', () => {
    const reducer = createReducerBackedStore(makeDoc());
    const first = stagedCandidate();
    const second = { ...stagedCandidate(), imageName: 'new-selection.png' };
    reducer.store.dispatch({ candidate: first, projectId: reducer.projectId, type: 'appendCanvasStagingCandidate' });
    reducer.store.dispatch({ candidate: second, projectId: reducer.projectId, type: 'appendCanvasStagingCandidate' });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const before = reducer.store.getState().projects[0]!.canvas;

    expect(engine.layers.commitStagedImage(stagedSelection(first, 1))).toEqual({ status: 'stale' });
    expect(reducer.store.getState().projects[0]!.canvas).toBe(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('returns busy without changing staging or history while an edit lease is active', async () => {
    const reducer = createReducerBackedStore(makeDoc());
    reducer.store.dispatch({
      candidate: stagedCandidate(),
      projectId: reducer.projectId,
      type: 'appendCanvasStagingCandidate',
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const exported = await engine.exports.exportLayerPixels('a');
    if (exported.status !== 'ok') {
      throw new Error('expected current layer guard');
    }
    const operations = getCanvasOperations(engine);
    expect(
      operations.controller.start({
        cleanupPreview: vi.fn(),
        guard: exported.guard,
        identity: { kind: 'filter', layerId: 'a', projectId: reducer.projectId },
      })
    ).not.toBeNull();
    const before = reducer.store.getState().projects[0]!.canvas;

    expect(engine.layers.commitStagedImage(stagedSelection(stagedCandidate()))).toEqual({ status: 'busy' });
    expect(reducer.store.getState().projects[0]!.canvas).toBe(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    operations.controller.cancel();
    engine.lifecycle.dispose();
  });

  it('returns busy without changing staging or history while a pointer gesture is active', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const reducer = createReducerBackedStore(makeDoc());
    reducer.store.dispatch({
      candidate: stagedCandidate(),
      projectId: reducer.projectId,
      type: 'appendCanvasStagingCandidate',
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    engine.tools.setTool('view');
    overlay.fire('pointerdown', pointerAt(10, 10));
    const before = reducer.store.getState().projects[0]!.canvas;

    expect(engine.layers.commitStagedImage(stagedSelection(stagedCandidate()))).toEqual({ status: 'busy' });
    expect(reducer.store.getState().projects[0]!.canvas).toBe(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    overlay.fire('pointerup', pointerAt(10, 10, { buttons: 0 }));
    engine.lifecycle.dispose();
  });

  it('returns stale without changing staging or history when the reducer rejects acceptance', () => {
    const document = makeDoc();
    const canvas = makeCanvas(document);
    canvas.stagingArea = {
      ...canvas.stagingArea,
      isVisible: true,
      pendingImageIds: ['staged-result.png'],
      pendingImages: [stagedCandidate()],
    };
    const mutationPort: CanvasProjectMutationPort = {
      dispatch: () => false,
      getCanvasState: () => canvas,
      subscribe: () => () => undefined,
    };
    const { store } = createFakeStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort,
      projectId: 'p1',
      store,
    });

    expect(engine.layers.commitStagedImage(stagedSelection(stagedCandidate()))).toEqual({ status: 'stale' });
    expect(canvas.stagingArea.pendingImages).toEqual([stagedCandidate()]);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('returns stale without history when reducer state cannot converge into the document mirror', () => {
    const document = makeDoc();
    const canvas = makeCanvas(document);
    canvas.stagingArea = {
      ...canvas.stagingArea,
      isVisible: true,
      pendingImageIds: ['staged-result.png'],
      pendingImages: [stagedCandidate()],
    };
    let reducerCanvas = canvas;
    let committed = false;
    let postCommitReads = 0;
    const mutationPort: CanvasProjectMutationPort = {
      dispatch: (mutation) => {
        if (mutation.type !== 'commitStagedImage') {
          return false;
        }
        reducerCanvas = {
          ...canvas,
          document: { ...document, layers: [mutation.layer, ...document.layers], selectedLayerId: mutation.layer.id },
          stagingArea: { ...canvas.stagingArea, pendingImageIds: [], pendingImages: [] },
        };
        committed = true;
        return true;
      },
      getCanvasState: () => {
        if (!committed) {
          return canvas;
        }
        postCommitReads += 1;
        return postCommitReads === 1 ? reducerCanvas : canvas;
      },
      subscribe: () => () => undefined,
    };
    const { store } = createFakeStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort,
      projectId: 'p1',
      store,
    });

    expect(engine.layers.commitStagedImage(stagedSelection(stagedCandidate()))).toEqual({ status: 'stale' });
    expect(engine.document.getDocument()).toBe(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('returns missing without history for an absent staged candidate or project', () => {
    const reducer = createReducerBackedStore(makeDoc());
    const candidateMissingEngine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    expect(
      candidateMissingEngine.layers.commitStagedImage(
        stagedSelection({ ...stagedCandidate(), imageName: 'missing.png' })
      )
    ).toEqual({
      status: 'missing',
    });
    expect(candidateMissingEngine.stores.canUndo.get()).toBe(false);
    candidateMissingEngine.lifecycle.dispose();

    const document = makeDoc();
    const { store } = createFakeStore(document);
    const projectMissingEngine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: {
        dispatch: () => false,
        getCanvasState: () => null,
        subscribe: () => () => undefined,
      },
      projectId: 'missing-project',
      store,
    });
    expect(projectMissingEngine.layers.commitStagedImage(stagedSelection(stagedCandidate()))).toEqual({
      status: 'missing',
    });
    expect(projectMissingEngine.stores.canUndo.get()).toBe(false);
    projectMissingEngine.lifecycle.dispose();
  });
});

describe('structural raster publication failure atomicity', () => {
  const sentinelLayer = (): CanvasLayerContract => ({
    ...rasterLayer('selection-sentinel'),
    isEnabled: false,
  });

  const createFaultHarness = (document: CanvasDocumentContractV2) => {
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
    expectedDocument: CanvasDocumentContractV2
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
      const document: CanvasDocumentContractV2 = {
        ...makeDoc(),
        bbox: { height: 7, width: 6, x: 2, y: 3 },
        layers: [source, sentinelLayer()],
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

  it.each(structuralFaults)(
    'commitLayerCopy keeps document, cache, selection, and history exact when final cache %s fails',
    async (failure) => {
      const source = rasterLayer('copy-source');
      const document = { ...makeDoc(), layers: [source, sentinelLayer()], selectedLayerId: 'selection-sentinel' };
      const harness = createFaultHarness(document);
      const sourceCache = await snapshotLayerCache(harness.engine, source.id);
      const expectedDocument = harness.engine.document.getDocument()!;
      const copy = { ...structuredClone(source), id: 'copy-result', name: 'Copy result' };
      (harness.store.dispatch as Mock).mockClear();
      harness.bitmapStore.markLayerDirty.mockClear();
      armStructuralFault(harness.faults, failure, { allocation: 1, draw: 1 });

      expect(() => harness.engine.layers.commitLayerCopy('Copy layer', source.id, copy, 0)).toThrow(
        structuralFaultMessage(failure)
      );

      expectInitialFailureExact(harness, expectedDocument);
      expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === copy.id)).toBe(false);
      await expectLayerCacheExact(harness.engine, source.id, sourceCache);
      harness.engine.lifecycle.dispose();
    }
  );

  it.each(structuralFaults)(
    'commitLayerConversion keeps document, cache, selection, and history exact when final cache %s fails',
    async (failure) => {
      const source = rasterLayer('conversion-source');
      const document = { ...makeDoc(), layers: [source, sentinelLayer()], selectedLayerId: 'selection-sentinel' };
      const harness = createFaultHarness(document);
      const sourceCache = await snapshotLayerCache(harness.engine, source.id);
      const expectedDocument = harness.engine.document.getDocument()!;
      const live = harness.engine.document.getDocument()!.layers.find((layer) => layer.id === source.id)!;
      if (live.type !== 'raster') {
        throw new Error('expected a raster conversion source');
      }
      const converted: CanvasLayerContract = {
        ...structuredClone(live),
        adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
        type: 'control',
        withTransparencyEffect: false,
      };
      (harness.store.dispatch as Mock).mockClear();
      harness.bitmapStore.markLayerDirty.mockClear();
      armStructuralFault(harness.faults, failure, { allocation: 1, draw: 1 });

      expect(() => harness.engine.layers.commitLayerConversion('Convert layer', live, converted)).toThrow(
        structuralFaultMessage(failure)
      );

      expectInitialFailureExact(harness, expectedDocument);
      await expectLayerCacheExact(harness.engine, source.id, sourceCache);
      harness.engine.lifecycle.dispose();
    }
  );

  it.each(structuralFaults)(
    'copyLayerToRaster keeps document, cache, selection, and history exact when final cache %s fails',
    async (failure) => {
      const source = rasterLayer('raster-copy-source');
      const document = { ...makeDoc(), layers: [source, sentinelLayer()], selectedLayerId: 'selection-sentinel' };
      const harness = createFaultHarness(document);
      const sourceCache = await snapshotLayerCache(harness.engine, source.id);
      const expectedDocument = harness.engine.document.getDocument()!;
      (harness.store.dispatch as Mock).mockClear();
      harness.bitmapStore.markLayerDirty.mockClear();
      armStructuralFault(harness.faults, failure, { allocation: 1, draw: 1 });

      await expect(harness.engine.layers.copyLayerToRaster(source.id)).rejects.toThrow(structuralFaultMessage(failure));

      expectInitialFailureExact(harness, expectedDocument);
      await expectLayerCacheExact(harness.engine, source.id, sourceCache);
      harness.engine.lifecycle.dispose();
    }
  );

  it.each(structuralFaults)(
    'boolean merge keeps document, source caches, selection, and history exact when final cache %s fails',
    async (failure) => {
      const pair = twoPaintDoc();
      const document = {
        ...pair,
        layers: [...pair.layers, sentinelLayer()],
        selectedLayerId: 'selection-sentinel',
      };
      const harness = createFaultHarness(document);
      const upperCache = await snapshotLayerCache(harness.engine, 'upper');
      const belowCache = await snapshotLayerCache(harness.engine, 'below');
      const expectedDocument = harness.engine.document.getDocument()!;
      (harness.store.dispatch as Mock).mockClear();
      harness.bitmapStore.markLayerDirty.mockClear();
      armStructuralFault(harness.faults, failure, { allocation: 3, draw: 4 });

      await expect(harness.engine.layers.booleanMergeRasterLayers('upper', 'intersect')).rejects.toThrow(
        structuralFaultMessage(failure)
      );

      expectInitialFailureExact(harness, expectedDocument);
      await expectLayerCacheExact(harness.engine, 'upper', upperCache);
      await expectLayerCacheExact(harness.engine, 'below', belowCache);
      harness.engine.lifecycle.dispose();
    }
  );

  it.each(structuralFaults)(
    'masked extraction keeps document, source caches, selection, and history exact when final cache %s fails',
    async (failure) => {
      const pair = twoPaintDoc();
      const mask: CanvasInpaintMaskLayerContract = {
        ...maskLayer('atomic-mask'),
        mask: {
          bitmap: { height: 20, imageName: 'atomic-mask-bitmap', width: 20 },
          fill: { color: '#e07575', style: 'diagonal' },
          offset: { x: 15, y: 25 },
        },
      };
      const document = {
        ...pair,
        layers: [mask, ...pair.layers, sentinelLayer()],
        selectedLayerId: 'selection-sentinel',
      };
      const harness = createFaultHarness(document);
      const maskCache = await snapshotLayerCache(harness.engine, mask.id);
      const upperCache = await snapshotLayerCache(harness.engine, 'upper');
      const belowCache = await snapshotLayerCache(harness.engine, 'below');
      const expectedDocument = harness.engine.document.getDocument()!;
      (harness.store.dispatch as Mock).mockClear();
      harness.bitmapStore.markLayerDirty.mockClear();
      armStructuralFault(harness.faults, failure, { allocation: 2, draw: 4 });

      await expect(harness.engine.exports.extractMaskedArea(mask.id)).rejects.toThrow(structuralFaultMessage(failure));

      expectInitialFailureExact(harness, expectedDocument);
      await expectLayerCacheExact(harness.engine, mask.id, maskCache);
      await expectLayerCacheExact(harness.engine, 'upper', upperCache);
      await expectLayerCacheExact(harness.engine, 'below', belowCache);
      harness.engine.lifecycle.dispose();
    }
  );

  const createCropReplayHarness = async () => {
    const source = rasterLayer('crop-replay-source');
    const document: CanvasDocumentContractV2 = {
      ...makeDoc(),
      bbox: { height: 7, width: 6, x: 2, y: 3 },
      layers: [source, sentinelLayer()],
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

      expect(() => harness.engine.history.undo()).toThrow(structuralFaultMessage(failure));

      expect(harness.engine.document.getDocument()).toBe(committedDocument);
      expect(harness.engine.document.getDocument()).toEqual(committedDocument);
      expect(harness.engine.stores.canUndo.get()).toBe(true);
      expect(harness.engine.stores.canRedo.get()).toBe(false);
      await expectLayerCacheExact(harness.engine, harness.source.id, committedCache);
      harness.engine.history.undo();
      expect(harness.engine.document.getDocument()).toEqual(harness.originalDocument);
      harness.engine.lifecycle.dispose();
    }
  );

  it.each(structuralFaults)(
    'crop redo cache-%s failure leaves the restored state exact and retryable',
    async (failure) => {
      const harness = await createCropReplayHarness();
      const committedDocument = structuredClone(harness.engine.document.getDocument()!);
      harness.engine.history.undo();
      const restoredDocument = harness.engine.document.getDocument()!;
      const restoredCache = await snapshotLayerCache(harness.engine, harness.source.id);
      armStructuralFault(harness.faults, failure, { allocation: 0, draw: 0 });

      expect(() => harness.engine.history.redo()).toThrow(structuralFaultMessage(failure));

      expect(harness.engine.document.getDocument()).toBe(restoredDocument);
      expect(harness.engine.document.getDocument()).toEqual(harness.originalDocument);
      expect(harness.engine.stores.canUndo.get()).toBe(false);
      expect(harness.engine.stores.canRedo.get()).toBe(true);
      await expectLayerCacheExact(harness.engine, harness.source.id, restoredCache);
      harness.engine.history.redo();
      expect(harness.engine.document.getDocument()).toEqual(committedDocument);
      harness.engine.lifecycle.dispose();
    }
  );

  it('commitLayerCopy redo cache-draw failure leaves the restored state exact and retryable', async () => {
    const source = rasterLayer('copy-replay-source');
    const copy = { ...structuredClone(source), id: 'copy-replay-result', name: 'Copy replay result' };
    const document = { ...makeDoc(), layers: [source, sentinelLayer()], selectedLayerId: 'selection-sentinel' };
    const harness = createFaultHarness(document);
    const originalCache = await snapshotLayerCache(harness.engine, source.id);
    const originalDocument = harness.engine.document.getDocument()!;
    expect(harness.engine.layers.commitLayerCopy('Copy layer', source.id, copy, 0)).toBe(true);
    const committedDocument = structuredClone(harness.engine.document.getDocument()!);
    harness.engine.history.undo();
    const restoredDocument = harness.engine.document.getDocument()!;
    harness.faults.armDraw(0);

    expect(() => harness.engine.history.redo()).toThrow('structural cache draw failed');

    expect(harness.engine.document.getDocument()).toBe(restoredDocument);
    expect(harness.engine.document.getDocument()).toEqual(originalDocument);
    expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === copy.id)).toBe(false);
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(true);
    await expectLayerCacheExact(harness.engine, source.id, originalCache);
    harness.engine.history.redo();
    expect(harness.engine.document.getDocument()).toEqual(committedDocument);
    expect(harness.engine.document.getDocument()!.layers.filter((layer) => layer.id === copy.id)).toHaveLength(1);
    harness.engine.lifecycle.dispose();
  });

  it('commitLayerConversion keeps failed undo and redo cache preparation exact and retryable', async () => {
    const source = rasterLayer('conversion-replay-source');
    const document = { ...makeDoc(), layers: [source, sentinelLayer()], selectedLayerId: 'selection-sentinel' };
    const harness = createFaultHarness(document);
    await snapshotLayerCache(harness.engine, source.id);
    const originalDocument = harness.engine.document.getDocument()!;
    const live = originalDocument.layers.find((layer) => layer.id === source.id)!;
    if (live.type !== 'raster') {
      throw new Error('expected a raster conversion source');
    }
    const converted: CanvasLayerContract = {
      ...structuredClone(live),
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      type: 'control',
      withTransparencyEffect: false,
    };
    expect(harness.engine.layers.commitLayerConversion('Convert layer', live, converted)).toBe(true);
    const committedDocument = harness.engine.document.getDocument()!;
    const committedCache = await snapshotLayerCache(harness.engine, source.id);
    harness.faults.armDraw(0);

    expect(() => harness.engine.history.undo()).toThrow('structural cache draw failed');

    expect(harness.engine.document.getDocument()).toBe(committedDocument);
    expect(harness.engine.stores.canUndo.get()).toBe(true);
    expect(harness.engine.stores.canRedo.get()).toBe(false);
    await expectLayerCacheExact(harness.engine, source.id, committedCache);
    harness.engine.history.undo();
    expect(harness.engine.document.getDocument()).toEqual(originalDocument);
    const restoredDocument = harness.engine.document.getDocument()!;
    const restoredCache = await snapshotLayerCache(harness.engine, source.id);
    harness.faults.armAllocation(0);

    expect(() => harness.engine.history.redo()).toThrow('structural cache allocation failed');

    expect(harness.engine.document.getDocument()).toBe(restoredDocument);
    expect(harness.engine.document.getDocument()).toEqual(originalDocument);
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(true);
    await expectLayerCacheExact(harness.engine, source.id, restoredCache);
    harness.engine.history.redo();
    expect(harness.engine.document.getDocument()).toEqual(committedDocument);
    harness.engine.lifecycle.dispose();
  });

  it('boolean redo cache-draw failure leaves sources, selection, and history exact and retryable', async () => {
    const pair = twoPaintDoc();
    const document = {
      ...pair,
      layers: [...pair.layers, sentinelLayer()],
      selectedLayerId: 'selection-sentinel',
    };
    const harness = createFaultHarness(document);
    const upperCache = await snapshotLayerCache(harness.engine, 'upper');
    const belowCache = await snapshotLayerCache(harness.engine, 'below');
    const originalDocument = harness.engine.document.getDocument()!;
    await expect(harness.engine.layers.booleanMergeRasterLayers('upper', 'exclude')).resolves.toBe('merged');
    const committedDocument = structuredClone(harness.engine.document.getDocument()!);
    const resultId = harness.engine.document.getDocument()!.selectedLayerId!;
    harness.engine.history.undo();
    const restoredDocument = harness.engine.document.getDocument()!;
    harness.faults.armDraw(0);

    expect(() => harness.engine.history.redo()).toThrow('structural cache draw failed');

    expect(harness.engine.document.getDocument()).toBe(restoredDocument);
    expect(harness.engine.document.getDocument()).toEqual(originalDocument);
    expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === resultId)).toBe(false);
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(true);
    await expectLayerCacheExact(harness.engine, 'upper', upperCache);
    await expectLayerCacheExact(harness.engine, 'below', belowCache);
    harness.engine.history.redo();
    expect(harness.engine.document.getDocument()).toEqual(committedDocument);
    harness.engine.lifecycle.dispose();
  });

  it('masked extraction redo cache-draw failure leaves sources, selection, and history exact and retryable', async () => {
    const pair = twoPaintDoc();
    const mask: CanvasInpaintMaskLayerContract = {
      ...maskLayer('extract-replay-mask'),
      mask: {
        bitmap: { height: 20, imageName: 'extract-replay-mask-bitmap', width: 20 },
        fill: { color: '#e07575', style: 'diagonal' },
        offset: { x: 15, y: 25 },
      },
    };
    const document = {
      ...pair,
      layers: [mask, ...pair.layers, sentinelLayer()],
      selectedLayerId: 'selection-sentinel',
    };
    const harness = createFaultHarness(document);
    const maskCache = await snapshotLayerCache(harness.engine, mask.id);
    const upperCache = await snapshotLayerCache(harness.engine, 'upper');
    const belowCache = await snapshotLayerCache(harness.engine, 'below');
    const originalDocument = harness.engine.document.getDocument()!;
    const extracted = await harness.engine.exports.extractMaskedArea(mask.id);
    expect(extracted.status).toBe('extracted');
    if (extracted.status !== 'extracted') {
      throw new Error('expected masked extraction');
    }
    const committedDocument = structuredClone(harness.engine.document.getDocument()!);
    harness.engine.history.undo();
    const restoredDocument = harness.engine.document.getDocument()!;
    harness.faults.armDraw(0);

    expect(() => harness.engine.history.redo()).toThrow('structural cache draw failed');

    expect(harness.engine.document.getDocument()).toBe(restoredDocument);
    expect(harness.engine.document.getDocument()).toEqual(originalDocument);
    expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === extracted.layerId)).toBe(false);
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(true);
    await expectLayerCacheExact(harness.engine, mask.id, maskCache);
    await expectLayerCacheExact(harness.engine, 'upper', upperCache);
    await expectLayerCacheExact(harness.engine, 'below', belowCache);
    harness.engine.history.redo();
    expect(harness.engine.document.getDocument()).toEqual(committedDocument);
    harness.engine.lifecycle.dispose();
  });

  it('copyLayerToRaster undo restores the exact prior selection', async () => {
    const source = rasterLayer('selection-copy-source');
    const document = { ...makeDoc(), layers: [source, sentinelLayer()], selectedLayerId: 'selection-sentinel' };
    const harness = createFaultHarness(document);
    const originalDocument = harness.engine.document.getDocument()!;
    const copiedId = await harness.engine.layers.copyLayerToRaster(source.id);
    expect(copiedId).not.toBeNull();

    harness.engine.history.undo();

    expect(harness.engine.document.getDocument()).toEqual(originalDocument);
    expect(harness.engine.document.getDocument()!.selectedLayerId).toBe('selection-sentinel');
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('keeps a reducer-rejected copy undo retryable until its prior selection exists again', async () => {
    const source = rasterLayer('rejected-undo-source');
    const sentinel = sentinelLayer();
    const document = { ...makeDoc(), layers: [source, sentinel], selectedLayerId: sentinel.id };
    const harness = createFaultHarness(document);
    const copiedId = await harness.engine.layers.copyLayerToRaster(source.id);
    expect(copiedId).not.toBeNull();

    harness.store.dispatch({ ids: [sentinel.id], type: 'removeCanvasLayers' });
    expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === copiedId)).toBe(true);

    expect(() => harness.engine.history.undo()).toThrow('Canvas document mutation was rejected');
    expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === copiedId)).toBe(true);
    expect(harness.engine.stores.canUndo.get()).toBe(true);
    expect(harness.engine.stores.canRedo.get()).toBe(false);

    harness.store.dispatch({ layer: sentinel, type: 'addCanvasLayer' });
    expect(() => harness.engine.history.undo()).not.toThrow();
    expect(harness.engine.document.getDocument()!.layers.some((layer) => layer.id === copiedId)).toBe(false);
    expect(harness.engine.document.getDocument()!.selectedLayerId).toBe(sentinel.id);
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(harness.engine.stores.canRedo.get()).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('publishes a prepared structural mutation to its bound project while another project is active', () => {
    const source = rasterLayer('inactive-source');
    const document = { ...makeDoc(), layers: [source], selectedLayerId: source.id };
    const harness = createFaultHarness(document);
    const liveSource = harness.engine.document.getDocument()!.layers[0]!;
    const copy = { ...structuredClone(source), id: 'inactive-copy' };
    harness.store.dispatch({ type: 'createProject' });
    const otherProjectId = harness.store.getState().activeProjectId;
    const otherBefore = structuredClone(
      harness.store.getState().projects.find((project) => project.id === otherProjectId)!.canvas.document
    );
    (harness.store.dispatch as Mock).mockClear();

    expect(harness.engine.layers.commitLayerCopy('Copy layer', liveSource.id, copy, 0)).toBe(true);

    expect(harness.store.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'applyCanvasLayerStackMutation' })
    );
    expect(harness.store.getState().projects.find((project) => project.id === otherProjectId)!.canvas.document).toEqual(
      otherBefore
    );
    expect(harness.engine.document.getDocument()?.layers[0]?.id).toBe(copy.id);
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
    const document: CanvasDocumentContractV2 = {
      ...makeDoc(),
      bbox: { height: 7, width: 6, x: 2, y: 3 },
      layers: [source, sentinelLayer()],
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
        const layer = reducer.store
          .getState()
          .projects.find((project) => project.id === reducer.projectId)
          ?.canvas.document.layers.find((candidate) => candidate.id === layerId);
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
    expect(engine.document.getDocument()!.layers.find((layer) => layer.id === source.id)).toMatchObject({
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
  /** [upper raster, mask, lower raster] — the interleaved case round 1 no-oped. */
  const interleavedDoc = (): CanvasDocumentContractV2 => {
    const base = twoPaintDoc();
    const [upper, below] = base.layers as [CanvasLayerContract, CanvasLayerContract];
    return { ...base, layers: [upper, maskLayer('mid-mask'), below] };
  };

  const setup = (doc: CanvasDocumentContractV2) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const { projectId, store } = createReducerBackedStore(doc);
    const base = createTestStubRasterBackend();
    const surfaces: StubRasterSurface[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (width, height) => {
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
    return { engine, raf, store, surfaces };
  };

  it('waits for contributor rasterization before applying one atomic insertion', async () => {
    const { engine, raf } = setup(interleavedDoc());
    const pending = engine.layers.mergeVisibleRasterLayers();
    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'mid-mask', 'below']);

    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(await pending).toBe('merged');
    expect(
      engine.document
        .getDocument()!
        .layers.slice(1)
        .map((layer) => layer.id)
    ).toEqual(['upper', 'mid-mask', 'below']);

    engine.lifecycle.dispose();
  });

  it('creates a selected raster at index zero and preserves every source layer', async () => {
    const { engine, raf } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');

    const layers = engine.document.getDocument()!.layers;
    expect(layers).toHaveLength(4);
    expect(layers[0]).toMatchObject({
      blendMode: 'normal',
      isEnabled: true,
      isLocked: false,
      name: 'upper merged',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(layers.slice(1).map((layer) => layer.id)).toEqual(['upper', 'mid-mask', 'below']);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(layers[0]!.id);

    engine.lifecycle.dispose();
  });

  it('includes locked parametric rasters, excludes hidden rasters, and composites bottom-to-top', async () => {
    const doc = twoPaintDoc();
    const upper: CanvasRasterLayerContractV2 = {
      ...(doc.layers[0] as CanvasRasterLayerContractV2),
      isLocked: true,
    };
    const below: CanvasRasterLayerContractV2 = {
      ...(doc.layers[1] as CanvasRasterLayerContractV2),
      source: {
        angle: 0,
        height: 60,
        kind: 'linear',
        stops: [
          { color: '#000000', offset: 0 },
          { color: '#ffffff', offset: 1 },
        ],
        type: 'gradient',
        width: 60,
      },
    };
    const hidden: CanvasRasterLayerContractV2 = {
      ...(doc.layers[1] as CanvasRasterLayerContractV2),
      id: 'hidden',
      isEnabled: false,
      name: 'hidden',
      source: { bitmap: { height: 500, imageName: 'hidden-bmp', width: 500 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -1000, y: -1000 },
    };
    const sourceLayers = [upper, maskLayer('mid-mask'), hidden, below];
    const { engine, raf, surfaces } = setup({ ...doc, layers: sourceLayers });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');
    const merged = engine.document.getDocument()!.layers[0]!;
    expect(merged).toMatchObject({ source: { offset: { x: 10, y: 20 }, type: 'paint' }, type: 'raster' });
    expect(engine.document.getDocument()!.layers.slice(1)).toEqual(sourceLayers);

    const composite = surfaces.find(
      (surface) =>
        surface.width === 60 &&
        surface.height === 60 &&
        surface.callLog.some(
          (entry) => entry.op === 'set' && entry.args[0] === 'globalCompositeOperation' && entry.args[1] === 'multiply'
        )
    );
    expect(composite).toBeDefined();
    expect(
      composite?.callLog
        .filter((entry) => entry.op === 'drawImage')
        .map((entry) => (entry.args[0] as { width: number }).width)
    ).toEqual([60, 40]);
    expect(
      composite?.callLog
        .filter((entry) => entry.op === 'set' && entry.args[0] === 'globalAlpha')
        .map((entry) => entry.args[1])
    ).toEqual([1, 0.5, 1]);

    engine.lifecycle.dispose();
  });

  it('creates another composite layer when invoked repeatedly', async () => {
    const { engine, raf } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');
    const firstId = engine.document.getDocument()!.layers[0]!.id;
    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');
    const layers = engine.document.getDocument()!.layers;
    expect(layers).toHaveLength(5);
    expect(layers[0]!.id).not.toBe(firstId);
    expect(layers[1]!.id).toBe(firstId);
    expect(layers.slice(2).map((layer) => layer.id)).toEqual(['upper', 'mid-mask', 'below']);

    engine.lifecycle.dispose();
  });

  it('undoes and redoes only the generated layer', async () => {
    const { engine, raf } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const before = engine.document.getDocument()!;

    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');
    const merged = engine.document.getDocument()!.layers[0]!;

    engine.history.undo();
    expect(engine.document.getDocument()!.layers).toEqual(before.layers);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(before.selectedLayerId);

    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(merged);
    expect(engine.document.getDocument()!.layers.slice(1)).toEqual(before.layers);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(merged.id);

    engine.lifecycle.dispose();
  });

  it('blocks the group merge without closing an operation and restores it after cancel', async () => {
    const { engine, raf } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const exported = await engine.exports.exportLayerPixels('upper');
    if (exported.status !== 'ok') {
      throw new Error('expected an operation guard');
    }
    const session = getCanvasOperations(engine).controller.start({
      cleanupPreview: vi.fn(),
      guard: exported.guard,
      identity: { kind: 'filter', layerId: 'upper', projectId: engine.projectId },
    });

    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('busy');
    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'mid-mask', 'below']);
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter' },
      status: 'active',
    });

    session?.cancel();
    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('merged');
    engine.lifecycle.dispose();
  });

  it('returns nothing for empty raster layers and preserves them', async () => {
    const base = twoPaintDoc();
    const doc: CanvasDocumentContractV2 = {
      ...base,
      layers: base.layers.map((layer) => ({
        ...layer,
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' as const },
      })),
    };
    const { engine, raf } = setup(doc);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'below']);
    expect(await engine.layers.mergeVisibleRasterLayers()).toBe('nothing');
    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'below']);

    engine.lifecycle.dispose();
  });
});

// ---- rasterizeLayer: parametric → paint, undoable via param re-convert ----

const shapeLayerDoc = (over: { isLocked?: boolean } = {}): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [
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
  ],
  selectedLayerId: 'shape1',
  version: 2,
  width: 100,
});

describe('rasterizeLayer (parametric → paint)', () => {
  const setup = (doc: CanvasDocumentContractV2) => {
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

  it('bakes to a CONTENT-sized paint layer at identity and dispatches convertCanvasLayer with the offset', () => {
    const { dispatch, engine, surfaces } = setup(shapeLayerDoc());
    const before = surfaces.length;

    expect(engine.layers.rasterizeLayer('shape1')).toBe(true);

    const converts = convertCalls(dispatch);
    expect(converts).toHaveLength(1);
    const convert = converts[0];
    expect(convert).toMatchObject({ id: 'shape1', targetType: 'raster', type: 'convertCanvasLayer' });
    if (convert?.type === 'convertCanvasLayer') {
      expect(convert.layer.type).toBe('raster');
      if (convert.layer.type === 'raster') {
        // Content-sized: the shape (60×40) at transform (10,20) bakes to a paint
        // layer whose bitmap sits at offset (10,20); the transform resets to identity.
        expect(convert.layer.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });
        expect(convert.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
      }
    }

    // A fresh CONTENT-sized surface (the transformed shape bounds, 60×40) was
    // allocated and the parametric cache baked into it.
    const baked = surfaces[before];
    expect(baked).toBeDefined();
    expect(baked?.width).toBe(60);
    expect(baked?.height).toBe(40);
    expect(baked?.callLog.some((e) => e.op === 'drawImage')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('undo re-converts to the ORIGINAL parametric source (no pixel snapshot)', () => {
    const { dispatch, engine } = setup(shapeLayerDoc());
    engine.layers.rasterizeLayer('shape1');

    engine.history.undo();
    const converts = convertCalls(dispatch);
    // forward convert + undo convert.
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

    // Redo re-applies the paint conversion (bitmap:null at the baked offset).
    engine.history.redo();
    const afterRedo = convertCalls(dispatch);
    expect(afterRedo).toHaveLength(3);
    if (afterRedo[2]?.type === 'convertCanvasLayer' && afterRedo[2].layer.type === 'raster') {
      expect(afterRedo[2].layer.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });
    }
    engine.lifecycle.dispose();
  });

  it('redo re-bakes from params rather than pinning the doc-sized surface (byte-budget honesty)', () => {
    // Regression: the history entry declared bytes:256 but captured the doc-sized
    // `baked` surface in its redo closure, so repeated rasterizes could retain
    // gigabytes invisible to HISTORY_BYTE_BUDGET. Redo must re-bake instead.
    const { engine, surfaces } = setup(shapeLayerDoc());

    expect(engine.layers.rasterizeLayer('shape1')).toBe(true);
    const afterApply = surfaces.length;

    engine.history.undo();
    // Undo only re-converts (a dispatch) — it bakes nothing.
    expect(surfaces.length).toBe(afterApply);

    engine.history.redo();
    // Redo allocates FRESH surfaces: it re-baked from params (content-sized to the
    // transformed shape bounds, 60×40) rather than reusing a surface pinned by the
    // entry.
    expect(surfaces.length).toBeGreaterThan(afterApply);
    const rebaked = surfaces.at(-1);
    expect(rebaked?.width).toBe(60);
    expect(rebaked?.height).toBe(40);
    expect(rebaked?.callLog.some((e) => e.op === 'drawImage')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('rasterizes a gradient layer', () => {
    const doc = shapeLayerDoc();
    doc.layers[0] = {
      ...doc.layers[0],
      source: { angle: 45, kind: 'linear', stops: [{ color: '#000', offset: 0 }], type: 'gradient' },
    } as CanvasLayerContract;
    const { dispatch, engine } = setup(doc);
    expect(engine.layers.rasterizeLayer('shape1')).toBe(true);
    expect(convertCalls(dispatch)).toHaveLength(1);
    engine.lifecycle.dispose();
  });

  it('is a no-op for a locked layer, a missing layer, and a non-parametric (paint) layer', () => {
    const { dispatch, engine } = setup(shapeLayerDoc({ isLocked: true }));
    expect(engine.layers.rasterizeLayer('shape1')).toBe(false);
    expect(engine.layers.rasterizeLayer('nope')).toBe(false);
    expect(convertCalls(dispatch)).toHaveLength(0);
    engine.lifecycle.dispose();

    const paintDoc = shapeLayerDoc();
    paintDoc.layers[0] = { ...paintDoc.layers[0], source: { bitmap: null, type: 'paint' } } as CanvasLayerContract;
    const paint = setup(paintDoc);
    expect(paint.engine.layers.rasterizeLayer('shape1')).toBe(false);
    paint.engine.lifecycle.dispose();
  });

  // ---- rasterize → undo → bitmap-store flush: source-type guard --------
  //
  // Reviewer-flagged bug: rasterize bakes the shape to a paint layer and marks
  // it dirty in the bitmap store; undo re-converts it back to the parametric
  // shape. Nothing previously cleared the pending dirty mark, so the eventual
  // debounced (or barrier) flush would encode the paint-cache surface — still
  // populated, since a source swap doesn't clear it — and dispatch
  // `updateCanvasLayerSource({ type: 'paint', ... })`, silently flipping the
  // parametric layer back to paint with stale, wrong-extent pixels.
  //
  // These tests wire a REAL `createBitmapStore` (exercising the actual guard
  // in `flushLayer`) through `opts.bitmapStore`, with test-controlled
  // encode/upload stubs (no real network) and a `getLayerSurface` that always
  // resolves — mirroring the bug precondition that a source swap does NOT
  // clear the cache. `getLayerSource` reads the engine's own mirrored
  // document, so it reflects the exact reducer round trip the test drives via
  // `setDocument`.
  describe('rasterize → undo → bitmap-store flush (source-type guard)', () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    const setupWithRealBitmapStore = (doc: CanvasDocumentContractV2) => {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
      const { setDocument, store } = createReactiveStore(doc);
      const dispatch = store.dispatch as Mock;

      const encodeSurface = vi.fn(() => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })));
      const uploadImage = vi.fn(() => Promise.resolve({ height: 100, imageName: 'img-x', width: 100 }));
      // Always resolves, regardless of the layer's current source: mirrors the
      // real cache, whose surface a source swap does NOT clear (only marks stale).
      const fakeSurface = createTestStubRasterBackend().createSurface(10, 10);

      // Forward-declared: `getLayerSource` closes over it, but is only ever
      // CALLED once `engine` is assigned below (bitmap-store flushes never
      // happen synchronously during construction).
      let engine: ReturnType<typeof createCanvasEngine>;
      const bitmapStore = createBitmapStore({
        dispatch: createTestMutationPort(store, 'p1').dispatch,
        encodeSurface,
        getLayerSource: (layerId) => {
          const layer = engine.document.getDocument()?.layers.find((candidate) => candidate.id === layerId);
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

    /**
     * Applies the most recently dispatched `convertCanvasLayer` action onto
     * `doc`, simulating what the real reducer would do — `createReactiveStore`'s
     * `dispatch` is a bare spy and does not mutate state on its own.
     */
    const applyLastConvert = (doc: CanvasDocumentContractV2, dispatch: Mock): CanvasDocumentContractV2 => {
      const converts = convertCalls(dispatch);
      const last = converts.at(-1);
      if (last?.type !== 'convertCanvasLayer') {
        throw new Error('expected a convertCanvasLayer dispatch');
      }
      return { ...doc, layers: doc.layers.map((layer) => (layer.id === last.id ? last.layer : layer)) };
    };

    /**
     * Rasterizes `shape1` (paint bake, dirty-marks the bitmap store), applies
     * that conversion to the mirrored document, then undoes it and applies
     * THAT conversion too — the full reducer round trip a live document would
     * go through. Leaves the document back at its original parametric shape
     * source, with the paint-bake dirty mark for `shape1` still pending.
     */
    const rasterizeThenUndo = () => {
      vi.useFakeTimers();
      let doc = shapeLayerDoc();
      const harness = setupWithRealBitmapStore(doc);
      const { dispatch, engine, raf, setDocument } = harness;

      expect(engine.layers.rasterizeLayer('shape1')).toBe(true);
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      engine.history.undo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      return harness;
    };

    /** Every dispatched `updateCanvasLayerSource` whose source is `paint`. */
    const paintSourceDispatches = (dispatch: Mock): Extract<EngineTestAction, { type: 'updateCanvasLayerSource' }>[] =>
      dispatch.mock.calls
        .map((call) => call[0] as EngineTestAction)
        .filter(
          (action): action is Extract<EngineTestAction, { type: 'updateCanvasLayerSource' }> =>
            action.type === 'updateCanvasLayerSource' && action.source.type === 'paint'
        );

    it('await flushPendingUploads(): the layer stays the parametric shape and no paint dispatch fires', async () => {
      const { dispatch, engine, uploadImage } = rasterizeThenUndo();

      await engine.lifecycle.flushPendingUploads();

      // The guard drops the flush before it ever encodes/uploads.
      expect(uploadImage).not.toHaveBeenCalled();
      expect(paintSourceDispatches(dispatch)).toHaveLength(0);
      const layer = engine.document.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layer.source.type).toBe('shape');

      engine.lifecycle.dispose();
    });

    it('advancing the debounce timer: the layer stays the parametric shape and no paint dispatch fires', async () => {
      const { dispatch, engine, uploadImage } = rasterizeThenUndo();

      await vi.advanceTimersByTimeAsync(1500);

      expect(uploadImage).not.toHaveBeenCalled();
      expect(paintSourceDispatches(dispatch)).toHaveLength(0);
      const layer = engine.document.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layer.source.type).toBe('shape');

      engine.lifecycle.dispose();
    });

    it('redo after rasterize → flush → undo re-dispatches the paint image ref (fix round 2: no permanent bitmap:null)', async () => {
      // Reviewer round 2, data-loss finding: rasterize → flush (contract lands
      // on img-x, and the store's `lastApplied` remembers img-x) → undo (back
      // to shape) → redo (convertCanvasLayer resets to `paint {bitmap: null}`)
      // → a further flush re-bakes IDENTICAL pixels, so the content-hash dedupe
      // resolves back to img-x — but the old `lastApplied`-based redundant-
      // dispatch skip treated that as "already applied" and swallowed the
      // dispatch, permanently stranding the document on `bitmap: null`. This
      // drives the FULL sequence (including the first flush landing for real)
      // and asserts the second flush actually re-dispatches the ref.
      vi.useFakeTimers();
      let doc = shapeLayerDoc();
      const { dispatch, engine, raf, setDocument, uploadImage } = setupWithRealBitmapStore(doc);

      // Rasterize → bake to paint → flush lands img-x.
      expect(engine.layers.rasterizeLayer('shape1')).toBe(true);
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await vi.advanceTimersByTimeAsync(1500);
      await engine.lifecycle.flushPendingUploads();
      expect(uploadImage).toHaveBeenCalledTimes(1);

      const firstPersist = paintSourceDispatches(dispatch).at(-1);
      expect(firstPersist).toBeDefined();
      expect(firstPersist?.source).toMatchObject({ bitmap: { imageName: 'img-x' }, type: 'paint' });
      // Apply the persisted ref onto the mirrored document, as the real reducer
      // would — the document now genuinely points at img-x, not `bitmap: null`.
      doc = {
        ...doc,
        layers: doc.layers.map((layer) =>
          layer.id === firstPersist?.id ? { ...layer, source: firstPersist.source } : layer
        ),
      };
      setDocument(doc);
      raf.flush();

      // Undo → back to the parametric shape source.
      engine.history.undo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      // Redo → paint bake again; the fresh conversion lands on `bitmap: null`
      // (only the debounced flush fills in the persisted ref).
      engine.history.redo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();
      const layerAfterRedo = engine.document.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layerAfterRedo.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });

      // Flush again: the re-baked pixels are identical, so the content hash
      // dedupes back to img-x with NO new upload — but the ref must still be
      // re-dispatched into the contract, since the document currently reads
      // `bitmap: null`, not img-x.
      await vi.advanceTimersByTimeAsync(1500);
      await engine.lifecycle.flushPendingUploads();

      expect(uploadImage).toHaveBeenCalledTimes(1); // dedupe hit: no re-upload
      const dispatchesAfterRedo = paintSourceDispatches(dispatch);
      const secondPersist = dispatchesAfterRedo.at(-1);
      expect(secondPersist).toBeDefined();
      expect(secondPersist).not.toBe(firstPersist);
      expect(secondPersist?.source).toMatchObject({ bitmap: { imageName: 'img-x' }, type: 'paint' });

      // Applying it restores the document's bitmap ref — no longer null.
      doc = {
        ...doc,
        layers: doc.layers.map((layer) =>
          layer.id === secondPersist?.id ? { ...layer, source: secondPersist.source } : layer
        ),
      };
      setDocument(doc);
      const layerFinal = engine.document.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layerFinal.source.type).toBe('paint');
      if (layerFinal.source.type === 'paint') {
        expect(layerFinal.source.bitmap).not.toBeNull();
        expect(layerFinal.source.bitmap?.imageName).toBe('img-x');
      }

      engine.lifecycle.dispose();
    });

    it('sanity check: a normal (non-reverted) rasterize DOES flush to paint (the guard only blocks the reverted case)', async () => {
      vi.useFakeTimers();
      let doc = shapeLayerDoc();
      const { dispatch, engine, raf, setDocument, uploadImage } = setupWithRealBitmapStore(doc);

      expect(engine.layers.rasterizeLayer('shape1')).toBe(true);
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await vi.advanceTimersByTimeAsync(1500);
      await engine.lifecycle.flushPendingUploads();

      expect(uploadImage).toHaveBeenCalledTimes(1);
      expect(paintSourceDispatches(dispatch)).toHaveLength(1);

      engine.lifecycle.dispose();
    });
  });
});

// ---- nudgeSelectedLayer: bounds/lock logic + coalescing ----------------

const selectedImageDoc = (
  overrides: { isLocked?: boolean; isEnabled?: boolean } = {},
  selectedLayerId: string | null = 'a'
): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [{ ...rasterLayer('a'), ...overrides }],
  selectedLayerId,
  version: 2,
  width: 100,
});

describe('nudgeSelectedLayer', () => {
  const setup = (doc: CanvasDocumentContractV2) => {
    const { store } = createFakeStore(doc);
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    return { dispatch, engine };
  };

  it('no-ops with no selection', () => {
    const { dispatch, engine } = setup(selectedImageDoc({}, null));
    engine.layers.nudgeSelectedLayer(1, 0);
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('no-ops on a locked selected layer', () => {
    const { dispatch, engine } = setup(selectedImageDoc({ isLocked: true }));
    engine.layers.nudgeSelectedLayer(0, 1);
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('no-ops on a hidden selected layer', () => {
    const { dispatch, engine } = setup(selectedImageDoc({ isEnabled: false }));
    engine.layers.nudgeSelectedLayer(0, 1);
    expect(dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('dispatches a transform update and records an undoable entry', () => {
    const { dispatch, engine } = setup(selectedImageDoc());
    engine.layers.nudgeSelectedLayer(3, -2);
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenNthCalledWith(1, {
      id: 'a',
      patch: { transform: { x: 3, y: -2 } },
      type: 'updateCanvasLayer',
    });
    expect(engine.stores.canUndo.get()).toBe(true);

    // Undo dispatches the inverse (back to the original position).
    engine.history.undo();
    expect(dispatch).toHaveBeenNthCalledWith(2, {
      id: 'a',
      patch: { transform: { x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });
    engine.lifecycle.dispose();
  });

  it('coalesces a rapid same-layer burst into one history entry', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_000);
    const { engine } = setup(selectedImageDoc());
    engine.layers.nudgeSelectedLayer(1, 0);
    engine.layers.nudgeSelectedLayer(1, 0);
    engine.layers.nudgeSelectedLayer(1, 0);
    // A single undo empties the stack: the burst collapsed to one entry.
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.history.undo();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
    vi.restoreAllMocks();
  });

  it('starts a fresh entry once the coalescing window elapses', () => {
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(1_000);
    const { engine } = setup(selectedImageDoc());
    engine.layers.nudgeSelectedLayer(1, 0);
    now.mockReturnValue(2_000); // > 500ms later
    engine.layers.nudgeSelectedLayer(1, 0);
    // Two distinct entries: one undo still leaves something to undo.
    engine.history.undo();
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.history.undo();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
    vi.restoreAllMocks();
  });
});

// ---- move tool: drag gesture through the pointer pipeline ---------------

describe('move tool: drag through the pipeline', () => {
  it('commits one structural transform update and records an undoable entry', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    // A document-sized paint layer is hit-testable everywhere and pre-selected.
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

  it('a click (no drag) dispatches a selection change and no transform update', () => {
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
    overlay.fire('pointerup', pointerAt(20, 20, { buttons: 0 }));

    const actions = dispatch.mock.calls.map((call) => call[0] as EngineTestAction);
    expect(actions.some((action) => action.type === 'updateCanvasLayer')).toBe(false);
    expect(actions.some((action) => action.type === 'setCanvasSelectedLayer')).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.lifecycle.dispose();
  });
});

// ---- ride-along: composed auto-create + stroke history entry ------------

const imageSelectedDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  // Selected layer is an image (not paintable) → a brush stroke auto-creates a
  // fresh paint layer for the gesture.
  layers: [rasterLayer('img')],
  selectedLayerId: 'img',
  version: 2,
  width: 100,
});

describe('engine-owned history: composed auto-create + stroke entry', () => {
  it('undo removes the auto-created layer (no pixel restore); redo re-adds it and re-applies the stroke', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

    const { store } = createReactiveStore(imageSelectedDoc());
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

    // The gesture auto-created a paint layer and reported it on the commit event.
    expect(strokes).toHaveLength(1);
    const created = strokes[0]!.createdLayer;
    expect(created).toBeDefined();
    const newLayerId = created!.layer.id;
    expect(engine.stores.canUndo.get()).toBe(true);

    // The auto-create dispatched addCanvasLayer for the new id.
    const addCalls = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'addCanvasLayer');
    expect(addCalls).toHaveLength(1);

    const putBefore = () => putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.beforeImageData);
    const beforeUndoPutBefore = putBefore();

    // Undo: removes the auto-created layer, and does NOT restore pre-stroke pixels
    // (the layer's cache is gone).
    engine.history.undo();
    const removeAfterUndo = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'removeCanvasLayers');
    expect(removeAfterUndo).toHaveLength(1);
    expect(removeAfterUndo[0]).toEqual({ ids: [newLayerId], type: 'removeCanvasLayers' });
    // No new before-pixel putImageData was introduced by the undo.
    expect(putBefore()).toBe(beforeUndoPutBefore);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    // Redo: re-adds the layer and re-applies the stroke's after pixels.
    engine.history.redo();
    const addAfterRedo = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .filter((action) => action.type === 'addCanvasLayer');
    expect(addAfterRedo).toHaveLength(2); // original auto-create + redo re-add
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.afterImageData)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.lifecycle.dispose();
  });
});

describe('setStagedPreview', () => {
  const emptyDoc = (bbox = { height: 100, width: 100, x: 0, y: 0 }): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox,
    height: 100,
    layers: [],
    selectedLayerId: null,
    version: 2,
    width: 100,
  });

  /** A stub backend whose `createImageBitmap` is deferred, so decodes resolve on demand. */
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

  /** Screen-surface staged-preview draws are the 5-arg `drawImage(canvas, x, y, w, h)` calls. */
  const stagedDraws = (surface: StubRasterSurface): unknown[][] =>
    surface.callLog.filter((entry) => entry.op === 'drawImage' && entry.args.length === 5).map((entry) => entry.args);

  const dataUrl = (tag: string) => `data:image/png;base64,${tag}`;

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
    raf.flush(); // initial (no layers, no preview)
    screen.surface.callLog.length = 0;

    // Two rapid selections; the second supersedes the first.
    engine.previews.setStagedPreview({ dataUrl: dataUrl('AAAA'), height: 10, width: 10 }); // decode #0
    engine.previews.setStagedPreview({ dataUrl: dataUrl('BBBB'), height: 20, width: 20 }); // decode #1

    // The newer decode resolves first and is drawn; a frame must have been scheduled.
    bitmaps.resolveBitmap(1);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    // The stale (older) decode resolves afterwards and must be dropped.
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();

    const draws = stagedDraws(screen.surface);
    expect(draws.length).toBeGreaterThan(0);
    // Every staged draw is the 20x20 candidate at the bbox origin; the 10x10
    // stale decode never reaches the screen.
    for (const args of draws) {
      expect(args.slice(1)).toEqual([0, 0, 20, 20]);
    }

    engine.lifecycle.dispose();
  });

  it('clears the preview on null', async () => {
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
    engine.surface.attach(screen.element, createFakeCanvas().element);

    engine.previews.setStagedPreview({ dataUrl: dataUrl('AAAA'), height: 10, width: 10 });
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();
    expect(stagedDraws(screen.surface).length).toBeGreaterThan(0);

    screen.surface.callLog.length = 0;
    engine.previews.setStagedPreview(null);
    raf.flush();
    expect(stagedDraws(screen.surface)).toHaveLength(0);

    engine.lifecycle.dispose();
  });

  it('follows the current bbox origin, not the bbox at set time', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const bitmaps = createDeferredBitmapBackend();
    const { setDocument, store } = createReactiveStore(emptyDoc({ height: 100, width: 100, x: 0, y: 0 }));
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    engine.surface.attach(screen.element, createFakeCanvas().element);

    engine.previews.setStagedPreview({ dataUrl: dataUrl('AAAA'), height: 10, width: 10 });
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();
    expect(stagedDraws(screen.surface).at(-1)!.slice(1)).toEqual([0, 0, 10, 10]);

    // Move the bbox (an ordinary edit, same document revision — not a replacement).
    screen.surface.callLog.length = 0;
    setDocument(emptyDoc({ height: 10, width: 10, x: 30, y: 40 }));
    raf.flush();
    expect(stagedDraws(screen.surface).at(-1)!.slice(1)).toEqual([30, 40, 10, 10]);

    engine.lifecycle.dispose();
  });

  it('decodes an imageName candidate through the resolver and draws it at the bbox', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(emptyDoc({ height: 100, width: 100, x: 5, y: 7 }));
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    engine.surface.attach(screen.element, createFakeCanvas().element);

    engine.previews.setStagedPreview({ imageName: 'staged-candidate' });
    await flushMicrotasks();
    raf.flush();

    expect(resolver).toHaveBeenCalledWith('staged-candidate');
    // The decoded (stub, 0-sized) surface is drawn at the current bbox origin.
    expect(stagedDraws(screen.surface).at(-1)!.slice(1, 3)).toEqual([5, 7]);

    engine.lifecycle.dispose();
  });

  it('stores an image candidate placement and renders it independently of the current bbox', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const base = createTestStubRasterBackend();
    const backend: StubRasterBackend = {
      ...base,
      createImageBitmap: () => Promise.resolve({ close: vi.fn(), height: 17, width: 23 } as unknown as ImageBitmap),
    };
    const { setDocument, store } = createReactiveStore(emptyDoc({ height: 100, width: 100, x: 50, y: 60 }));
    const engine = createCanvasEngine({
      backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const screen = createFakeCanvas();
    engine.surface.attach(screen.element, createFakeCanvas().element);

    engine.previews.setStagedPreview({
      imageName: 'placed-candidate',
      placement: { height: 34, opacity: 0.4, width: 46, x: -8, y: 13 },
    });
    await flushMicrotasks();
    raf.flush();

    expect(stagedDraws(screen.surface).at(-1)!.slice(1)).toEqual([-8, 13, 46, 34]);
    const alphaSets = screen.surface.callLog
      .filter((entry) => entry.op === 'set' && entry.args[0] === 'globalAlpha')
      .map((entry) => entry.args[1]);
    expect(alphaSets).toContain(0.4);

    // A placed candidate no longer follows the bbox, so moving the bbox only
    // invalidates overlay chrome and must not recomposite every canvas layer.
    screen.surface.callLog.length = 0;
    setDocument(emptyDoc({ height: 50, width: 50, x: 2, y: 3 }));
    raf.flush();
    expect(stagedDraws(screen.surface)).toHaveLength(0);
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
    adjustments: { brightness: 0.2, contrast: -0.1, saturation: 0.3 },
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

  const filterDoc = (layers: CanvasLayerContract[] = [filterLayer()]): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers,
    selectedLayerId: layers[0]?.id ?? null,
    version: 2,
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
    const before = structuredClone(engine.document.getDocument()!.layers[0]!);
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
    const after = engine.document.getDocument()!.layers[0]!;
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

    engine.history.undo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(before);
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

    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(after);
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
    expect(engine.document.getDocument()!.layers.map((candidate) => candidate.id)).toEqual([
      result.layerId,
      'source',
      'below',
    ]);
    expect(engine.document.getDocument()!.layers.find((candidate) => candidate.id === 'source')).toEqual(sourceBefore);
    const copy = engine.document.getDocument()!.layers[0]!;
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

    engine.history.undo();
    expect(engine.document.getDocument()!.layers.map((candidate) => candidate.id)).toEqual(['source', 'below']);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(below.id);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.history.redo();
    expect(engine.document.getDocument()!.layers.map((candidate) => candidate.id)).toEqual([
      result.layerId,
      'source',
      'below',
    ]);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
    expect(engine.document.getDocument()!.layers[0]).toEqual(copy);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('atomically applies a filter and nonzero local origin to a control layer with exact undo/redo', async () => {
    const source: CanvasLayerContract = {
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'normal',
      filter: { settings: { radius: 1 }, type: 'old_filter' },
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control',
      opacity: 1,
      source: { image: { height: 10, imageName: 'control-source', width: 10 }, type: 'image' },
      transform: { rotation: 12, scaleX: 2, scaleY: 3, x: 40, y: 50 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const { projectId, store } = createReducerBackedStore(filterDoc([source]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const before = structuredClone(source);
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable control export');
    }

    const result = await engine.layers.commitRasterFilterResult({
      filter: { settings: { radius: 8 }, type: 'content_shuffle' },
      guard: exported.guard,
      image: durableImage,
      mode: 'replace',
      rect: { height: 10, width: 10, x: 7, y: -3 },
      target: 'apply',
    });

    expect(result).toEqual({ layerId: source.id, status: 'committed' });
    const after = engine.document.getDocument()!.layers[0]!;
    expect(after).toMatchObject({
      filter: { settings: { radius: 8 }, type: 'content_shuffle' },
      source: { bitmap: durableImage, offset: { x: 7, y: -3 }, type: 'paint' },
      transform: source.transform,
      type: 'control',
    });
    engine.history.undo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(before);
    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(after);
    engine.lifecycle.dispose();
  });

  it.each(['raster', 'control'] as const)(
    'saves a filtered result as a %s layer above a control source',
    async (target) => {
      const source: CanvasLayerContract = {
        adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
        blendMode: 'normal',
        id: 'control',
        isEnabled: true,
        isLocked: false,
        name: 'Control',
        opacity: 1,
        source: { image: { height: 10, imageName: 'control-source', width: 10 }, type: 'image' },
        transform: { rotation: 12, scaleX: 2, scaleY: 3, x: 40, y: 50 },
        type: 'control',
        withTransparencyEffect: true,
      };
      const { projectId, store } = createReducerBackedStore(filterDoc([source]));
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore: createSpyBitmapStore(),
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels(source.id);
      if (exported.status !== 'ok') {
        throw new Error('expected a filterable control export');
      }

      const result = await engine.layers.commitRasterFilterResult({
        filter: { settings: {}, type: 'canny_edge_detection' },
        guard: exported.guard,
        image: durableImage,
        mode: 'copy',
        rect: { height: 10, width: 10, x: 7, y: -3 },
        target,
      });

      expect(result.status).toBe('committed');
      expect(engine.document.getDocument()!.layers.map((layer) => layer.type)).toEqual([target, 'control']);
      expect(engine.document.getDocument()!.layers[0]).toMatchObject({
        filter: { settings: {}, type: 'canny_edge_detection' },
        source: { offset: { x: 7, y: -3 }, type: 'paint' },
        transform: source.transform,
      });
      engine.lifecycle.dispose();
    }
  );

  it('preserves raster source placement when saving the filtered result as control', async () => {
    const source = filterLayer('source');
    const { projectId, store } = createReducerBackedStore(filterDoc([source]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable raster export');
    }

    const result = await engine.layers.commitRasterFilterResult({
      filter: { settings: {}, type: 'canny_edge_detection' },
      guard: exported.guard,
      image: durableImage,
      mode: 'copy',
      rect: { height: 10, width: 10, x: -4, y: 7 },
      target: 'control',
    });

    expect(result.status).toBe('committed');
    expect(engine.document.getDocument()!.layers[0]).toMatchObject({
      source: { offset: { x: -4, y: 7 }, type: 'paint' },
      transform: source.transform,
      type: 'control',
    });
    engine.lifecycle.dispose();
  });

  it.each([
    { base: 'z-image', expectedKind: 'z_image_control' },
    { base: 'sd-1', expectedKind: 'controlnet' },
    { base: 'flux', expectedKind: 'controlnet' },
  ] as const)(
    'uses $expectedKind defaults when Save Filter As Control runs under $base',
    async ({ base, expectedKind }) => {
      const source = filterLayer('source');
      const { projectId, store } = createReducerBackedStore(filterDoc([source]), base);
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        bitmapStore: createSpyBitmapStore(),
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels(source.id);
      if (exported.status !== 'ok') {
        throw new Error('expected a filterable raster export');
      }

      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode: 'copy',
        rect: exported.rect,
        target: 'control',
      });

      expect(result.status).toBe('committed');
      const created = engine.document.getDocument()!.layers[0];
      expect(created?.type === 'control' ? created.adapter.kind : null).toBe(expectedKind);
      engine.lifecycle.dispose();
    }
  );

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

      expect(() => engine.history.undo()).toThrow(`replay cache ${failure} failed`);

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

      engine.history.undo();
      expect(engine.document.getDocument()!.layers[0]).toEqual(source);
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(true);
      unsubscribeThumbnail();
      engine.lifecycle.dispose();
    }
  );

  it.each(['allocation', 'draw'] as const)(
    'keeps replace redo retryable and exact when detached cache %s fails',
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
      const filteredDocument = structuredClone(engine.document.getDocument()!);
      engine.history.undo();
      const expectedDocument = structuredClone(engine.document.getDocument()!);
      const expectedCache = await engine.exports.exportLayerPixels(source.id);
      if (expectedCache.status !== 'ok') {
        throw new Error('expected restored source pixels');
      }
      const expectedCalls = structuredClone((expectedCache.surface as StubRasterSurface).callLog);
      const adjustedDeletesBefore = adjustedSurfaceCacheDeletes.length;
      const thumbnailListener = vi.fn();
      const unsubscribeThumbnail = engine.stores.thumbnailVersion.subscribe(thumbnailListener);
      bitmapStore.markLayerDirty.mockClear();
      (store.dispatch as Mock).mockClear();
      faults.arm(failure);

      expect(() => engine.history.redo()).toThrow(`replay cache ${failure} failed`);

      expect(engine.document.getDocument()).toEqual(expectedDocument);
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(true);
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

      engine.history.redo();
      expect(engine.document.getDocument()).toEqual(filteredDocument);
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(engine.stores.canRedo.get()).toBe(false);
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

    engine.history.undo();
    bitmapStore.markLayerDirty.mockClear();
    engine.history.redo();

    expect(bitmapStore.discardLayer).toHaveBeenCalledTimes(2);
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    const redone = engine.document.getDocument()!.layers[0];
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

    const drainUntil = async (predicate: () => boolean, maxTicks = 50): Promise<void> => {
      for (let i = 0; i < maxTicks && !predicate(); i += 1) {
        await Promise.resolve();
      }
    };

    const createRealBitmapStoreHarness = async (
      uploadImage: (blob: Blob) => Promise<{ height: number; imageName: string; width: number }>
    ) => {
      vi.stubGlobal('Path2D', class FakePath2D {});
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
        ...createRecordingRasterBackend(),
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
          const layer = store
            .getState()
            .projects.find((project) => project.id === projectId)
            ?.canvas.document.layers.find((candidate) => candidate.id === layerId);
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
      expect(harness.engine.document.getDocument()!.layers[0]).toMatchObject({
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

    it('invalidates a deferred pre-filter upload before it can overwrite the durable replace', async () => {
      vi.useFakeTimers();
      const uploaded = createDeferred<{ height: number; imageName: string; width: number }>();
      const uploadImage = vi.fn(() => uploaded.promise);
      const harness = await createRealBitmapStoreHarness(uploadImage);
      const barrier = harness.bitmapStore.flushPendingUploads();
      await drainUntil(() => uploadImage.mock.calls.length === 1);
      expect(uploadImage).toHaveBeenCalledOnce();

      await expect(
        harness.engine.layers.commitRasterFilterResult({
          guard: harness.exported.guard,
          image: durableImage,
          mode: 'replace',
          rect: harness.exported.rect,
        })
      ).resolves.toEqual({ layerId: harness.source.id, status: 'committed' });
      const committed = structuredClone(harness.engine.document.getDocument()!);
      uploaded.resolve({ height: 10, imageName: 'stale-in-flight-paint', width: 10 });
      await barrier;
      await vi.advanceTimersByTimeAsync(3000);

      expect(bitmapUpdateActions(harness.store)).toHaveLength(0);
      expect(harness.engine.document.getDocument()).toEqual(committed);
      const cache = await harness.engine.exports.exportLayerPixels(harness.source.id);
      expect(cache.status).toBe('ok');
      if (cache.status === 'ok') {
        expect(drawGraphContains(cache.surface as StubRasterSurface, harness.backend, 'bitmap-filtered-result')).toBe(
          true
        );
      }
      harness.engine.lifecycle.dispose();
    });

    it('redo discards a deferred upload from the restored unpersisted paint snapshot', async () => {
      vi.useFakeTimers();
      const uploaded = createDeferred<{ height: number; imageName: string; width: number }>();
      const uploadImage = vi.fn(() => uploaded.promise);
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

      harness.engine.history.undo();
      await harness.refreshPlacement();
      const barrier = harness.bitmapStore.flushPendingUploads();
      await drainUntil(() => uploadImage.mock.calls.length === 1);
      expect(uploadImage).toHaveBeenCalledOnce();

      harness.engine.history.redo();
      await harness.refreshPlacement();
      expect(harness.engine.document.getDocument()).toEqual(committed);
      uploaded.resolve({ height: 10, imageName: 'stale-undo-paint', width: 10 });
      await barrier;
      await vi.advanceTimersByTimeAsync(3000);

      expect(harness.discardLayer).toHaveBeenCalledTimes(2);
      expect(bitmapUpdateActions(harness.store)).toHaveLength(0);
      expect(harness.engine.document.getDocument()).toEqual(committed);
      expect(harness.engine.stores.canUndo.get()).toBe(true);
      expect(harness.engine.stores.canRedo.get()).toBe(false);
      const redone = harness.engine.document.getDocument()!.layers[0];
      expect(redone).toMatchObject({ source: { bitmap: durableImage, type: 'paint' } });
      harness.engine.lifecycle.dispose();
    });

    it('copy undo generation-cancels a deferred upload before redo reuses the layer id', async () => {
      vi.useFakeTimers();
      const uploaded = createDeferred<{ height: number; imageName: string; width: number }>();
      const uploadImage = vi.fn(() => uploaded.promise);
      const harness = await createRealBitmapStoreHarness(uploadImage);
      // The harness creates live source paint to establish a guarded export;
      // this test targets only persistence owned by the subsequently-created copy.
      harness.bitmapStore.discardLayer(harness.source.id);
      const copied = await harness.engine.layers.commitRasterFilterResult({
        guard: harness.exported.guard,
        image: durableImage,
        mode: 'copy',
        rect: harness.exported.rect,
      });
      if (copied.status !== 'committed') {
        throw new Error('expected a committed filtered copy');
      }
      const durableCopy = structuredClone(
        harness.engine.document.getDocument()!.layers.find((layer) => layer.id === copied.layerId)!
      );

      harness.engine.selection.selectAll();
      harness.engine.selection.fillSelection();
      await harness.refreshPlacement(copied.layerId);
      const barrier = harness.bitmapStore.flushPendingUploads();
      await drainUntil(() => uploadImage.mock.calls.length === 1);
      expect(uploadImage).toHaveBeenCalledOnce();

      // Undo the paint edit, then undo the copy itself. Redo reuses the exact
      // layer id while the pre-undo upload is still unresolved.
      harness.engine.history.undo();
      await harness.refreshPlacement(copied.layerId);
      harness.engine.history.undo();
      expect((await harness.engine.exports.exportLayerPixels(copied.layerId)).status).toBe('missing');
      harness.engine.history.redo();
      expect(harness.engine.document.getDocument()!.layers.find((layer) => layer.id === copied.layerId)).toEqual(
        durableCopy
      );

      uploaded.resolve({ height: 10, imageName: 'stale-copy-paint', width: 10 });
      await barrier;
      await vi.advanceTimersByTimeAsync(3000);

      const staleUpdates = (harness.store.dispatch as Mock).mock.calls
        .map((call) => call[0] as EngineTestAction)
        .filter((action) => action.type === 'updateCanvasLayerSource' && action.id === copied.layerId);
      expect(staleUpdates).toHaveLength(0);
      expect(harness.discardLayer).toHaveBeenCalledWith(copied.layerId);
      expect(harness.engine.document.getDocument()!.layers.find((layer) => layer.id === copied.layerId)).toEqual(
        durableCopy
      );
      expect(harness.engine.stores.canUndo.get()).toBe(true);
      const cache = await harness.engine.exports.exportLayerPixels(copied.layerId);
      expect(cache.status).toBe('ok');
      if (cache.status === 'ok') {
        expect(drawGraphContains(cache.surface as StubRasterSurface, harness.backend, 'bitmap-filtered-result')).toBe(
          true
        );
      }
      harness.engine.lifecycle.dispose();
    });
  });

  it.each(['allocation', 'draw'] as const)(
    'keeps copy redo retryable and exact when detached cache %s fails',
    async (failure) => {
      const source = filterLayer('source');
      const selected = filterLayer('selected');
      const beforeDocument = { ...filterDoc([source, selected]), selectedLayerId: selected.id };
      const { projectId, store } = createReducerBackedStore(beforeDocument);
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
        mode: 'copy',
        rect: exported.rect,
      });
      if (result.status !== 'committed') {
        throw new Error('expected a committed copy');
      }
      const copy = structuredClone(engine.document.getDocument()!.layers[0]!);
      engine.history.undo();
      expect(engine.document.getDocument()).toEqual(beforeDocument);
      const adjustedDeletesBefore = adjustedSurfaceCacheDeletes.length;
      const thumbnailListener = vi.fn();
      const unsubscribeThumbnail = engine.stores.thumbnailVersion.subscribe(thumbnailListener);
      bitmapStore.markLayerDirty.mockClear();
      (store.dispatch as Mock).mockClear();
      faults.arm(failure);

      expect(() => engine.history.redo()).toThrow(`replay cache ${failure} failed`);

      expect(engine.document.getDocument()).toEqual(beforeDocument);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
      expect((await engine.exports.exportLayerPixels(result.layerId)).status).toBe('missing');
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(true);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(thumbnailListener).not.toHaveBeenCalled();
      expect(adjustedSurfaceCacheDeletes).toHaveLength(adjustedDeletesBefore);

      engine.history.redo();
      expect(engine.document.getDocument()!.layers[0]).toEqual(copy);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
      expect((await engine.exports.exportLayerPixels(result.layerId)).status).toBe('ok');
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(engine.stores.canRedo.get()).toBe(false);
      unsubscribeThumbnail();
      engine.lifecycle.dispose();
    }
  );

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

  it('returns aborted without starting durable-image decode when already cancelled', async () => {
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const harness = await createGuardHarness(resolver);
    const controller = new AbortController();
    controller.abort();

    await expect(harness.commit(controller.signal)).resolves.toEqual({ status: 'aborted' });
    expect(resolver).not.toHaveBeenCalled();
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('passes the exact cancellation signal into durable-image resolution', async () => {
    const resolver = vi.fn((_imageName: string, _signal?: AbortSignal) => Promise.resolve(new Blob()));
    const layer = filterLayer();
    const reducer = createReducerBackedStore(filterDoc([layer]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: resolver,
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const exported = await engine.exports.exportLayerPixels(layer.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable export');
    }
    const controller = new AbortController();

    await expect(
      engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode: 'replace',
        rect: exported.rect,
        signal: controller.signal,
      })
    ).resolves.toMatchObject({ status: 'committed' });
    expect(resolver).toHaveBeenCalledWith(durableImage.imageName, controller.signal);
    engine.lifecycle.dispose();
  });

  it('maps an abort rejection from durable-image resolution to aborted without mutation', async () => {
    const resolver = vi.fn(() => Promise.reject(new DOMException('cancelled', 'AbortError')));
    const harness = await createGuardHarness(resolver);
    const controller = new AbortController();

    await expect(harness.commit(controller.signal)).resolves.toEqual({ status: 'aborted' });
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

  it('returns missing without mutation when the source is deleted', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [] });
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'missing' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('returns stale without mutation when the source contract changes', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [{ ...harness.layer, opacity: 0.2 }] });
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'stale' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('returns locked without mutation when the source becomes locked', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [{ ...harness.layer, isLocked: true }] });
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'locked' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('returns stale without mutation when the guarded source changes layer type', async () => {
    const harness = await createPendingGuardHarness();
    const { adjustments: _adjustments, ...base } = harness.layer;
    const control: CanvasLayerContract = {
      ...base,
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      type: 'control',
      withTransparencyEffect: false,
    };
    harness.setDocument({ ...harness.document, layers: [control] });
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'stale' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('returns stale without mutation after the source cache version changes', async () => {
    const harness = await createPendingGuardHarness();
    await harness.engine.diagnostics.clearCaches();
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'stale' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('returns stale without mutation after document replacement reuses the source layer', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, height: 200, width: 200 }, 1);
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'stale' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('commits to the bound project after the active project changes', async () => {
    const harness = await createPendingGuardHarness();
    harness.setActiveProjectId('p2');
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ layerId: 'L', status: 'committed' });
    expect(harness.store.dispatch).toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('returns busy without mutation while a pointer gesture is open', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const harness = await createPendingGuardHarness();
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    harness.engine.surface.attach(screen.element, overlay.element);
    overlay.fire('pointerdown', pointerAt(5, 5));
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'busy' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it.each(['replace', 'copy'] as const)(
    'leaves document, cache, selection, history, and side effects exact when %s cache preparation fails',
    async (mode) => {
      const baseBackend = createTestStubRasterBackend();
      let successfulCreatesBeforeFailure: number | null = null;
      const backend: StubRasterBackend = {
        ...baseBackend,
        createSurface: (width, height) => {
          if (successfulCreatesBeforeFailure !== null) {
            if (successfulCreatesBeforeFailure === 0) {
              throw new Error('filter cache allocation failed');
            }
            successfulCreatesBeforeFailure -= 1;
          }
          return baseBackend.createSurface(width, height);
        },
      };
      const source = filterLayer('source');
      const selected = filterLayer('selected');
      const beforeDocument = { ...filterDoc([source, selected]), selectedLayerId: selected.id };
      const { projectId, store } = createReducerBackedStore(beforeDocument);
      const bitmapStore = createSpyBitmapStore();
      const engine = createCanvasEngine({
        backend,
        bitmapStore,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels(source.id);
      if (exported.status !== 'ok') {
        throw new Error('expected a filterable export');
      }
      const cacheSurface = exported.surface;
      const cacheCalls = structuredClone((cacheSurface as StubRasterSurface).callLog);
      const thumbnailVersion = engine.stores.thumbnailVersion.get(source.id);
      const thumbnailListener = vi.fn();
      const unsubscribeThumbnail = engine.stores.thumbnailVersion.subscribe(thumbnailListener);
      const adjustedDeletesBefore = adjustedSurfaceCacheDeletes.length;
      bitmapStore.markLayerDirty.mockClear();
      (store.dispatch as Mock).mockClear();

      // Commit allocates the decoded candidate first. Replace also captures its
      // before-pixels snapshot. The following allocation is cache preparation.
      successfulCreatesBeforeFailure = mode === 'replace' ? 2 : 1;
      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode,
        rect: exported.rect,
      });
      successfulCreatesBeforeFailure = null;

      expect(result).toEqual({ message: 'filter cache allocation failed', status: 'failed' });
      expect(engine.document.getDocument()).toEqual(beforeDocument);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(false);
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(thumbnailListener).not.toHaveBeenCalled();
      expect(engine.stores.thumbnailVersion.get(source.id)).toBe(thumbnailVersion);
      expect(adjustedSurfaceCacheDeletes).toHaveLength(adjustedDeletesBefore);

      const afterExport = await engine.exports.exportLayerPixels(source.id);
      expect(afterExport.status).toBe('ok');
      if (afterExport.status === 'ok') {
        expect(afterExport.surface).toBe(cacheSurface);
        expect(afterExport.rect).toEqual(exported.rect);
        expect(afterExport.guard.cacheVersion).toBe(exported.guard.cacheVersion);
        expect((afterExport.surface as StubRasterSurface).callLog).toEqual(cacheCalls);
      }

      unsubscribeThumbnail();
      engine.lifecycle.dispose();
    }
  );

  it.each(['replace', 'copy'] as const)(
    'leaves document, cache, selection, history, and side effects exact when %s cache preparation draw fails',
    async (mode) => {
      const baseBackend = createTestStubRasterBackend();
      let successfulDrawsBeforeFailure: number | null = null;
      const backend: StubRasterBackend = {
        ...baseBackend,
        createSurface: (width, height) => {
          const surface = baseBackend.createSurface(width, height);
          const ctx = new Proxy(surface.ctx, {
            get: (target, property, receiver) => {
              const value = Reflect.get(target, property, receiver);
              if (property !== 'drawImage' || typeof value !== 'function') {
                return value;
              }
              return (...args: unknown[]) => {
                if (successfulDrawsBeforeFailure !== null) {
                  if (successfulDrawsBeforeFailure === 0) {
                    throw new Error('filter cache draw failed');
                  }
                  successfulDrawsBeforeFailure -= 1;
                }
                return Reflect.apply(value, target, args);
              };
            },
          });
          Object.defineProperty(surface, 'ctx', { value: ctx });
          return surface;
        },
      };
      const source = filterLayer('source');
      const selected = filterLayer('selected');
      const beforeDocument = { ...filterDoc([source, selected]), selectedLayerId: selected.id };
      const { projectId, store } = createReducerBackedStore(beforeDocument);
      const bitmapStore = createSpyBitmapStore();
      const engine = createCanvasEngine({
        backend,
        bitmapStore,
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      const exported = await engine.exports.exportLayerPixels(source.id);
      if (exported.status !== 'ok') {
        throw new Error('expected a filterable export');
      }
      const cacheSurface = exported.surface;
      const cacheCalls = structuredClone((cacheSurface as StubRasterSurface).callLog);
      const thumbnailVersion = engine.stores.thumbnailVersion.get(source.id);
      const thumbnailListener = vi.fn();
      const unsubscribeThumbnail = engine.stores.thumbnailVersion.subscribe(thumbnailListener);
      const adjustedDeletesBefore = adjustedSurfaceCacheDeletes.length;
      bitmapStore.markLayerDirty.mockClear();
      (store.dispatch as Mock).mockClear();

      // Commit draws the decoded candidate first. Replace also captures its
      // before-pixels snapshot. The following draw prepares the detached cache.
      successfulDrawsBeforeFailure = mode === 'replace' ? 2 : 1;
      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode,
        rect: exported.rect,
      });
      successfulDrawsBeforeFailure = null;

      expect(result).toEqual({ message: 'filter cache draw failed', status: 'failed' });
      expect(engine.document.getDocument()).toEqual(beforeDocument);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(false);
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(thumbnailListener).not.toHaveBeenCalled();
      expect(engine.stores.thumbnailVersion.get(source.id)).toBe(thumbnailVersion);
      expect(adjustedSurfaceCacheDeletes).toHaveLength(adjustedDeletesBefore);

      const afterExport = await engine.exports.exportLayerPixels(source.id);
      expect(afterExport.status).toBe('ok');
      if (afterExport.status === 'ok') {
        expect(afterExport.surface).toBe(cacheSurface);
        expect(afterExport.rect).toEqual(exported.rect);
        expect(afterExport.guard.cacheVersion).toBe(exported.guard.cacheVersion);
        expect((afterExport.surface as StubRasterSurface).callLog).toEqual(cacheCalls);
      }

      unsubscribeThumbnail();
      engine.lifecycle.dispose();
    }
  );

  it.each(['replace', 'copy'] as const)(
    'commits %s after dispatch even when render scheduling throws, without redundant persistence',
    async (mode) => {
      const raf = createControllableRaf();
      vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
      vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
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
      engine.surface.attach(createFakeCanvas().element, createFakeCanvas().element);
      raf.flush();
      bitmapStore.markLayerDirty.mockImplementation(() => {
        throw new Error('unexpected filter result persistence');
      });
      vi.stubGlobal('requestAnimationFrame', () => {
        throw new Error('render scheduling failed');
      });

      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode,
        rect: exported.rect,
      });

      if (result.status !== 'committed') {
        throw new Error(`expected a committed result, received ${JSON.stringify(result)}`);
      }
      expect(result).toMatchObject({ status: 'committed' });
      expect(store.dispatch).toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      engine.lifecycle.dispose();
    }
  );

  it.each(['replace', 'copy'] as const)(
    'commits %s when a thumbnail observer throws, without redundant persistence',
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
      const unsubscribeThumbnail = engine.stores.thumbnailVersion.subscribe(() => {
        throw new Error('thumbnail observer failed');
      });
      bitmapStore.markLayerDirty.mockImplementation(() => {
        throw new Error('unexpected filter result persistence');
      });

      const result = await engine.layers.commitRasterFilterResult({
        guard: exported.guard,
        image: durableImage,
        mode,
        rect: exported.rect,
      });

      expect(result).toMatchObject({ status: 'committed' });
      expect(store.dispatch).toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      unsubscribeThumbnail();
      engine.lifecycle.dispose();
    }
  );

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

  it('refreshes the mirror and renders the filtered result when an earlier document observer interrupts dispatch', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const source = filterLayer('source');
    const { projectId, store } = createReducerBackedStore(filterDoc([source]));
    const unsubscribeFault = store.subscribe(() => {
      throw new Error('earlier document observer failed');
    });
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(recordingBitmap('filtered-result')));
    const engine = createCanvasEngine({
      backend,
      bitmapStore: createSpyBitmapStore(),
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
    unsubscribeFault();

    expect(result).toEqual({ layerId: source.id, status: 'committed' });
    const reducerDocument = store.getState().projects.find((project) => project.id === projectId)!.canvas.document;
    expect(reducerDocument.layers[0]).toMatchObject({
      source: { bitmap: durableImage, type: 'paint' },
    });
    expect(engine.document.getDocument()).toBe(reducerDocument);
    expect(engine.stores.canUndo.get()).toBe(true);
    const committedExport = await engine.exports.exportLayerPixels(source.id);
    expect(committedExport.status).toBe('ok');
    if (committedExport.status === 'ok') {
      expect(drawGraphContains(committedExport.surface as StubRasterSurface, backend, 'bitmap-filtered-result')).toBe(
        true
      );
    }

    const screen = createFakeCanvas();
    engine.surface.attach(screen.element, createFakeCanvas().element);
    raf.flush();
    expect(drawGraphContains(screen.surface, backend, 'bitmap-filtered-result')).toBe(true);
    engine.lifecycle.dispose();
  });

  it('finishes copy undo and restores the exact prior selection after an applied observer fault', async () => {
    const source = filterLayer('source');
    const selected = filterLayer('selected');
    const beforeDocument = { ...filterDoc([source, selected]), selectedLayerId: selected.id };
    const { projectId, store } = createReducerBackedStore(beforeDocument);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
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
      mode: 'copy',
      rect: exported.rect,
    });
    if (result.status !== 'committed') {
      throw new Error('expected a committed copy');
    }
    const copy = structuredClone(engine.document.getDocument()!.layers[0]!);
    const unsubscribeFault = store.subscribe(() => {
      throw new Error('copy undo observer failed');
    });

    expect(() => engine.history.undo()).not.toThrow();

    expect(engine.document.getDocument()).toEqual(beforeDocument);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
    expect((await engine.exports.exportLayerPixels(result.layerId)).status).toBe('missing');
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    unsubscribeFault();

    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(copy);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('keeps copy undo retryable when removal fails after restoring the prior selection', async () => {
    const source = filterLayer('source');
    const selected = filterLayer('selected');
    const beforeDocument = { ...filterDoc([source, selected]), selectedLayerId: selected.id };
    const { dispatch, projectId, store } = createReducerBackedStore(beforeDocument);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
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
      mode: 'copy',
      rect: exported.rect,
    });
    if (result.status !== 'committed') {
      throw new Error('expected a committed copy');
    }
    const reducerDispatch = dispatch.getMockImplementation();
    if (!reducerDispatch) {
      throw new Error('expected reducer-backed dispatch');
    }
    let failRemoval = true;
    dispatch.mockImplementation((action: EngineTestAction) => {
      if (failRemoval && action.type === 'removeCanvasLayers') {
        failRemoval = false;
        throw new Error('copy removal failed');
      }
      reducerDispatch(action);
    });

    expect(() => engine.history.undo()).toThrow('copy removal failed');

    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(true);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(beforeDocument);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it.each(['replace', 'copy'] as const)('commits %s when a history-state observer throws', async (mode) => {
    const source = filterLayer('source');
    const { projectId, store } = createReducerBackedStore(filterDoc([source]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected a filterable export');
    }
    const unsubscribeFault = engine.stores.canUndo.subscribe(() => {
      throw new Error('history observer failed');
    });

    const result = await engine.layers.commitRasterFilterResult({
      guard: exported.guard,
      image: durableImage,
      mode,
      rect: exported.rect,
    });

    expect(result).toMatchObject({ status: 'committed' });
    expect(engine.stores.canUndo.get()).toBe(true);
    if (result.status === 'committed') {
      expect((await engine.exports.exportLayerPixels(result.layerId)).status).toBe('ok');
    }
    unsubscribeFault();
    engine.lifecycle.dispose();
  });
});

describe('commitGeneratedImageResult', () => {
  const generatedImage = { height: 18, imageName: 'workflow-result.png', width: 24 };
  const generatedOrigin = { x: -9, y: 14 };

  const workflowRaster = (id = 'source'): CanvasRasterLayerContractV2 => ({
    adjustments: { brightness: 0.2, contrast: -0.1, saturation: 0.3 },
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

  const workflowDocument = (layers: CanvasLayerContract[] = [workflowRaster()]): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers,
    selectedLayerId: layers.at(-1)?.id ?? null,
    version: 2,
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

  it.each([
    { base: 'z-image', expectedKind: 'z_image_control' },
    { base: 'sd-1', expectedKind: 'controlnet' },
    { base: 'flux', expectedKind: 'controlnet' },
  ] as const)(
    'uses $expectedKind defaults when workflow Copy To Control runs under $base',
    async ({ base, expectedKind }) => {
      const source = workflowRaster();
      const { projectId, store } = createReducerBackedStore(workflowDocument([source]), base);
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
        target: 'copy-control',
      });

      expect(result.status).toBe('committed');
      if (result.status !== 'committed') {
        throw new Error('expected a committed control copy');
      }
      const created = engine.document.getDocument()!.layers.find((layer) => layer.id === result.layerId);
      expect(created?.type === 'control' ? created.adapter.kind : null).toBe(expectedKind);
      engine.lifecycle.dispose();
    }
  );

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
    const after = engine.document.getDocument()!.layers[0]!;
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

    engine.history.undo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(after);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    expect(bitmapStore.discardLayer).toHaveBeenCalledTimes(2);
    engine.lifecycle.dispose();
  });

  it('replaces a control while preserving adapter, filter, transparency effect, and layer presentation', async () => {
    const source = workflowControl();
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

    expect(engine.document.getDocument()!.layers[0]).toEqual({
      ...source,
      source: { bitmap: generatedImage, offset: generatedOrigin, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    engine.history.undo();
    expect(engine.document.getDocument()!.layers[0]).toEqual(source);
    engine.history.redo();
    expect(engine.document.getDocument()!.layers[0]).toMatchObject({
      adapter: source.type === 'control' ? source.adapter : undefined,
      filter: source.type === 'control' ? source.filter : undefined,
      withTransparencyEffect: true,
    });
    engine.lifecycle.dispose();
  });

  it('copies a generated result to an unlocked raster immediately above its source and restores selection on undo', async () => {
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

    expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual([
      above.id,
      result.layerId,
      source.id,
      selected.id,
    ]);
    expect(engine.document.getDocument()!.layers[2]).toEqual(source);
    expect(engine.document.getDocument()!.layers[1]).toMatchObject({
      id: result.layerId,
      isEnabled: true,
      isLocked: false,
      opacity: 1,
      source: { bitmap: generatedImage, offset: generatedOrigin, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.history.redo();
    expect(engine.document.getDocument()!.layers[1]?.id).toBe(result.layerId);
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

  it('returns missing after decode when the source was deleted', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [] });
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status: 'missing' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns locked after decode when the source became locked', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [{ ...harness.source, isLocked: true }] });
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status: 'locked' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns unsupported after decode when the source became a mask', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [maskLayer(harness.source.id)] });
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status: 'unsupported' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns stale after decode when the immutable source contract changed', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, layers: [{ ...harness.source, opacity: 0.1 }] });
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status: 'stale' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns stale after decode when the document was replaced with the same source object', async () => {
    const harness = await createPendingGuardHarness();
    harness.setDocument({ ...harness.document, width: 200 }, 1);
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status: 'stale' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('commits after decode when another project became active', async () => {
    const harness = await createPendingGuardHarness();
    harness.setActiveProjectId('p2');
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ layerId: 'source', status: 'committed' });
    expect(harness.store.dispatch).toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns busy after decode while a pointer gesture is open', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const harness = await createPendingGuardHarness();
    const overlay = createInputCanvas();
    harness.engine.surface.attach(createInputCanvas().element, overlay.element);
    overlay.fire('pointerdown', pointerAt(5, 5));
    harness.decoded.resolve(new Blob());
    await expect(harness.pending).resolves.toEqual({ status: 'busy' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('does not publish when an operation starts during generated-image decode', async () => {
    const harness = await createPendingGuardHarness();
    const cleanupPreview = vi.fn();
    getCanvasOperations(harness.engine).controller.start({
      cleanupPreview,
      guard: harness.exported.guard,
      identity: { kind: 'filter', layerId: harness.source.id, projectId: 'p1' },
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

  it('does not publish when an operation starts and ends during generated-image decode', async () => {
    const harness = await createPendingGuardHarness();
    const operation = getCanvasOperations(harness.engine).controller.start({
      cleanupPreview: vi.fn(),
      guard: harness.exported.guard,
      identity: { kind: 'filter', layerId: harness.source.id, projectId: 'p1' },
    });
    operation?.cancel();

    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'busy' });
    expect(harness.engine.document.getDocument()).toEqual(harness.document);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    expect(getCanvasOperations(harness.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    harness.engine.lifecycle.dispose();
  });

  it('returns aborted without resolving the generated image when already cancelled', async () => {
    const source = workflowRaster();
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const { projectId, store } = createReducerBackedStore(workflowDocument([source]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: resolver,
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    resolver.mockClear();
    const controller = new AbortController();
    controller.abort();

    await expect(
      engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        signal: controller.signal,
        target: 'replace',
      })
    ).resolves.toEqual({ status: 'aborted' });
    expect(resolver).not.toHaveBeenCalled();
    expect(engine.document.getDocument()).toEqual(workflowDocument([source]));
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
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

  it('returns stale when unpersisted paint pixels change while the generated image resolves', async () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
    const source: CanvasRasterLayerContractV2 = {
      ...workflowRaster(),
      source: {
        bitmap: { height: 10, imageName: 'workflow-paint-source.png', width: 10 },
        type: 'paint',
      },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    };
    const generatedBlob = createDeferred<Blob>();
    const { projectId, store } = createReducerBackedStore(workflowDocument([source]));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: (imageName) =>
        imageName === generatedImage.imageName ? generatedBlob.promise : Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    const pending = engine.layers.commitGeneratedImageResult({
      guard: exported.guard,
      image: generatedImage,
      origin: generatedOrigin,
      target: 'replace',
    });

    engine.selection.selectAll();
    engine.selection.fillSelection();
    const afterPaintEdit = structuredClone(engine.document.getDocument()!);
    generatedBlob.resolve(new Blob());

    await expect(pending).resolves.toEqual({ status: 'stale' });
    expect(engine.document.getDocument()).toEqual(afterPaintEdit);
    expect(engine.document.getDocument()!.layers[0]).not.toMatchObject({ source: { bitmap: generatedImage } });
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it.each([
    { successfulCreates: 0, target: 'replace' as const },
    { successfulCreates: 2, target: 'replace' as const },
    { successfulCreates: 1, target: 'copy-raster' as const },
  ])(
    'leaves document, cache, selection, and history exact when $target allocation fails after $successfulCreates successful creates',
    async ({ successfulCreates, target }) => {
      const source = workflowRaster();
      const selected = workflowRaster('selected');
      const document = { ...workflowDocument([source, selected]), selectedLayerId: selected.id };
      const faults = createGeneratedFaultBackend();
      const bitmap = recordingBitmap('allocation-failure');
      faults.backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
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
      const cache = exported.surface;
      const cacheCalls = structuredClone((cache as StubRasterSurface).callLog);
      (store.dispatch as Mock).mockClear();
      faults.armAllocation(successfulCreates);

      const result = await engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        target,
      });

      expect(result).toEqual({ message: 'generated cache allocation failed', status: 'failed' });
      expect(bitmap.close).toHaveBeenCalledOnce();
      expect(engine.document.getDocument()).toEqual(document);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(false);
      const after = await engine.exports.exportLayerPixels(source.id);
      expect(after.status).toBe('ok');
      if (after.status === 'ok') {
        expect(after.surface).toBe(cache);
        expect(after.guard.cacheVersion).toBe(exported.guard.cacheVersion);
        expect((after.surface as StubRasterSurface).callLog).toEqual(cacheCalls);
      }
      engine.lifecycle.dispose();
    }
  );

  it.each([
    { successfulDraws: 0, target: 'replace' as const },
    { successfulDraws: 2, target: 'replace' as const },
    { successfulDraws: 1, target: 'copy-raster' as const },
  ])(
    'leaves document, cache, selection, and history exact when $target draw fails after $successfulDraws successful draws',
    async ({ successfulDraws, target }) => {
      const source = workflowRaster();
      const selected = workflowRaster('selected');
      const document = { ...workflowDocument([source, selected]), selectedLayerId: selected.id };
      const faults = createGeneratedFaultBackend();
      const bitmap = recordingBitmap('draw-failure');
      faults.backend.createImageBitmap = vi.fn(() => Promise.resolve(bitmap));
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
      const cache = exported.surface;
      const cacheCalls = structuredClone((cache as StubRasterSurface).callLog);
      (store.dispatch as Mock).mockClear();
      faults.armDraw(successfulDraws);

      const result = await engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        target,
      });

      expect(result).toEqual({ message: 'generated cache draw failed', status: 'failed' });
      expect(bitmap.close).toHaveBeenCalledOnce();
      expect(engine.document.getDocument()).toEqual(document);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(false);
      const after = await engine.exports.exportLayerPixels(source.id);
      expect(after.status).toBe('ok');
      if (after.status === 'ok') {
        expect(after.surface).toBe(cache);
        expect(after.guard.cacheVersion).toBe(exported.guard.cacheVersion);
        expect((after.surface as StubRasterSurface).callLog).toEqual(cacheCalls);
      }
      engine.lifecycle.dispose();
    }
  );

  it.each([
    { successfulCreates: 2, target: 'replace' as const },
    { successfulCreates: 1, target: 'copy-raster' as const },
  ])(
    'returns aborted without mutation when $target is cancelled during cache preparation',
    async ({ successfulCreates, target }) => {
      const source = workflowRaster();
      const document = workflowDocument([source]);
      const faults = createGeneratedFaultBackend();
      const controller = new AbortController();
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
      (store.dispatch as Mock).mockClear();
      faults.armCreateHook(successfulCreates, () => controller.abort());

      await expect(
        engine.layers.commitGeneratedImageResult({
          guard: exported.guard,
          image: generatedImage,
          origin: generatedOrigin,
          signal: controller.signal,
          target,
        })
      ).resolves.toEqual({ status: 'aborted' });
      expect(engine.document.getDocument()).toEqual(document);
      expect(store.dispatch).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it.each([
    { successfulCreates: 2, target: 'replace' as const },
    { successfulCreates: 1, target: 'copy-raster' as const },
  ])('revalidates the exact guard immediately before $target publication', async ({ successfulCreates, target }) => {
    const source = workflowRaster();
    const document = workflowDocument([source]);
    const reactive = createReactiveStore(document);
    const faults = createGeneratedFaultBackend();
    const engine = createCanvasEngine({
      backend: faults.backend,
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store: reactive.store,
    });
    const exported = await engine.exports.exportLayerPixels(source.id);
    if (exported.status !== 'ok') {
      throw new Error('expected an exportable workflow source');
    }
    (reactive.store.dispatch as Mock).mockClear();
    faults.armCreateHook(successfulCreates, () => {
      reactive.setDocument({ ...document, layers: [{ ...source, opacity: 0.2 }] });
    });

    await expect(
      engine.layers.commitGeneratedImageResult({
        guard: exported.guard,
        image: generatedImage,
        origin: generatedOrigin,
        target,
      })
    ).resolves.toEqual({ status: 'stale' });
    expect(reactive.store.dispatch).not.toHaveBeenCalled();
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
    expect(() => engine.history.undo()).toThrow('generated cache allocation failed');
    expect(engine.document.getDocument()).toEqual(committed);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    faults.armDraw(0);
    expect(() => engine.history.redo()).toThrow('generated cache draw failed');
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.history.redo();
    expect(engine.document.getDocument()).toEqual(committed);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('keeps copy redo retryable, absent, and selection-exact when detached cache preparation fails', async () => {
    const source = workflowRaster();
    const selected = workflowRaster('selected');
    const document = { ...workflowDocument([source, selected]), selectedLayerId: selected.id };
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
    const result = await engine.layers.commitGeneratedImageResult({
      guard: exported.guard,
      image: generatedImage,
      origin: generatedOrigin,
      target: 'copy-raster',
    });
    if (result.status !== 'committed') {
      throw new Error('expected a committed workflow copy');
    }
    const committed = structuredClone(engine.document.getDocument()!);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    faults.armAllocation(0);
    expect(() => engine.history.redo()).toThrow('generated cache allocation failed');
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(selected.id);
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.history.redo();
    expect(engine.document.getDocument()).toEqual(committed);
    expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
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

  const sourceDocument = (layer: CanvasLayerContract = sourceLayer()): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers: [layer],
    selectedLayerId: layer.id,
    version: 2,
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
    vi.stubGlobal('Path2D', class FakePath2D {});
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

  it('reports selected when a has-selection observer throws after the replacement is applied', async () => {
    const harness = await createHarness({ alpha: 255, seedSelection: false });
    const unsubscribeFault = harness.engine.stores.hasSelection.subscribe(() => {
      throw new Error('selection state observer failed');
    });

    await expect(
      harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect)
    ).resolves.toEqual({
      status: 'selected',
    });

    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    unsubscribeFault();
    harness.engine.lifecycle.dispose();
  });

  it('reports selected and clears derived selection state for an empty mask without clearing the prior surface', async () => {
    const base = alphaBackend(0);
    const surfaces: StubRasterSurface[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (width, height) => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    };
    const harness = await createHarness({ backend });
    const priorMask = surfaces.at(-1);
    if (!priorMask) {
      throw new Error('expected a seeded selection surface');
    }
    const priorMaskClear = vi.fn(() => {
      throw new Error('prior selection surface must not be cleared');
    });
    Object.defineProperty(priorMask.ctx, 'clearRect', { value: priorMaskClear });

    await expect(
      harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect)
    ).resolves.toEqual({
      status: 'selected',
    });

    expect(priorMaskClear).not.toHaveBeenCalled();
    expect(harness.engine.stores.hasSelection.get()).toBe(false);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('closes the decoded bitmap and preserves selection when replacement-surface allocation fails', async () => {
    const base = alphaBackend(255);
    const close = vi.fn();
    let failAllocation = false;
    const backend: StubRasterBackend = {
      ...base,
      createImageBitmap: vi.fn(() =>
        Promise.resolve({ close, height: resultImage.height, width: resultImage.width } as unknown as ImageBitmap)
      ),
      createSurface: (width, height) => {
        if (failAllocation) {
          throw new Error('selection surface allocation failed');
        }
        return base.createSurface(width, height);
      },
    };
    const harness = await createHarness({ backend });
    failAllocation = true;

    await expect(
      harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect)
    ).resolves.toEqual({
      message: 'selection surface allocation failed',
      status: 'failed',
    });

    expect(close).toHaveBeenCalledOnce();
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns aborted before decode without touching the existing selection', async () => {
    const imageResolver = vi.fn(() => Promise.resolve(new Blob()));
    const harness = await createHarness({ imageResolver });
    const controller = new AbortController();
    controller.abort();

    await expect(
      harness.engine.selection.replaceSelectionFromImage(harness.guard, resultImage, resultRect, controller.signal)
    ).resolves.toEqual({ status: 'aborted' });
    expect(imageResolver).not.toHaveBeenCalled();
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('returns aborted when cancellation lands during image resolution', async () => {
    const decoded = createDeferred<Blob>();
    const harness = await createHarness({ imageResolver: () => decoded.promise });
    const controller = new AbortController();

    const pending = harness.engine.selection.replaceSelectionFromImage(
      harness.guard,
      resultImage,
      resultRect,
      controller.signal
    );
    controller.abort();
    decoded.resolve(new Blob());

    await expect(pending).resolves.toEqual({ status: 'aborted' });
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
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
        harness.setDocument({ ...harness.document, layers: [] }),
    },
    {
      expected: 'stale',
      label: 'source contract edit',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) =>
        harness.setDocument({ ...harness.document, layers: [{ ...harness.layer, opacity: 0.5 }] }),
    },
    {
      expected: 'locked',
      label: 'source lock',
      mutate: (harness: Awaited<ReturnType<typeof createHarness>>) =>
        harness.setDocument({ ...harness.document, layers: [{ ...harness.layer, isLocked: true }] }),
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
        harness.setDocument({ ...harness.document, layers: [mask] });
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

  it('returns busy for an open pointer gesture without mutating the selection', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const harness = await createPendingHarness();
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    harness.engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    overlay.fire('pointerdown', pointerAt(5, 5));
    harness.decoded.resolve(new Blob());

    await expect(harness.pending).resolves.toEqual({ status: 'busy' });
    expect(harness.engine.stores.hasSelection.get()).toBe(true);
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    harness.engine.lifecycle.dispose();
  });

  it('returns stale after an unpersisted paint-cache edit without mutating the selection', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const decoded = createDeferred<Blob>();
    const document = paintDoc();
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: alphaBackend(0),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => decoded.promise,
      projectId,
      store,
    });
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    overlay.fire('pointermove', pointerAt(8, 8));
    overlay.fire('pointerup', pointerAt(8, 8, { buttons: 0 }));
    const exported = await engine.exports.exportLayerPixels('paint1');
    if (exported.status !== 'ok') {
      throw new Error('expected live paint pixels');
    }
    engine.selection.selectAll();
    const pending = engine.selection.replaceSelectionFromImage(exported.guard, resultImage, resultRect);

    overlay.fire('pointerdown', pointerAt(15, 15));
    overlay.fire('pointermove', pointerAt(18, 18));
    overlay.fire('pointerup', pointerAt(18, 18, { buttons: 0 }));
    decoded.resolve(new Blob());

    await expect(pending).resolves.toEqual({ status: 'stale' });
    expect(engine.stores.hasSelection.get()).toBe(true);
    engine.lifecycle.dispose();
  });
});

describe('commitMaskImageResult', () => {
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

  const docFor = (target: 'inpaint_mask' | 'regional_guidance'): CanvasDocumentContractV2 => {
    const layer = source();
    const below = { ...source(), id: 'below', name: 'Below' };
    return {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [target === 'inpaint_mask' ? existingInpaint() : existingRegional(), layer, below],
      selectedLayerId: below.id,
      version: 2,
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
          fill: { color: '#e07575', style: 'diagonal' },
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
          fill: { color: '#fae150', style: 'solid' },
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
    'adds a complete $target directly above the source with one exact history entry',
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
      const created = engine.document.getDocument()!.layers.find((layer) => layer.id === result.layerId)!;
      expect(created).toEqual({ ...expected, id: result.layerId });
      expect(engine.document.getDocument()!.layers.map((layer) => layer.id)).toEqual([
        document.layers[0]!.id,
        result.layerId,
        'source',
        'below',
      ]);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
      expect(imageResolver).not.toHaveBeenCalled();
      expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
      expect(engine.stores.canUndo.get()).toBe(true);

      engine.history.undo();
      expect(engine.document.getDocument()).toEqual(document);
      expect(engine.document.getDocument()!.selectedLayerId).toBe('below');
      expect(engine.stores.canUndo.get()).toBe(false);
      expect(engine.stores.canRedo.get()).toBe(true);

      engine.history.redo();
      expect(engine.document.getDocument()!.layers.find((layer) => layer.id === result.layerId)).toEqual(created);
      expect(engine.document.getDocument()!.selectedLayerId).toBe(result.layerId);
      expect(engine.stores.canUndo.get()).toBe(true);
      expect(engine.stores.canRedo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it('commits and refreshes the document mirror when an earlier observer throws after the reducer adds the mask', async () => {
    const document = docFor('inpaint_mask');
    const { projectId, store } = createReducerBackedStore(document);
    const unsubscribeFault = store.subscribe(() => {
      throw new Error('earlier mask observer failed');
    });
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
    unsubscribeFault();

    expect(result.status).toBe('committed');
    if (result.status !== 'committed') {
      throw new Error('expected a committed mask');
    }
    expect(engine.document.getDocument()).toBe(
      store.getState().projects.find((project) => project.id === projectId)!.canvas.document
    );
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    engine.lifecycle.dispose();
  });

  it('finishes mask undo exactly when document observers throw after both reducer mutations', async () => {
    const document = docFor('regional_guidance');
    const { projectId, store } = createReducerBackedStore(document);
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
      target: 'regional_guidance',
    });
    if (result.status !== 'committed') {
      throw new Error('expected a committed mask');
    }
    const unsubscribeFault = store.subscribe(() => {
      throw new Error('mask undo observer failed');
    });

    expect(() => engine.history.undo()).not.toThrow();

    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.document.getDocument()!.selectedLayerId).toBe('below');
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    unsubscribeFault();
    engine.lifecycle.dispose();
  });

  it('keeps mask undo failure-atomic when restoring the prior selection fails before reducer application', async () => {
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
    let failSelection = true;
    dispatch.mockImplementation((action: EngineTestAction) => {
      if (failSelection && action.type === 'setCanvasSelectedLayer') {
        failSelection = false;
        throw new Error('mask selection restore failed');
      }
      reducerDispatch(action);
    });

    expect(() => engine.history.undo()).toThrow('mask selection restore failed');

    expect(engine.document.getDocument()).toEqual(committedDocument);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('keeps mask undo retryable when removal fails after restoring the prior selection', async () => {
    const document = docFor('regional_guidance');
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
      target: 'regional_guidance',
    });
    if (result.status !== 'committed') {
      throw new Error('expected a committed mask');
    }
    const reducerDispatch = dispatch.getMockImplementation();
    if (!reducerDispatch) {
      throw new Error('expected reducer-backed dispatch');
    }
    let failRemoval = true;
    dispatch.mockImplementation((action: EngineTestAction) => {
      if (failRemoval && action.type === 'removeCanvasLayers') {
        failRemoval = false;
        throw new Error('mask removal failed');
      }
      reducerDispatch(action);
    });

    expect(() => engine.history.undo()).toThrow('mask removal failed');

    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(true);
    expect(engine.document.getDocument()!.selectedLayerId).toBe('below');
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.history.undo();
    expect(engine.document.getDocument()).toEqual(document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  const guardHarness = async () => {
    const layer = source();
    const document: CanvasDocumentContractV2 = { ...docFor('inpaint_mask'), layers: [layer] };
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

  it('returns aborted before any mask mutation', async () => {
    const harness = await guardHarness();
    const controller = new AbortController();
    controller.abort();

    await expect(
      harness.engine.layers.commitMaskImageResult({
        guard: harness.guard,
        image: resultImage,
        rect: resultRect,
        signal: controller.signal,
        target: 'inpaint_mask',
      })
    ).resolves.toEqual({ status: 'aborted' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it.each([
    {
      expected: 'missing',
      label: 'source deletion',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) =>
        harness.setDocument({ ...harness.document, layers: [] }),
    },
    {
      expected: 'stale',
      label: 'source contract edit',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) =>
        harness.setDocument({ ...harness.document, layers: [{ ...harness.layer, opacity: 0.4 }] }),
    },
    {
      expected: 'locked',
      label: 'source lock',
      mutate: (harness: Awaited<ReturnType<typeof guardHarness>>) =>
        harness.setDocument({ ...harness.document, layers: [{ ...harness.layer, isLocked: true }] }),
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
        harness.setDocument({ ...harness.document, layers: [mask] });
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

  it('returns busy during an open pointer gesture without adding a mask', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    const harness = await guardHarness();
    const screen = createInputCanvas();
    const overlay = createInputCanvas();
    harness.engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    overlay.fire('pointerdown', pointerAt(5, 5));

    await expect(
      harness.engine.layers.commitMaskImageResult({
        guard: harness.guard,
        image: resultImage,
        rect: resultRect,
        target: 'inpaint_mask',
      })
    ).resolves.toEqual({ status: 'busy' });
    expect(harness.store.dispatch).not.toHaveBeenCalled();
    expect(harness.engine.stores.canUndo.get()).toBe(false);
    harness.engine.lifecycle.dispose();
  });
});

describe('guarded filter previews', () => {
  const spandrelModel = {
    base: 'any',
    hash: 'hash',
    key: 'upscaler',
    name: 'Upscaler',
    type: 'spandrel_image_to_image',
  };
  const emptyDoc = (): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers: [],
    selectedLayerId: null,
    version: 2,
    width: 100,
  });

  /**
   * A layer that can be added/removed from the document to drive the mirror's
   * `onLayersChanged`/`onDocumentReplaced` callbacks, without itself consuming a
   * `createImageBitmap` call — a `bitmap: null` paint source rasterizes
   * synchronously (a clear), unlike an image source, so it can't shift the call
   * ordering the pruning tests rely on to target a SPECIFIC filter-preview decode.
   */
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
    const { store } = createReactiveStore({ ...emptyDoc(), layers: [layer] });
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

  it('setFilterOperationAutoProcess toggles the session and stops auto-runs', async () => {
    const layer = guardableLayer('auto-filter-toggle');
    const { store } = createReactiveStore({ ...emptyDoc(), layers: [layer] });
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

    expect(getCanvasOperations(engine).setFilterOperationAutoProcess(false)).toBe('stale');

    expect(getCanvasOperations(engine).startFilterOperation(layer.id)).toBe('started');
    expect(getCanvasOperations(engine).setFilterOperationAutoProcess(false)).toBe('updated');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({ autoProcess: false });

    getCanvasOperations(engine).updateFilterOperation({
      settings: { high_threshold: 210, low_threshold: 90 },
      type: 'canny_edge_detection',
    });
    await new Promise((resolve) => {
      setTimeout(resolve, FILTER_AUTO_PROCESS_DEBOUNCE_MS + 200);
    });
    expect(runGraph).not.toHaveBeenCalled();

    engine.tools.setInteractionLocked(true);
    expect(getCanvasOperations(engine).setFilterOperationAutoProcess(true)).toBe('blocked');
    engine.lifecycle.dispose();
  });

  it('clears the session and ends the operation after a committed filter commit', async () => {
    const source = guardableLayer('commit-clear');
    const document = { ...emptyDoc(), layers: [source], selectedLayerId: source.id };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      bitmapStore: createSpyBitmapStore(),
      filterDeps: {
        runGraph: () => Promise.resolve({ height: 10, imageName: 'out.png', origin: 'test', output: {}, width: 10 }),
        uploadIntermediate: () => Promise.resolve({ imageName: 'input.png' }),
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels(source.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');
    await expect(getCanvasOperations(engine).processFilterOperation()).resolves.toBe('completed');

    await expect(getCanvasOperations(engine).commitFilterOperation('apply', () => Promise.resolve())).resolves.toBe(
      'committed'
    );

    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');
    engine.lifecycle.dispose();
  });

  it.each(['apply', 'raster'] as const)(
    'rejects a decoded dimension mismatch for canny %s without scaling, then retries successfully',
    async (target) => {
      const source = guardableLayer('dimension-source');
      const document = { ...emptyDoc(), layers: [source], selectedLayerId: source.id };
      const { projectId, store } = createReducerBackedStore(document);
      const base = createTestStubRasterBackend();
      const decodedDimensions = [
        { height: 10, width: 9 },
        { height: 10, width: 10 },
        { height: 9, width: 10 },
        { height: 10, width: 10 },
      ];
      const decodedBitmaps: ImageBitmap[] = [];
      const surfaces: StubRasterSurface[] = [];
      const backend: StubRasterBackend = {
        ...base,
        createImageBitmap: () => {
          const dimensions = decodedDimensions.shift();
          if (!dimensions) {
            throw new Error('unexpected filter decode');
          }
          const bitmap = { ...dimensions, close: vi.fn(), decodedFilter: true } as unknown as ImageBitmap;
          decodedBitmaps.push(bitmap);
          return Promise.resolve(bitmap);
        },
        createSurface: (width, height) => {
          const surface = base.createSurface(width, height);
          surfaces.push(surface);
          return surface;
        },
      };
      const engine = createCanvasEngine({
        backend,
        bitmapStore: createSpyBitmapStore(),
        filterDeps: {
          runGraph: () =>
            Promise.resolve({ height: 10, imageName: 'canny.png', origin: 'test', output: {}, width: 10 }),
          uploadIntermediate: () => Promise.resolve({ imageName: 'input.png' }),
        },
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      expect((await engine.exports.exportLayerPixels(source.id)).status).toBe('ok');
      expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');

      await expect(getCanvasOperations(engine).processFilterOperation()).resolves.toBe('stale');
      expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
        error: 'Canny Edge Detection output dimensions 9x10 do not match source dimensions 10x10.',
        preview: null,
        status: 'error',
      });
      expect(engine.document.getDocument()).toEqual(document);

      await expect(getCanvasOperations(engine).processFilterOperation()).resolves.toBe('completed');
      await expect(getCanvasOperations(engine).commitFilterOperation(target, () => Promise.resolve())).resolves.toBe(
        'stale'
      );
      expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
        error: 'Canny Edge Detection output dimensions 10x9 do not match source dimensions 10x10.',
        preview: { imageName: 'canny.png' },
        status: 'error',
      });
      expect(engine.document.getDocument()).toEqual(document);

      await expect(getCanvasOperations(engine).commitFilterOperation(target, () => Promise.resolve())).resolves.toBe(
        'committed'
      );

      const bitmapDraws = surfaces.flatMap((surface) =>
        surface.callLog.filter(
          (entry) => entry.op === 'drawImage' && decodedBitmaps.includes(entry.args[0] as ImageBitmap)
        )
      );
      expect(bitmapDraws.length).toBeGreaterThan(0);
      expect(bitmapDraws.every((entry) => entry.args.length === 3)).toBe(true);
      expect(engine.document.getDocument()?.layers).toHaveLength(target === 'apply' ? 1 : 2);
      engine.lifecycle.dispose();
    }
  );

  it('interaction lock blocks Filter and Select Object launches without creating sessions', async () => {
    const layer = guardableLayer('locked-launch');
    const { store } = createReactiveStore({ ...emptyDoc(), layers: [layer] });
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels(layer.id)).status).toBe('ok');

    engine.tools.setInteractionLocked(true);

    expect(getCanvasOperations(engine).startFilterOperation(layer.id)).toBe('locked');
    expect(getCanvasOperations(engine).startSelectObject(layer.id)).toBe('locked');
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).stores.samSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('keeps an active Select Object operation available to cancel while interaction is locked', async () => {
    const layer = guardableLayer('active-before-lock');
    const { store } = createReactiveStore({ ...emptyDoc(), layers: [layer] });
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels(layer.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startSelectObject(layer.id)).toBe('started');

    engine.tools.setInteractionLocked(true);

    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object' },
      status: 'active',
    });
    expect(getCanvasOperations(engine).stores.samSession.get()).not.toBeNull();
    expect(engine.stores.activeTool.get()).toBe('view');

    getCanvasOperations(engine).cancelSelectObjectSession();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(getCanvasOperations(engine).stores.samSession.get()).toBeNull();

    engine.tools.setInteractionLocked(false);
    expect(engine.stores.activeTool.get()).toBe('view');
    engine.lifecycle.dispose();
  });

  /** A stub backend whose `createImageBitmap` is deferred, so decodes resolve on demand. */
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
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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

  it('owns a guarded filter session independently of the launching view', async () => {
    const layer = { ...guardableLayer('L'), filter: { settings: { radius: 2 }, type: 'content_shuffle' } };
    const document = { ...emptyDoc(), layers: [layer] };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      filterDeps: {
        runGraph: vi.fn(() =>
          Promise.resolve({ height: 10, imageName: 'filtered', origin: 'canvas', output: {}, width: 10 })
        ),
        uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'input' })),
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('L')).status).toBe('ok');

    expect(getCanvasOperations(engine).startFilterOperation('L')).toBe('started');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      draft: layer.filter,
      initialFilter: layer.filter,
      layerId: 'L',
      layerName: layer.name,
      layerType: 'raster',
      status: 'ready',
    });
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter', layerId: 'L', projectId: 'p1' },
      status: 'active',
    });
    await getCanvasOperations(engine).processFilterOperation();
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      preview: { imageName: 'filtered' },
      status: 'ready',
    });

    engine.lifecycle.dispose();
  });

  it('starts an unfiltered layer with a recommendation but preserves an existing manual filter and Cancel', async () => {
    const unfiltered = guardableLayer('recommended');
    const manual = {
      ...guardableLayer('manual'),
      filter: { settings: { coarse: true }, type: 'lineart_edge_detection' },
    };
    const document = { ...emptyDoc(), layers: [unfiltered, manual] };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('recommended')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('recommended', 'normal_map')).toBe('started');
    expect(getCanvasOperations(engine).stores.filterSession.get()?.draft).toEqual({ settings: {}, type: 'normal_map' });
    getCanvasOperations(engine).cancelFilterOperation();
    expect(engine.document.getDocument()).toEqual({ ...document, selectedLayerId: 'recommended' });

    expect((await engine.exports.exportLayerPixels('manual')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('manual', 'normal_map')).toBe('not-ready');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).startFilterOperation('manual')).toBe('started');
    expect(getCanvasOperations(engine).stores.filterSession.get()?.draft).toEqual(manual.filter);
    getCanvasOperations(engine).cancelFilterOperation();
    expect(engine.document.getDocument()).toEqual({ ...document, selectedLayerId: 'manual' });
    engine.lifecycle.dispose();
  });

  it('does not replace another layer filter session with a recommendation', async () => {
    const first = guardableLayer('first');
    const second = guardableLayer('second');
    const document = { ...emptyDoc(), layers: [first, second] };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('first')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('first')).toBe('started');
    getCanvasOperations(engine).updateFilterOperation({ settings: { coarse: true }, type: 'lineart_edge_detection' });
    const active = getCanvasOperations(engine).stores.filterSession.get();

    expect((await engine.exports.exportLayerPixels('second')).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation('second', 'normal_map')).toBe('not-ready');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toEqual(active);
    engine.lifecycle.dispose();
  });

  it('does not overwrite a same-layer draft or preview on rapid recommendations', async () => {
    const layer = guardableLayer('recommended');
    const document = { ...emptyDoc(), layers: [layer] };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      filterDeps: {
        runGraph: vi.fn(() =>
          Promise.resolve({ height: 10, imageName: 'filtered', origin: 'canvas', output: {}, width: 10 })
        ),
        uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'input' })),
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels(layer.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation(layer.id, 'normal_map')).toBe('started');
    getCanvasOperations(engine).updateFilterOperation({ settings: { coarse: true }, type: 'lineart_edge_detection' });
    await getCanvasOperations(engine).processFilterOperation();
    const active = getCanvasOperations(engine).stores.filterSession.get();

    expect(getCanvasOperations(engine).startFilterOperation(layer.id, 'canny_edge_detection')).toBe('not-ready');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toEqual(active);
    engine.lifecycle.dispose();
  });

  it('does not replace a different active canvas operation with a recommendation', async () => {
    const first = guardableLayer('first');
    const second = guardableLayer('second');
    const document = { ...emptyDoc(), layers: [first, second] };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    expect((await engine.exports.exportLayerPixels('first')).status).toBe('ok');
    expect((await engine.exports.exportLayerPixels('second')).status).toBe('ok');
    expect(getCanvasOperations(engine).startSelectObject('first')).toBe('started');
    const operation = getCanvasOperations(engine).controller.getSnapshot();

    expect(getCanvasOperations(engine).startFilterOperation('second', 'normal_map')).toBe('not-ready');
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual(operation);
    expect(getCanvasOperations(engine).stores.samSession.get()?.sourceRect).toEqual({
      height: 10,
      width: 10,
      x: 0,
      y: 0,
    });
    engine.lifecycle.dispose();
  });

  const filterFlowLayer = (
    type: 'raster' | 'control'
  ): Extract<CanvasLayerContract, { type: 'raster' | 'control' }> => {
    const base = {
      blendMode: 'multiply' as const,
      filter: { settings: { radius: 2 }, type: 'content_shuffle' },
      id: 'filter-source',
      isEnabled: true,
      isLocked: false,
      name: 'Filter source',
      opacity: 0.6,
      source: {
        bitmap: { height: 10, imageName: 'filter-source.png', width: 10 },
        offset: { x: 7, y: -3 },
        type: 'paint' as const,
      },
      transform: { rotation: 12, scaleX: 2, scaleY: 3, x: 40, y: 50 },
    };
    return type === 'raster'
      ? { ...base, type: 'raster' }
      : {
          ...base,
          adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
          type: 'control',
          withTransparencyEffect: true,
        };
  };

  const createFilterFlowHarness = async (sourceType: 'raster' | 'control') => {
    const source = filterFlowLayer(sourceType);
    const document = { ...emptyDoc(), layers: [source], selectedLayerId: source.id };
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

  it('blocks Filter actions called directly after an external lock and still allows Cancel', async () => {
    const { document, engine } = await createFilterFlowHarness('raster');
    const makeDurable = vi.fn(() => Promise.resolve());
    const before = getCanvasOperations(engine).stores.filterSession.get();

    engine.tools.setInteractionLocked(true);

    expect(getCanvasOperations(engine).updateFilterOperation({ settings: { radius: 99 }, type: 'img_blur' })).toBe(
      'blocked'
    );
    expect(getCanvasOperations(engine).resetFilterOperation({ radius: 0 })).toBe('blocked');
    await expect(getCanvasOperations(engine).processFilterOperation()).resolves.toBe('blocked');
    await expect(getCanvasOperations(engine).commitFilterOperation('apply', makeDurable)).resolves.toBe('blocked');
    expect(makeDurable).not.toHaveBeenCalled();
    expect(engine.document.getDocument()).toEqual(document);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toEqual(before);

    getCanvasOperations(engine).cancelFilterOperation();
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    engine.tools.setInteractionLocked(false);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).updateFilterOperation({ settings: {}, type: 'img_blur' })).toBe('stale');
    expect(getCanvasOperations(engine).resetFilterOperation({})).toBe('stale');
    engine.lifecycle.dispose();
  });

  it('blocks Filter mutation when an external lock begins during durability', async () => {
    const { document, engine } = await createFilterFlowHarness('raster');
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(engine).commitFilterOperation('apply', () => durable.promise);
    await vi.waitFor(() => expect(getCanvasOperations(engine).stores.filterSession.get()?.status).toBe('committing'));

    engine.tools.setInteractionLocked(true);
    durable.resolve();

    await expect(pending).resolves.toBe('blocked');
    expect(engine.document.getDocument()).toEqual(document);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      preview: { imageName: 'filtered.png' },
      status: 'ready',
    });
    getCanvasOperations(engine).cancelFilterOperation();
    engine.lifecycle.dispose();
  });

  it('interrupts Filter processing on an external lock without losing the draft or operation', async () => {
    const source = filterFlowLayer('raster');
    const { projectId, store } = createReducerBackedStore({
      ...emptyDoc(),
      layers: [source],
      selectedLayerId: source.id,
    });
    const graph = createDeferred<{ height: number; imageName: string; origin: string; output: {}; width: number }>();
    const runGraph = vi.fn(() => graph.promise);
    const engine = createCanvasEngine({
      backend: filterBitmapBackend(),
      bitmapStore: createSpyBitmapStore(),
      filterDeps: {
        runGraph,
        uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'filter-input.png' })),
      },
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels(source.id)).status).toBe('ok');
    expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');
    expect(getCanvasOperations(engine).updateFilterOperation({ settings: { radius: 7 }, type: 'img_blur' })).toBe(
      'updated'
    );
    const pending = getCanvasOperations(engine).processFilterOperation();
    await vi.waitFor(() => expect(runGraph).toHaveBeenCalledOnce());

    engine.tools.setInteractionLocked(true);
    graph.resolve({ height: 10, imageName: 'late.png', origin: 'test', output: {}, width: 10 });

    await expect(pending).resolves.toBe('stale');
    expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
      draft: { settings: { radius: 7 }, type: 'img_blur' },
      preview: null,
      status: 'ready',
    });
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({ phase: 'ready', status: 'active' });
    getCanvasOperations(engine).cancelFilterOperation();
    engine.lifecycle.dispose();
  });

  it.each([
    {
      expectedRect: { height: 22, width: 22, x: 1, y: -9 },
      output: { height: 22, width: 22 },
      settings: { blur_type: 'gaussian', radius: 2 },
      target: 'apply',
      type: 'img_blur',
    },
    {
      expectedRect: { height: 14, width: 14, x: 5, y: -5 },
      output: { height: 14, width: 14 },
      settings: { blur_type: 'box', radius: 2 },
      target: 'raster',
      type: 'img_blur',
    },
    {
      expectedRect: { height: 30, width: 40, x: 7, y: -3 },
      output: { height: 30, width: 40 },
      settings: { autoScale: true, model: spandrelModel, scale: 4 },
      target: 'control',
      type: 'spandrel_filter',
    },
  ] as const)(
    'preserves $type/$target output geometry through preview, commit, and one-entry undo/redo',
    async ({ expectedRect, output, settings, target, type }) => {
      const source = filterFlowLayer('raster');
      const document = { ...emptyDoc(), layers: [source], selectedLayerId: source.id };
      const { projectId, store } = createReducerBackedStore(document);
      const engine = createCanvasEngine({
        backend: filterBitmapBackend(output.width, output.height),
        bitmapStore: createSpyBitmapStore(),
        filterDeps: {
          runGraph: vi.fn(() =>
            Promise.resolve({ ...output, imageName: 'filtered.png', origin: 'canvas', output: {} })
          ),
          uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'filter-input.png' })),
        },
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      expect((await engine.exports.exportLayerPixels(source.id)).status).toBe('ok');
      expect(getCanvasOperations(engine).startFilterOperation(source.id)).toBe('started');
      getCanvasOperations(engine).updateFilterOperation({ settings: structuredClone(settings), type });

      await getCanvasOperations(engine).processFilterOperation();
      expect(getCanvasOperations(engine).stores.filterSession.get()?.preview?.rect).toEqual(expectedRect);
      await getCanvasOperations(engine).commitFilterOperation(target, () => Promise.resolve());

      const committedId = target === 'apply' ? source.id : engine.document.getDocument()!.layers[0]!.id;
      const committed = engine.document.getDocument()!.layers.find((layer) => layer.id === committedId)!;
      expect(committed.transform).toEqual(source.transform);
      if (!('source' in committed)) {
        throw new Error('expected a raster or control filter result');
      }
      expect(committed.source).toMatchObject({
        bitmap: { height: output.height, width: output.width },
        offset: { x: expectedRect.x, y: expectedRect.y },
        type: 'paint',
      });
      const committedExport = await engine.exports.exportLayerPixels(committedId);
      expect(committedExport.status === 'ok' ? committedExport.rect : null).toEqual(expectedRect);
      expect(engine.stores.canUndo.get()).toBe(true);

      engine.history.undo();
      expect(engine.document.getDocument()).toEqual(document);
      expect(engine.stores.canUndo.get()).toBe(false);
      engine.history.redo();
      const redone = engine.document.getDocument()!.layers.find((layer) => layer.id === committedId)!;
      expect(redone).toEqual(committed);
      expect(engine.stores.canRedo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it.each(['raster', 'control'] as const)(
    'processes, retries durability, and applies %s with one origin-preserving history entry',
    async (sourceType) => {
      const { document, engine, source } = await createFilterFlowHarness(sourceType);
      const makeDurable = vi
        .fn<(imageName: string) => Promise<void>>()
        .mockRejectedValueOnce(new Error('promotion failed'))
        .mockResolvedValueOnce();

      await getCanvasOperations(engine).commitFilterOperation('apply', makeDurable);
      expect(getCanvasOperations(engine).stores.filterSession.get()).toMatchObject({
        error: 'promotion failed',
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
      expect(engine.document.getDocument()!.layers[0]).toMatchObject({
        filter: source.filter,
        opacity: source.opacity,
        blendMode: source.blendMode,
        source: { offset: { x: 7, y: -3 }, type: 'paint' },
        transform: source.transform,
        type: sourceType,
      });
      expect(engine.stores.canUndo.get()).toBe(true);
      engine.history.undo();
      expect(engine.document.getDocument()).toEqual(document);
      expect(engine.stores.canUndo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it.each(['raster', 'control'] as const)(
    'processes and saves as %s above the source with one history entry',
    async (target) => {
      const { engine, source } = await createFilterFlowHarness('control');

      await getCanvasOperations(engine).commitFilterOperation(target, () => Promise.resolve());

      expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
      expect(engine.document.getDocument()!.layers.map((layer) => layer.type)).toEqual([target, 'control']);
      expect(engine.document.getDocument()!.layers[0]).toMatchObject({
        blendMode: source.blendMode,
        opacity: source.opacity,
        source: { offset: { x: 7, y: -3 }, type: 'paint' },
        transform: source.transform,
      });
      expect(engine.stores.canUndo.get()).toBe(true);
      engine.history.undo();
      expect(engine.document.getDocument()!.layers).toEqual([source]);
      expect(engine.stores.canUndo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it('cancels an in-flight promotion when a replacement operation starts', async () => {
    const { engine, source } = await createFilterFlowHarness('raster');
    let resolvePromotion!: () => void;
    const promotion = new Promise<void>((resolve) => {
      resolvePromotion = resolve;
    });
    const pending = getCanvasOperations(engine).commitFilterOperation('apply', () => promotion);
    await vi.waitFor(() => expect(getCanvasOperations(engine).stores.filterSession.get()?.status).toBe('committing'));

    expect(getCanvasOperations(engine).startSelectObject(source.id)).toBe('started');
    resolvePromotion();
    await pending;

    expect(engine.document.getDocument()!.layers[0]).toEqual(source);
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object' },
      status: 'active',
    });
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('aborts an in-flight upload when a newer Process request wins', async () => {
    const source = filterFlowLayer('raster');
    const document = { ...emptyDoc(), layers: [source], selectedLayerId: source.id };
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
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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

  it.each(['filter', 'select-object'] as const)(
    'centrally blocks edits and undo during %s, then restores them after cancel',
    async (kind) => {
      const layer = guardableLayer('L');
      const document = { ...emptyDoc(), layers: [layer], selectedLayerId: layer.id };
      const { projectId, store } = createReducerBackedStore(document);
      const engine = createCanvasEngine({
        backend: createTestStubRasterBackend(),
        imageResolver: () => Promise.resolve(new Blob()),
        projectId,
        store,
      });
      expect((await engine.exports.exportLayerPixels('L')).status).toBe('ok');
      engine.layers.commitStructural(
        'Rename',
        { id: 'L', patch: { name: 'Renamed' }, type: 'updateCanvasLayer' },
        { id: 'L', patch: { name: 'L' }, type: 'updateCanvasLayer' }
      );
      const guard = engine.exports.captureLayerExportGuard('L');
      if (!guard) {
        throw new Error('expected a current operation guard');
      }
      const operation =
        kind === 'filter'
          ? getCanvasOperations(engine).controller.start({
              cleanupPreview: vi.fn(),
              guard,
              identity: { kind, layerId: 'L', projectId },
            })
          : null;
      if (kind === 'select-object') {
        expect(getCanvasOperations(engine).startSelectObject('L')).toBe('started');
      }

      expect(engine.stores.documentEditingLocked.get()).toBe(true);
      engine.history.undo();
      expect(
        engine.layers.applyStructuralPreview({ id: 'L', patch: { opacity: 0.2 }, type: 'updateCanvasLayer' })
      ).toBe(false);
      engine.layers.commitStructural(
        'Blocked rename',
        { id: 'L', patch: { name: 'Blocked' }, type: 'updateCanvasLayer' },
        { id: 'L', patch: { name: 'Renamed' }, type: 'updateCanvasLayer' }
      );
      engine.layers.nudgeSelectedLayer(5, 0);
      await expect(
        engine.layers.commitRasterFilterResult({
          guard,
          image: { height: 10, imageName: 'unauthorized.png', width: 10 },
          mode: 'replace',
          rect: { height: 10, width: 10, x: 0, y: 0 },
        })
      ).resolves.toEqual({ status: 'busy' });

      expect(engine.document.getDocument()!.layers[0]).toMatchObject({ name: 'Renamed', transform: { x: 0 } });
      expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
        identity: { kind },
        status: 'active',
      });

      if (kind === 'select-object') {
        getCanvasOperations(engine).cancelSelectObjectSession();
      } else {
        operation?.cancel();
      }
      expect(engine.stores.documentEditingLocked.get()).toBe(false);
      engine.history.undo();
      expect(engine.document.getDocument()!.layers[0]?.name).toBe('L');
      engine.lifecycle.dispose();
    }
  );

  it('does not clear history while document editing is locked', async () => {
    const layer = guardableLayer('L');
    const document = { ...emptyDoc(), layers: [layer] };
    const { projectId, store } = createReducerBackedStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels(layer.id)).status).toBe('ok');
    engine.layers.commitStructural(
      'Rename',
      { id: layer.id, patch: { name: 'Renamed' }, type: 'updateCanvasLayer' },
      { id: layer.id, patch: { name: layer.name }, type: 'updateCanvasLayer' }
    );
    const guard = engine.exports.captureLayerExportGuard(layer.id);
    if (!guard) {
      throw new Error('expected an operation guard');
    }
    const operation = getCanvasOperations(engine).controller.start({
      cleanupPreview: vi.fn(),
      guard,
      identity: { kind: 'filter', layerId: layer.id, projectId },
    });

    engine.history.clearHistory();

    expect(engine.stores.canUndo.get()).toBe(true);
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter' },
      status: 'active',
    });
    operation?.cancel();
    engine.history.clearHistory();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('invalidates the owned filter session when its guarded source changes', async () => {
    const layer = guardableLayer('L');
    const document = { ...emptyDoc(), layers: [layer] };
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

    setDocument({ ...document, layers: [{ ...layer, opacity: 0.5 }] });

    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(getCanvasOperations(engine).stores.filterSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('rejects a guarded filter preview when its source changes during decode', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const layer = guardableLayer('L');
    const document = { ...emptyDoc(), layers: [layer] };
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
      layers: [{ ...layer, source: { ...layer.source, fill: '#000' } } as CanvasLayerContract],
    });
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('stale');
    engine.lifecycle.dispose();
  });

  it('rejects a guarded filter preview when raster adjustments change during decode', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const layer = guardableLayer('L');
    const document = { ...emptyDoc(), layers: [layer] };
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
      layers: [{ ...layer, adjustments: { brightness: 0.2, contrast: 0, saturation: 0 } }],
    });
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('stale');
    engine.lifecycle.dispose();
  });

  it('rejects a guarded filter preview when the paint cache version changes during decode', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
    const { store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      bitmapStore: createSpyBitmapStore(),
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
    await engine.diagnostics.clearCaches();
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('stale');
    engine.lifecycle.dispose();
  });

  it('returns missing when the guarded preview layer is deleted during decode', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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
    setDocument({ ...document, layers: [] });
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('missing');
    engine.lifecycle.dispose();
  });

  it('rejects a guarded filter preview after document replacement reuses the layer id', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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
    setDocument({ ...document, height: 200, width: 200 }, 1);
    bitmaps.resolveBitmap(0);

    await expect(pending).resolves.toBe('stale');
    engine.lifecycle.dispose();
  });

  it('lets only the newest guarded filter preview publish', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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
    const backend = createRecordingRasterBackend();
    const sourceNeedsBitmap =
      layer.type === 'raster' &&
      (layer.source.type === 'image' || (layer.source.type === 'paint' && layer.source.bitmap !== null));
    let bitmapCall = 0;
    backend.createImageBitmap = vi.fn(() =>
      Promise.resolve(recordingBitmap(sourceNeedsBitmap && bitmapCall++ === 0 ? 'layer-source' : 'guarded-preview'))
    );
    const document = { ...emptyDoc(), layers: [layer], selectedLayerId: layer.id };
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

  it('keeps a published guarded preview across an active-project away/back transition', async () => {
    const harness = await createPublishedGuardedPreview();

    harness.setActiveProjectId('p2');
    harness.setActiveProjectId('p1');
    harness.screen.callLog.length = 0;
    harness.engine.stores.checkerboard.set(!harness.engine.stores.checkerboard.get());
    harness.raf.flush();

    expect(screenDrawsBitmap(harness.screen, harness.backend, 'guarded-preview')).toBe(true);
    harness.engine.lifecycle.dispose();
  });

  it('publishes an in-flight guarded preview across an active-project away/back transition', async () => {
    const bitmaps = createDeferredBitmapBackend();
    const document = { ...emptyDoc(), layers: [guardableLayer('L')] };
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
      layers: [{ ...harness.layer, source: { ...harness.layer.source, fill: '#000' } } as CanvasLayerContract],
    });
    harness.raf.flush();

    expect(screenDrawsBitmap(harness.screen, harness.backend, 'guarded-preview')).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('clears an already-published guarded preview when raster adjustments change', async () => {
    const harness = await createPublishedGuardedPreview();
    harness.setDocument({
      ...harness.document,
      layers: [{ ...harness.layer, adjustments: { brightness: 0.3, contrast: 0, saturation: 0 } }],
    });
    harness.raf.flush();

    expect(screenDrawsBitmap(harness.screen, harness.backend, 'guarded-preview')).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('clears an already-published guarded preview when live paint advances the cache version', async () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
    const paint: CanvasRasterLayerContractV2 = {
      ...guardableLayer('L'),
      source: { bitmap: { height: 10, imageName: 'paint-source', width: 10 }, type: 'paint' },
      type: 'raster',
    };
    const harness = await createPublishedGuardedPreview(paint);

    harness.engine.selection.selectAll();
    harness.engine.selection.fillSelection();
    harness.raf.flush();

    expect(screenDrawsBitmap(harness.screen, harness.backend, 'guarded-preview')).toBe(false);
    harness.engine.lifecycle.dispose();
  });

  it('clears an already-published guarded preview when its cache is invalidated', async () => {
    const harness = await createPublishedGuardedPreview();

    await harness.engine.diagnostics.clearCaches();
    harness.raf.flush();

    expect(screenDrawsBitmap(harness.screen, harness.backend, 'guarded-preview')).toBe(false);
    harness.engine.lifecycle.dispose();
  });
});

// ---- C1: prop/transform edits must not wipe an unflushed paint layer -----
//
// A `bitmap: null` paint layer's strokes live ONLY in its raster cache until a
// debounced upload persists them. The paint rasterizer clears the surface for a
// null bitmap, so re-rasterizing such a layer WIPES the strokes. The engine must
// therefore invalidate a layer's cache only when its SOURCE reference changed —
// never for a prop/transform-only edit (opacity/blend/lock/rename/nudge), which
// the compositor already applies at draw time.

/** Full-surface clears (`clearRect(0,0,w,h)`) recorded on a stub surface — the wipe signature. */
const fullClearCount = (surface: StubRasterSurface): number =>
  surface.callLog.filter((entry) => entry.op === 'clearRect' && entry.args[0] === 0 && entry.args[1] === 0).length;

describe('document mirror wiring: prop vs source change (paint-pixel survival)', () => {
  /**
   * Rasterizes the pre-existing `paint1` layer to completion (as real frames
   * would before the user paints — the existing-layer paint path does NOT mark
   * the cache non-stale itself), then draws one stroke into that non-stale cache.
   * This reproduces the C1 state: real strokes living only in a cache that a
   * spurious re-rasterize (of the `bitmap: null` source) would clear to blank.
   */
  const paintOneStroke = async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // Initial rasterize of the (bitmap: null) paint layer → one full clear, then
    // the cache settles non-stale (the `.then` fires on a microtask).
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    // The painted layer cache is the only engine-backend surface that ever gets
    // `getImageData` (the before/after stroke capture); the scratch stroke
    // surface never does.
    const paintCache = surfaces.find((surface) => surface.callLog.some((entry) => entry.op === 'getImageData'));
    return { bitmapStore, engine, paintCache: paintCache!, raf, resolver, setDocument };
  };

  const paintOneStrokeWithReducer = async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const { projectId, store } = createReducerBackedStore(paintDoc());
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
    engine.tools.setTool('brush');
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    return engine;
  };

  it('accepts a persisted bitmap from reducer state when an earlier subscriber throws before the mirror refreshes', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const reducer = createReducerBackedStore(paintDoc());
    let throwAfterBitmapCommit = false;
    const unsubscribeThrower = reducer.store.subscribe(() => {
      if (throwAfterBitmapCommit) {
        throw new Error('earlier subscriber failed after reducer commit');
      }
    });
    const reportError = vi.fn();
    const uploadImage = vi
      .spyOn(canvasApplicationPort, 'uploadImage')
      .mockResolvedValue({ height: 40, imageName: 'persisted-before-mirror-refresh.png', width: 60 });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      reportError,
      store: reducer.store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    engine.tools.setTool('brush');
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const strokes: StrokeCommittedEvent[] = [];
    const unsubscribeStroke = engine.tools.onStrokeCommitted((stroke) => strokes.push(stroke));
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    expect(strokes).toHaveLength(1);
    expect(strokes[0]?.layerId).toBe('paint1');
    throwAfterBitmapCommit = true;

    await expect(engine.lifecycle.flushPendingUploads()).resolves.toBeUndefined();
    await expect(engine.lifecycle.flushPendingUploads()).resolves.toBeUndefined();

    const authoritativeLayer = reducer.store
      .getState()
      .projects.find((project) => project.id === reducer.projectId)
      ?.canvas.document.layers.find((layer) => layer.id === 'paint1');
    expect(uploadImage).toHaveBeenCalledOnce();
    const bitmapUpdates = reducer.dispatch.mock.calls
      .map(([action]) => action)
      .filter((action) => action.type === 'updateCanvasLayerSource' && action.id === 'paint1');
    expect(bitmapUpdates).toHaveLength(1);
    expect(authoritativeLayer).toMatchObject({
      source: { bitmap: { imageName: 'persisted-before-mirror-refresh.png' }, type: 'paint' },
    });
    expect(reportError).not.toHaveBeenCalled();

    unsubscribeStroke();
    unsubscribeThrower();
    engine.lifecycle.dispose();
    uploadImage.mockRestore();
  });

  it('keeps an unflushed paint layer’s pixels on a transform/opacity-only change (no re-rasterize)', async () => {
    const { engine, paintCache, raf, resolver, setDocument } = await paintOneStroke();

    // Baseline: the stroke composited into the cache (a drawImage). Exactly one
    // full clear so far — the initial rasterize; the stroke never clears.
    expect(paintCache.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    const clearsBefore = fullClearCount(paintCache);

    // A prop-only edit that PRESERVES the source reference (exactly as the reducer
    // does — it spreads `...layer`): opacity + a transform nudge.
    const doc = engine.document.getDocument()!;
    const layer = doc.layers[0]!;
    setDocument({
      ...doc,
      layers: [{ ...layer, opacity: 0.5, transform: { ...layer.transform, x: 12, y: -4 } }],
    });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    // The cache was NOT re-rasterized: no new full clear, so the painted pixels
    // survive. (A `bitmap: null` re-rasterize would clear the surface to blank.)
    expect(fullClearCount(paintCache)).toBe(clearsBefore);
    // No resolve/rasterize was even attempted for the paint layer.
    expect(resolver).not.toHaveBeenCalled();
    expect(engine.document.getDocument()!.layers[0]!.opacity).toBe(0.5);

    engine.lifecycle.dispose();
  });

  it('exports an unflushed paint layer from its ready live cache when the contract bitmap is still null', async () => {
    const { engine, paintCache } = await paintOneStroke();

    const result = await engine.exports.exportLayerPixels('paint1');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.surface).toBe(paintCache);
      expect(result.rect.width).toBeGreaterThan(0);
      expect(result.rect.height).toBeGreaterThan(0);
    }
    engine.lifecycle.dispose();
  });

  it('preserves live pixels through contract copy and conversion history', async () => {
    const engine = await paintOneStrokeWithReducer();
    const raster = engine.document.getDocument()!.layers[0]!;
    if (raster.type !== 'raster') {
      throw new Error('expected raster source');
    }
    const control: CanvasLayerContract = {
      ...raster,
      adapter: {
        beginEndStepPct: [0, 0.75],
        controlMode: 'balanced',
        kind: 'controlnet',
        model: null,
        weight: 1,
      },
      type: 'control',
      withTransparencyEffect: true,
    };
    const copy = { ...control, id: 'control-copy', name: 'Control copy' };

    expect(engine.layers.commitLayerCopy('Copy layer', raster.id, copy, 0)).toBe(true);
    expect((await engine.exports.exportLayerPixels(copy.id)).status).toBe('ok');
    engine.history.undo();
    expect(engine.document.getDocument()!.layers.some((layer) => layer.id === copy.id)).toBe(false);
    engine.history.redo();
    expect((await engine.exports.exportLayerPixels(copy.id)).status).toBe('ok');

    expect(engine.layers.commitLayerConversion('Convert layer', raster, control)).toBe(true);
    expect(engine.document.getDocument()!.layers.find((layer) => layer.id === raster.id)?.type).toBe('control');
    expect((await engine.exports.exportLayerPixels(raster.id)).status).toBe('ok');
    engine.history.undo();
    expect(engine.document.getDocument()!.layers.find((layer) => layer.id === raster.id)?.type).toBe('raster');
    expect((await engine.exports.exportLayerPixels(raster.id)).status).toBe('ok');
    engine.lifecycle.dispose();
  });

  it('commitLayerConversion requires the caller to hold the immutable live layer object', async () => {
    const { dispatch, projectId, store } = createReducerBackedStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exports.exportLayerPixels('a')).status).toBe('ok');
    const live = engine.document.getDocument()!.layers[0]!;
    if (live.type !== 'raster') {
      throw new Error('expected raster layer');
    }
    const converted: CanvasLayerContract = {
      ...live,
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      type: 'control',
      withTransparencyEffect: false,
    };
    dispatch.mockClear();

    expect(engine.layers.commitLayerConversion('Convert', structuredClone(live), converted)).toBe(false);
    expect(dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'convertCanvasLayer' }));
    expect(engine.document.getDocument()!.layers[0]!.type).toBe('raster');
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
    const expectedUnlocked = engine.document.getDocument()!.layers[0]!;
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
    expect(engine.document.getDocument()!.layers[0]).toMatchObject({ isLocked: true, type: 'raster' });
    engine.lifecycle.dispose();
  });

  it('flushing after a prop-only change persists the painted (non-blank) surface', async () => {
    const { bitmapStore, engine, paintCache, raf, setDocument } = await paintOneStroke();
    const clearsBefore = fullClearCount(paintCache);

    const doc = engine.document.getDocument()!;
    const layer = doc.layers[0]!;
    setDocument({ ...doc, layers: [{ ...layer, opacity: 0.25 }] });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    await engine.lifecycle.flushPendingUploads();

    // The flush barrier ran, and it operated on a cache that was never wiped: the
    // surface still carries the stroke (drawImage) with no re-rasterize clear, so
    // the bitmap store (which encodes this exact surface) persists real pixels.
    expect(bitmapStore.flushPendingUploads).toHaveBeenCalled();
    expect(fullClearCount(paintCache)).toBe(clearsBefore);
    expect(paintCache.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('re-rasterizes when the paint layer’s source genuinely changes (swap to a persisted bitmap)', async () => {
    const { engine, raf, resolver, setDocument } = await paintOneStroke();
    expect(resolver).not.toHaveBeenCalled();

    // A genuine source swap (undo/import → a NEW paint source object with a
    // persisted bitmap). isSelfEcho is false in the spy store, so this must
    // invalidate and re-rasterize — which decodes the persisted image.
    const doc = engine.document.getDocument()!;
    const layer = doc.layers[0] as CanvasRasterLayerContractV2;
    setDocument({
      ...doc,
      layers: [
        {
          ...layer,
          source: { bitmap: { contentHash: 'h', height: 100, imageName: 'persisted', width: 100 }, type: 'paint' },
        },
      ],
    });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    // The cache was invalidated and re-rasterized from the new persisted source.
    expect(resolver).toHaveBeenCalledWith('persisted', expect.any(AbortSignal));

    engine.lifecycle.dispose();
  });

  it('leaves image layers alone on a prop change but re-rasterizes on a source swap', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { setDocument, store } = createReactiveStore(makeDoc()); // one image layer 'a'
    const resolver = vi.fn((_imageName: string) => Promise.resolve(new Blob()));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    engine.surface.attach(createFakeCanvas().element, createFakeCanvas().element);

    // Initial rasterize of image 'a'.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'a', expect.any(AbortSignal));

    // Prop-only edit (opacity), source reference preserved: no re-rasterize.
    const doc = engine.document.getDocument()!;
    setDocument({ ...doc, layers: [{ ...doc.layers[0]!, opacity: 0.3 }] });
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);

    // Source swap (new image name → new source object): re-rasterizes the new source.
    const doc2 = engine.document.getDocument()!;
    const imgLayer = doc2.layers[0] as CanvasRasterLayerContractV2;
    setDocument({
      ...doc2,
      layers: [{ ...imgLayer, source: { image: { height: 10, imageName: 'a-v2', width: 10 }, type: 'image' } }],
    });
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'a-v2', expect.any(AbortSignal));

    engine.lifecycle.dispose();
  });
});

describe('hasExportableLayerContent', () => {
  const sourceLayer = (id: string, source: CanvasLayerSourceContract, isEnabled = true): CanvasLayerContract => ({
    blendMode: 'normal',
    id,
    isEnabled,
    isLocked: false,
    name: id,
    opacity: 1,
    source,
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  });

  const createContentEngine = (layers: CanvasLayerContract[]) => {
    const { store } = createFakeStore({ ...makeDoc(), layers });
    return createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
  };

  const createLiveUnpersistedLayer = async (layer: CanvasLayerContract) => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

    const { setDocument, store } = createReactiveStore({
      ...makeDoc(),
      layers: [layer],
      selectedLayerId: layer.id,
    });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);

    // Settle the initial empty paint/mask rasterization before drawing. A stroke
    // then grows that current cache without updating the persisted contract.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));

    return { engine, setDocument };
  };

  it('returns true for image, persisted paint, and every supported parametric source', () => {
    const sources: { id: string; source: CanvasLayerSourceContract }[] = [
      {
        id: 'image',
        source: { image: { height: 12, imageName: 'image', width: 14 }, type: 'image' },
      },
      {
        id: 'persisted-paint',
        source: {
          bitmap: { height: 15, imageName: 'paint', width: 16 },
          offset: { x: -3, y: 4 },
          type: 'paint',
        },
      },
      {
        id: 'rect',
        source: { fill: '#000', height: 18, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 20 },
      },
      {
        id: 'ellipse',
        source: {
          fill: '#000',
          height: 18,
          kind: 'ellipse',
          stroke: null,
          strokeWidth: 0,
          type: 'shape',
          width: 20,
        },
      },
      {
        id: 'gradient',
        source: {
          angle: 45,
          height: 22,
          kind: 'linear',
          stops: [{ color: '#000', offset: 0 }],
          type: 'gradient',
          width: 24,
        },
      },
      {
        id: 'text',
        source: {
          align: 'left',
          color: '#000',
          content: 'Export me',
          fontFamily: 'Inter',
          fontSize: 20,
          fontWeight: 400,
          lineHeight: 1.2,
          type: 'text',
        },
      },
    ];
    // Disabled layers remain exportable because Save/Clipboard may explicitly
    // export hidden content.
    const engine = createContentEngine(sources.map(({ id, source }) => sourceLayer(id, source, false)));

    for (const { id } of sources) {
      expect(engine.exports.hasExportableLayerContent(id), id).toBe(true);
    }
    engine.lifecycle.dispose();
  });

  it('returns false for empty paint, unsupported polygon, and missing ids', () => {
    const polygon: CanvasLayerSourceContract = {
      fill: '#000',
      height: 20,
      kind: 'polygon',
      points: [
        { x: 0, y: 0 },
        { x: 20, y: 0 },
        { x: 10, y: 20 },
      ],
      stroke: null,
      strokeWidth: 0,
      type: 'shape',
      width: 20,
    };
    const engine = createContentEngine([
      sourceLayer('empty-paint', { bitmap: null, type: 'paint' }),
      sourceLayer('polygon', polygon),
    ]);

    expect(engine.exports.hasExportableLayerContent('empty-paint')).toBe(false);
    expect(engine.exports.hasExportableLayerContent('polygon')).toBe(false);
    expect(engine.exports.hasExportableLayerContent('missing')).toBe(false);
    engine.lifecycle.dispose();
  });

  it('returns true for a persisted disabled mask and false for an empty mask', () => {
    const persistedMask: CanvasInpaintMaskLayerContract = {
      ...maskLayer('persisted-mask'),
      isEnabled: false,
      mask: {
        ...maskLayer('persisted-mask').mask,
        bitmap: { height: 17, imageName: 'persisted-mask', width: 19 },
        offset: { x: 2, y: 3 },
      },
    };
    const engine = createContentEngine([persistedMask, maskLayer('empty-mask')]);

    expect(engine.exports.hasExportableLayerContent('persisted-mask')).toBe(true);
    expect(engine.exports.hasExportableLayerContent('empty-mask')).toBe(false);
    engine.lifecycle.dispose();
  });

  it('returns true for current live unpersisted paint pixels', async () => {
    const { engine } = await createLiveUnpersistedLayer(sourceLayer('live-paint', { bitmap: null, type: 'paint' }));

    expect(engine.exports.hasExportableLayerContent('live-paint')).toBe(true);
    engine.lifecycle.dispose();
  });

  it('captures current unflushed pixels and bounds for a new bitmap-less paint layer', async () => {
    const { engine } = await createLiveUnpersistedLayer(
      sourceLayer('live-snapshot-paint', { bitmap: null, type: 'paint' })
    );
    const documentSnapshot = engine.document.captureSnapshot();

    const result = await engine.exports.captureRasterSnapshot(documentSnapshot!, ['live-snapshot-paint']);

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      throw new Error('Expected live paint snapshot');
    }
    const detached = result.snapshot.layerSurfaces.get('live-snapshot-paint');
    expect(detached?.rect.width).toBeGreaterThan(0);
    expect(detached?.rect.height).toBeGreaterThan(0);
    result.snapshot.release();
    engine.lifecycle.dispose();
  });

  it('returns false for a stale non-empty live cache with no persisted pixels', async () => {
    const { engine, setDocument } = await createLiveUnpersistedLayer(
      sourceLayer('stale-paint', { bitmap: null, type: 'paint' })
    );
    expect(engine.exports.hasExportableLayerContent('stale-paint')).toBe(true);

    const doc = engine.document.getDocument()!;
    const layer = doc.layers[0];
    if (!layer || layer.type !== 'raster') {
      throw new Error('expected raster paint layer');
    }
    // A genuine source-reference change invalidates the still-non-empty live
    // cache synchronously. Do not run the scheduled frame that would rebuild it.
    setDocument({
      ...doc,
      layers: [{ ...layer, source: { bitmap: null, type: 'paint' } }],
    });
    const { target } = createThumbnailTarget();
    expect(engine.previews.drawLayerThumbnail('stale-paint', target, 96)).toBe(true);
    expect(engine.exports.hasExportableLayerContent('stale-paint')).toBe(false);
    engine.lifecycle.dispose();
  });

  it('returns true for current live unpersisted mask pixels', async () => {
    const { engine } = await createLiveUnpersistedLayer(maskLayer('live-mask'));

    expect(engine.exports.hasExportableLayerContent('live-mask')).toBe(true);
    engine.lifecycle.dispose();
  });
});

// ---- I4: structural edits are no-ops during an active pointer gesture -----

describe('gesture guard: nudge / commitStructural mid-stroke', () => {
  const startOpenStroke = () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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
    // Open a stroke (pointer down + move, NO up) so the gesture stays active.
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(30, 30));
    return { dispatch, engine, overlay };
  };

  it('no-ops nudgeSelectedLayer while a stroke gesture is open', () => {
    const { dispatch, engine, overlay } = startOpenStroke();
    const before = dispatch.mock.calls.length;

    engine.layers.nudgeSelectedLayer(1, 0);
    // No structural transform dispatch, no history entry.
    expect(dispatch.mock.calls.filter((call) => call[0].type === 'updateCanvasLayer')).toHaveLength(0);
    expect(engine.stores.canUndo.get()).toBe(false);

    // After the gesture ends, the nudge lands.
    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    engine.layers.nudgeSelectedLayer(1, 0);
    expect(dispatch.mock.calls.length).toBeGreaterThan(before);
    expect(dispatch.mock.calls.some((call) => call[0].type === 'updateCanvasLayer')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('no-ops commitStructural while a stroke gesture is open, then commits after it ends', () => {
    const { dispatch, engine, overlay } = startOpenStroke();
    const forward: EngineTestAction = { id: 'x', type: 'setCanvasSelectedLayer' };
    const inverse: EngineTestAction = { id: null, type: 'setCanvasSelectedLayer' };

    expect(engine.layers.canCommitStructural()).toBe(false);
    expect(engine.layers.commitStructural('Select', forward, inverse)).toBe(false);
    // Nothing dispatched, nothing recorded on history mid-gesture.
    expect(dispatch.mock.calls.some((call) => call[0] === forward)).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);

    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    expect(engine.layers.canCommitStructural()).toBe(true);
    expect(engine.layers.commitStructural('Select', forward, inverse)).toBe(true);
    expect(dispatch.mock.calls.some((call) => call[0] === forward)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.lifecycle.dispose();
  });

  it('no-ops mergeLayerDown while a stroke gesture is open, then merges after it ends', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

    const { store } = createReactiveStore(twoPaintDoc());
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
    // Build both layer caches so a merge could otherwise succeed; await the async
    // decode so both are READY (merge refuses stale/in-flight caches — finding 20).
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    engine.tools.setTool('brush');

    // Open a stroke into the selected 'upper' paint layer (no pointer-up).
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(30, 30));

    // Mid-gesture merge is refused (matches commitStructural/nudge): not undoable.
    expect(engine.layers.mergeLayerDown('upper')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );

    // After the gesture ends, the merge lands.
    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    expect(engine.layers.mergeLayerDown('upper')).toBe(true);
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'mergeCanvasLayersDown')).toBe(
      true
    );

    engine.lifecycle.dispose();
  });
});

// ---- brush cursor ring: resizes on a size change with no pointer event ----

describe('brush cursor ring: live size updates', () => {
  it('invalidates the overlay when the brush size changes without a pointer move', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // A bare hover move (no button) sets the cursor ring at a known position.
    overlay.fire('pointermove', pointerAt(30, 30, { buttons: 0 }));
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    // The `[`/`]` path: a size step with NO pointer event must schedule a frame
    // (the ring redraws at its last position with the new radius).
    engine.tools.stepBrushSize(1);
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    // The options-bar slider path (a direct store write) likewise invalidates.
    engine.stores.brushOptions.set({ ...engine.stores.brushOptions.get(), size: 123 });
    expect(raf.pendingCount()).toBeGreaterThan(0);

    engine.lifecycle.dispose();
  });

  it('does not invalidate for a size change while no ring is shown (pointer off-canvas)', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    // No hover move happened, so there is no ring to resize: no frame scheduled.
    engine.tools.stepBrushSize(1);
    expect(raf.pendingCount()).toBe(0);

    engine.lifecycle.dispose();
  });
});

// ---- doc-replace mid-gesture: cancels the active tool gesture (I-follow-up) ----

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

    // Start a bbox move-drag (press inside the 100x100 frame, then drag).
    overlay.fire('pointerdown', pointerAt(40, 40));
    overlay.fire('pointermove', pointerAt(60, 60));
    expect(engine.stores.bboxPreview.get()).not.toBeNull();

    // A wholesale document swap mid-drag (dims change → onDocumentReplaced).
    setDocument({ ...paintDoc(), height: 200, width: 200 });

    // The gesture was cancelled: the transient preview is cleared.
    expect(engine.stores.bboxPreview.get()).toBeNull();

    // The eventual pointer-up must NOT commit a bbox against the replaced document.
    overlay.fire('pointerup', pointerAt(60, 60, { buttons: 0 }));
    expect(dispatch.mock.calls.some((call) => (call[0] as EngineTestAction).type === 'setCanvasBbox')).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('cancels a move-tool drag on a document swap, clearing the transform override', () => {
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
    engine.tools.setTool('move');

    // Drag the doc-sized paint layer.
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 40));

    const updatesBefore = dispatch.mock.calls.filter(
      (call) => (call[0] as EngineTestAction).type === 'updateCanvasLayer'
    ).length;

    setDocument({ ...paintDoc(), height: 200, width: 200 });

    // Pointer-up after the swap commits no transform update (the gesture was cancelled).
    overlay.fire('pointerup', pointerAt(50, 40, { buttons: 0 }));
    const updatesAfter = dispatch.mock.calls.filter(
      (call) => (call[0] as EngineTestAction).type === 'updateCanvasLayer'
    ).length;
    expect(updatesAfter).toBe(updatesBefore);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.lifecycle.dispose();
  });
});

// ---- Zoom-cost regressions: what must NOT scale with zoom ----------------
//
// The reported lag ("laggier the closer you zoom in") traced to the render loop
// recompositing the whole document on EVERY invalidation — including overlay-only
// hover frames, whose cost is otherwise constant. A full composite up-scales each
// doc-sized layer surface to fill the screen, so its fill-rate grows with zoom.
// These lock the two fixes: overlay-only frames skip the composite, and composites
// disable image smoothing when zoomed in (crisp + no bilinear up-scale per frame).

describe('zoom-cost: overlay-only frames skip the document composite', () => {
  /** A fake canvas that can BOTH fire pointer events and expose its recording surface. */
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

  /** Composite draws land on the screen surface as clear/fill/blit ops. */
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

    // Drain the attach `{ all }` frame and the paint layer's async rasterize
    // follow-up so no composite-triggering work is left pending.
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    // Isolate the hover frame.
    screen.surface.callLog.length = 0;
    overlay.surface.callLog.length = 0;

    // A hover move (no button pressed) resizes the cursor ring → `{ overlay: true }`.
    overlay.fire('pointermove', pointerAt(40, 40, { buttons: 0 }));
    raf.flush();

    // The screen composite did NOT run: zero clears/fills/blits landed on it. This
    // is the win — hover cost is now independent of zoom and document size.
    expect(compositeOps(screen.surface)).toHaveLength(0);
    // The overlay WAS redrawn: cleared, and the cursor ring arc drawn.
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

    // Zoom in to 20× → a view invalidation → recomposite with smoothing off.
    screen.surface.callLog.length = 0;
    engine.viewport.getViewport().zoomAtPoint(20, { x: 0, y: 0 });
    raf.flush();
    expect(findSet(screen.surface, 'imageSmoothingEnabled')).toContain(false);

    // Zoom out below 1× → recomposite with smoothing on (clean down-scale).
    screen.surface.callLog.length = 0;
    engine.viewport.getViewport().zoomAtPoint(0.5, { x: 0, y: 0 });
    raf.flush();
    expect(findSet(screen.surface, 'imageSmoothingEnabled')).toContain(true);

    engine.lifecycle.dispose();
  });

  it('builds the checkerboard pattern tile once across many composited frames', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    // Count surfaces allocated at the checkerboard tile size (CHECKERBOARD_SQUARE_PX * 2 = 16).
    const base = createTestStubRasterBackend();
    const tileSurfaceSizes: string[] = [];
    const backend: StubRasterBackend = {
      ...base,
      createSurface: (w: number, h: number): StubRasterSurface => {
        if (w === 16 && h === 16) {
          tileSurfaceSizes.push(`${w}x${h}`);
        }
        return base.createSurface(w, h);
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
    // Checkerboard is on by default; the transparent paintDoc fills with it.
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.surface.attach(screen.element, overlay.element);

    // Force several composited frames (each recomposites and re-`createPattern`s).
    for (let i = 0; i < 5; i++) {
      engine.viewport.getViewport().zoomAtPoint(1 + i, { x: 0, y: 0 });
      raf.flush();
      await flushMicrotasks();
    }

    // The tile surface was built exactly once and reused (no per-frame rebuild).
    expect(tileSurfaceSizes).toEqual(['16x16']);

    engine.lifecycle.dispose();
  });
});

// ---- checker colors: theme-token feed → tile rebuild + recomposite ---------

const fillStyleSets = (surface: StubRasterSurface): unknown[] =>
  surface.callLog.filter((e) => e.op === 'set' && e.args[0] === 'fillStyle').map((e) => e.args[1]);

describe('checker colors: fed-token tile rebuild', () => {
  it('builds the fallback-colored tile when never fed, then rebuilds with fed colors and recomposites', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    // Capture every 16x16 checker tile surface (CHECKERBOARD_SQUARE_PX * 2).
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

    // First composite built the tile once, using the React-free fallback colors.
    expect(tiles).toHaveLength(1);
    expect(fillStyleSets(tiles[0]!)).toEqual([DEFAULT_CHECKER_COLORS.a, DEFAULT_CHECKER_COLORS.b]);

    // Feed new (resolved-token) checker colors: the cached tile is dropped and
    // rebuilt with the new colors on the next composite, which is forced to run.
    engine.stores.checkerColors.set({ a: '#010101', b: '#020202' });
    raf.flush();
    await flushMicrotasks();

    expect(tiles).toHaveLength(2);
    expect(fillStyleSets(tiles[1]!)).toEqual(['#010101', '#020202']);

    engine.lifecycle.dispose();
  });

  it('does not rebuild the tile when fed colors are unchanged (equality-gated store)', async () => {
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

    // Re-feeding the SAME colors is a no-op (the store's equality check drops it),
    // so no invalidation and no tile rebuild.
    engine.stores.checkerColors.set({ ...DEFAULT_CHECKER_COLORS });
    raf.flush();
    await flushMicrotasks();
    expect(tiles).toHaveLength(1);

    engine.lifecycle.dispose();
  });
});

// ---- fitToView: content ∪ bbox (document rect retired as world bounds) ------

const noLayerDoc = (overrides: Partial<CanvasDocumentContractV2> = {}): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 1000,
  layers: [],
  selectedLayerId: null,
  version: 2,
  width: 1000,
  ...overrides,
});

describe('fitToView: content ∪ bbox', () => {
  it('fits the bbox (not the larger doc rect) on an empty canvas', () => {
    const { store } = createReactiveStore(noLayerDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.surface.resize(400, 400, 1);

    engine.viewport.fitToView();

    // avail = 400 - 48*2 = 304; fitting the 100px bbox → 3.04. Fitting the 1000px
    // doc rect would have been ~0.304 — the doc rect is no longer the fit target.
    expect(engine.viewport.getViewport().getZoom()).toBeCloseTo(3.04, 2);
    engine.lifecycle.dispose();
  });

  it('unions a renderable layer that lies beyond the bbox into the fit', () => {
    // A raster layer 100x100 translated to (900,900): content extends to 1000, far
    // past the 100px bbox. Fitting content ∪ bbox spans 0..1000 → zoom ~0.304.
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
    const { store } = createReactiveStore(noLayerDoc({ layers: [layer] }));
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
});

// ---- transform session: param commit (image) + pixel bake (paint) ----------

describe('transform session', () => {
  it('image layer Apply commits one structural transform with the exact inverse', () => {
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
    // Exactly one structural dispatch (the forward transform); no pixel work.
    expect(layerDispatches).toHaveLength(1);
    expect(layerDispatches[0]).toEqual({
      id: 'a',
      patch: { transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 } },
      type: 'updateCanvasLayer',
    });
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.stores.canUndo.get()).toBe(true);

    // Undo dispatches the exact inverse (the captured start transform).
    dispatch.mockClear();
    engine.history.undo();
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

  it('parametric (text) layer Apply commits ONE param transform, stays type text, no bake; undo restores', () => {
    // Regression: parametric layers (shape/gradient/text) could not be transformed
    // — `applyTransform` only handled image sources. Phase 5 "param for parametric":
    // the transform commits as a param edit and the source stays editable-forever.
    const textLayerDoc: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [
        {
          blendMode: 'normal',
          id: 't',
          isEnabled: true,
          isLocked: false,
          name: 'Text',
          opacity: 1,
          source: {
            align: 'left',
            color: '#000000',
            content: 'hi',
            fontFamily: 'sans',
            fontSize: 20,
            fontWeight: 400,
            lineHeight: 1.2,
            type: 'text',
          },
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
          type: 'raster',
        },
      ],
      selectedLayerId: 't',
      version: 2,
      width: 100,
    };
    const { store } = createReactiveStore(textLayerDoc);
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    engine.tools.setTool('transform');
    // A session opened on the (now hit-testable) text layer.
    expect(engine.stores.transformSession.get()).not.toBeNull();
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 });
    dispatch.mockClear();
    engine.layers.applyTransform();

    const actions = dispatch.mock.calls.map((call) => call[0] as EngineTestAction);
    const updates = actions.filter((a) => a.type === 'updateCanvasLayer');
    // ONE param transform; source untouched (stays text) — no convert, no bake.
    expect(updates).toHaveLength(1);
    expect(updates[0]).toEqual({
      id: 't',
      patch: { transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 } },
      type: 'updateCanvasLayer',
    });
    expect(actions.some((a) => a.type === 'convertCanvasLayer' || a.type === 'updateCanvasLayerSource')).toBe(false);
    expect(engine.stores.transformSession.get()).toBeNull();

    dispatch.mockClear();
    engine.history.undo();
    const undo = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((a) => a.type === 'updateCanvasLayer');
    expect(undo).toEqual({
      id: 't',
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

  it('an unchanged transform Apply cancels with no dispatch', () => {
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
    dispatch.mockClear();
    engine.layers.applyTransform();

    expect(dispatch.mock.calls.filter((c) => (c[0] as EngineTestAction).type === 'updateCanvasLayer')).toHaveLength(0);
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.lifecycle.dispose();
  });

  it('Cancel drops the session with no dispatch', () => {
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
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 3, scaleY: 3, x: 0, y: 0 });
    dispatch.mockClear();
    engine.layers.cancelTransform();

    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.transformSession.get()).toBeNull();

    engine.lifecycle.dispose();
  });

  it('paint layer Apply bakes pixels through the matrix, resets the transform, and composes one undo entry', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    // A paint layer that already carries a non-identity transform, so undo restores
    // a transform distinct from the post-bake identity. Content-sized: it needs a
    // persisted bitmap so its cache is non-empty (an empty paint layer is not
    // transformable).
    const movedPaint = paintDoc();
    const movedLayer = movedPaint.layers[0] as CanvasRasterLayerContractV2;
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
    raf.flush(); // build the paint layer cache

    const surfacesBeforeApply = surfaces.length;
    const dirtyBefore = bitmapStore.markLayerDirty.mock.calls.length;

    engine.tools.setTool('transform');
    engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 20 });
    dispatch.mockClear();
    engine.layers.applyTransform();

    // A fresh CONTENT-sized surface (the transformed content bounds, 100×100 here)
    // was allocated and drawn through the bake matrix. The 50×50 cache at scale 2 +
    // offset (10,20) bakes to bounds {10,20,100,100}, so the surface is 100×100 and
    // the bake transform's translation is shifted by the baked origin to (0,0).
    const baked = surfaces[surfacesBeforeApply];
    expect(baked).toBeDefined();
    expect(baked!.width).toBe(100);
    expect(baked!.height).toBe(100);
    const setTransforms = baked!.callLog.filter((entry) => entry.op === 'setTransform');
    // The non-identity setTransform carries the bake matrix scale (a=2, d=2), with
    // its translation shifted into baked-local space (e=0, f=0).
    expect(
      setTransforms.some(
        (entry) => entry.args[0] === 2 && entry.args[3] === 2 && entry.args[4] === 0 && entry.args[5] === 0
      )
    ).toBe(true);
    expect(baked!.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    // Determinism: the bake always smooths (a doc-space resample, unlike the
    // live composite which varies smoothing with zoom) rather than leaving it
    // at whatever the fresh surface's context happens to default to.
    expect(
      baked!.callLog.some(
        (entry) => entry.op === 'set' && entry.args[0] === 'imageSmoothingEnabled' && entry.args[1] === true
      )
    ).toBe(true);

    // The reducer transform was reset to identity (pixel-free structural change).
    const resetDispatch = dispatch.mock.calls
      .map((call) => call[0] as EngineTestAction)
      .find((action) => action.type === 'updateCanvasLayer');
    expect(resetDispatch).toEqual({
      id: 'paint1',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });

    // Baked pixels persist through the normal dirty path; session cleared; undoable.
    expect(bitmapStore.markLayerDirty.mock.calls.length).toBeGreaterThan(dirtyBefore);
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.stores.canUndo.get()).toBe(true);

    // Undo restores BOTH the old transform and the old pixels in one step.
    dispatch.mockClear();
    engine.history.undo();
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

    // A dims change is a wholesale document replacement.
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

    // Delete the session's layer via an ordinary layer-array edit (same dims,
    // same revision) — NOT a wholesale replace, so this exercises the
    // `onLayersChanged` teardown rather than `onDocumentReplaced`'s.
    setDocument({ ...doc, layers: [] });

    expect(engine.stores.transformSession.get()).toBeNull();

    engine.lifecycle.dispose();
  });

  // ---- temp-tool switch (space/alt hold) must not discard the session ----
  //
  // The pointer pipeline flags a modifier-hold switch (and its matching
  // restore) with `{ temporary: true }` on `setTool` (see
  // `pointerPipeline.test.ts`, "temporary modifier tools"). These tests drive
  // that same seam directly against the engine: the public `CanvasEngine`
  // type narrows `setTool` to one argument, but the runtime function accepts
  // the pipeline's second (`opts`) argument, so the cast below exercises the
  // exact call the pipeline makes on a real space/alt hold and release.
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
      // A numeric edit, as the options bar would drive.
      engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 1, x: 5, y: 0 });
      const edited = { rotation: 0, scaleX: 2, scaleY: 1, x: 5, y: 0 };
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      const setTool = engine.tools.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void;

      setTool('view', { temporary: true }); // space down
      expect(engine.stores.activeTool.get()).toBe('view');
      // The session — and the numeric edit — survive the hold.
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      setTool('transform', { temporary: true }); // space up: resume
      expect(engine.stores.activeTool.get()).toBe('transform');
      // Resuming does not reopen the session from the layer's committed
      // transform, discarding the edit.
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      engine.lifecycle.dispose();
    });

    it('a REAL tool switch (not temporary) still cancels the session', () => {
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
      expect(engine.stores.transformSession.get()).not.toBeNull();

      engine.tools.setTool('view'); // a real switch — no `{ temporary: true }`

      expect(engine.stores.transformSession.get()).toBeNull();

      engine.lifecycle.dispose();
    });

    it('cancels cleanly when the session layer is deleted mid-hold, instead of resurrecting it on resume', () => {
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
      expect(engine.stores.transformSession.get()).not.toBeNull();

      const setTool = engine.tools.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void;
      setTool('view', { temporary: true }); // space down

      // The session's layer is deleted while temp-switched away (e.g. via the
      // layers panel) — the layer-change teardown cancels the session
      // immediately, regardless of which tool is active.
      setDocument({ ...doc, layers: [] });
      expect(engine.stores.transformSession.get()).toBeNull();

      setTool('transform', { temporary: true }); // space up: resume
      // Resuming must not resurrect a session against the now layer-less
      // document.
      expect(engine.stores.transformSession.get()).toBeNull();

      engine.lifecycle.dispose();
    });
  });
});

// ---- Selection subsystem ------------------------------------------------

const lockedPaintDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [
    {
      blendMode: 'normal',
      id: 'paint1',
      isEnabled: true,
      isLocked: true,
      name: 'paint1',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    },
  ],
  selectedLayerId: 'paint1',
  version: 2,
  width: 100,
});

describe('engine selection: select all / deselect / invert + hasSelection store', () => {
  it('selectAll sets hasSelection; deselect clears it', () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
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

  it('invertSelection of an empty selection selects everything', () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
    const { store } = createFakeStore(paintDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.selection.invertSelection();
    expect(engine.stores.hasSelection.get()).toBe(true);
    engine.lifecycle.dispose();
  });
});

describe('engine selection: fill / erase', () => {
  const makeEngine = (doc: CanvasDocumentContractV2) => {
    vi.stubGlobal('Path2D', class FakePath2D {});
    const { store } = createFakeStore(doc);
    const bitmapStore = createSpyBitmapStore();
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    return { bitmapStore, engine };
  };

  it('fills an empty selected paint control without adding a raster layer', () => {
    const h = createControlSelectionHarness({ source: { bitmap: null, type: 'paint' } });
    h.engine.selection.selectAll();
    h.engine.selection.fillSelection();
    expect(h.engine.document.getDocument()!.layers).toHaveLength(1);
    expect(h.engine.document.getDocument()!.layers[0]).toMatchObject({ id: 'control', type: 'control' });
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('erases an existing selected paint control without adding a raster layer', async () => {
    const h = createControlSelectionHarness({
      source: { bitmap: { height: 10, imageName: 'paint-bitmap', width: 10 }, type: 'paint' },
    });
    await h.publishInitialCache();
    h.engine.selection.selectAll();
    h.engine.selection.eraseSelection();
    expect(h.engine.document.getDocument()!.layers).toHaveLength(1);
    expect(h.engine.document.getDocument()!.layers[0]).toMatchObject({ id: 'control', type: 'control' });
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it.each(['fill', 'erase'] as const)('materializes an image control and %ss it as one undo step', async (kind) => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument()!.layers[0]);
    h.engine.selection.selectAll();
    if (kind === 'fill') {
      h.engine.selection.fillSelection();
    } else {
      h.engine.selection.eraseSelection();
    }
    const after = structuredClone(h.engine.document.getDocument()!.layers[0]);
    expect(after).toMatchObject({ id: 'control', source: { type: 'paint' }, type: 'control' });
    h.engine.history.undo();
    expect(h.engine.document.getDocument()!.layers[0]).toEqual(before);
    h.engine.history.redo();
    expect(h.engine.document.getDocument()!.layers[0]).toEqual(after);
    h.engine.lifecycle.dispose();
  });

  it('does not materialize an image control when Erase has no overlapping pixels', async () => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 50, y: 50 },
    });
    await h.publishInitialCache();
    const before = structuredClone(h.engine.document.getDocument());
    h.engine.tools.setTool('lasso');
    h.overlay.fire('pointerdown', pointerAt(0, 0));
    h.overlay.fire('pointermove', pointerAt(10, 0));
    h.overlay.fire('pointermove', pointerAt(10, 10));
    h.overlay.fire('pointermove', pointerAt(0, 10));
    h.overlay.fire('pointerup', pointerAt(0, 0, { buttons: 0 }));
    h.engine.selection.eraseSelection();
    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.suspendLayer).not.toHaveBeenCalled();
    h.engine.lifecycle.dispose();
  });

  it.each([
    ['locked', { isLocked: true }],
    ['disabled', { isEnabled: false }],
  ] as const)('selection editing leaves a %s control unchanged', (_scenario, patch) => {
    const h = createControlSelectionHarness({ source: { bitmap: null, type: 'paint' }, ...patch });
    const before = structuredClone(h.engine.document.getDocument());
    h.engine.selection.selectAll();
    h.engine.selection.fillSelection();
    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    h.engine.lifecycle.dispose();
  });

  it('selection editing leaves a not-ready image control unchanged', () => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'pending-image', width: 10 }, type: 'image' },
    });
    const before = structuredClone(h.engine.document.getDocument());
    h.engine.selection.selectAll();
    h.engine.selection.fillSelection();
    expect(h.engine.document.getDocument()).toEqual(before);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    h.engine.lifecycle.dispose();
  });

  it.each(['fill', 'erase'] as const)('does not publish a byte-identical direct control %s', async (kind) => {
    const h = createControlSelectionHarness({
      source: { bitmap: { height: 10, imageName: 'paint-bitmap', width: 10 }, type: 'paint' },
    });
    await h.publishInitialCache();
    h.engine.tools.setTool('lasso');
    h.overlay.fire('pointerdown', pointerAt(0, 0));
    h.overlay.fire('pointermove', pointerAt(5, 0));
    h.overlay.fire('pointermove', pointerAt(5, 5));
    h.overlay.fire('pointermove', pointerAt(0, 5));
    h.overlay.fire('pointerup', pointerAt(0, 0, { buttons: 0 }));
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    const beforeThumbnailVersion = h.engine.stores.thumbnailVersion.get('control');
    h.setSelectionPixelWrites(false);

    if (kind === 'fill') {
      h.engine.selection.fillSelection();
    } else {
      h.engine.selection.eraseSelection();
    }

    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.thumbnailVersion.get('control')).toBe(beforeThumbnailVersion);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }

    h.setSelectionPixelWrites(true);
    if (kind === 'fill') {
      h.engine.selection.fillSelection();
    } else {
      h.engine.selection.eraseSelection();
    }
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('restores direct control cache growth after a byte-identical fill', async () => {
    const h = createControlSelectionHarness({ source: { bitmap: null, type: 'paint' } });
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeExport = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    const beforeThumbnailVersion = h.engine.stores.thumbnailVersion.get('control');
    h.engine.selection.selectAll();
    h.setSelectionPixelWrites(false);

    h.engine.selection.fillSelection();

    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.thumbnailVersion.get('control')).toBe(beforeThumbnailVersion);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe(beforeExport.status);

    h.setSelectionPixelWrites(true);
    h.engine.selection.fillSelection();
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('rolls back a byte-identical materialized control selection edit', async () => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    const beforeThumbnailVersion = h.engine.stores.thumbnailVersion.get('control');
    h.engine.selection.selectAll();
    h.setSelectionPixelWrites(false);

    h.engine.selection.fillSelection();

    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.thumbnailVersion.get('control')).toBe(beforeThumbnailVersion);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }

    h.setSelectionPixelWrites(true);
    h.engine.selection.fillSelection();
    expect(h.engine.stores.canUndo.get()).toBe(true);
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledTimes(2);
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledTimes(2);
    h.engine.lifecycle.dispose();
  });

  it('keeps materialized no-effect rollback authoritative when growth restoration throws once', async () => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    const beforeThumbnailVersion = h.engine.stores.thumbnailVersion.get('control');
    const createSurface = h.backend.createSurface.bind(h.backend);
    let didThrow = false;
    vi.spyOn(h.backend, 'createSurface').mockImplementation((width, height) => {
      const surface = createSurface(width, height);
      const resize = surface.resize.bind(surface);
      surface.resize = (nextWidth, nextHeight) => {
        if (!didThrow && surface.width === 100 && nextWidth === 20 && nextHeight === 20) {
          didThrow = true;
          throw new Error('selection rollback restoration failed');
        }
        resize(nextWidth, nextHeight);
      };
      return surface;
    });
    h.engine.selection.selectAll();
    h.setSelectionPixelWrites(false);

    expect(() => h.engine.selection.fillSelection()).toThrow('selection rollback restoration failed');

    expect(didThrow).toBe(true);
    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledOnce();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    expect(h.engine.stores.thumbnailVersion.get('control')).toBe(beforeThumbnailVersion);
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }

    vi.mocked(h.backend.createSurface).mockRestore();
    h.setSelectionPixelWrites(true);
    h.engine.selection.fillSelection();
    expect(h.engine.stores.canUndo.get()).toBe(true);
    expect(h.bitmapStore.suspendLayer).toHaveBeenCalledTimes(2);
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledTimes(2);
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
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(capturedReads[0]).toMatchObject({ height: 10, width: 10 });
    expect(capturedReads.at(-1)).toMatchObject({ height: 5, width: 5 });
    expect(
      (beforeCache.surface as StubRasterSurface).callLog.filter((entry) => entry.op === 'putImageData').at(-1)?.args[0]
    ).toBe(capturedReads[0]);

    Object.defineProperty(beforeCache.surface, 'ctx', { configurable: true, value: originalCtx });
    h.engine.selection.fillSelection();
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('rolls back a materialized control selection edit when masked compositing fails', async () => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    const createSurface = h.backend.createSurface.bind(h.backend);
    vi.spyOn(h.backend, 'createSurface').mockImplementation((width, height) => {
      const surface = createSurface(width, height);
      const originalCtx = surface.ctx;
      const failingCtx = new Proxy(originalCtx, {
        get(target, property, receiver) {
          const value = Reflect.get(target, property, receiver);
          if (property !== 'drawImage' || typeof value !== 'function') {
            return value;
          }
          return (...args: unknown[]) => {
            const result = Reflect.apply(value, target, args);
            if (target.globalCompositeOperation === 'destination-out') {
              throw new Error('selection compositing failed');
            }
            return result;
          };
        },
      });
      Object.defineProperty(surface, 'ctx', { configurable: true, value: failingCtx });
      return surface;
    });
    h.engine.selection.selectAll();

    expect(() => h.engine.selection.eraseSelection()).toThrow('selection compositing failed');

    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }

    vi.mocked(h.backend.createSurface).mockRestore();
    h.engine.selection.eraseSelection();
    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('preserves transaction-owned rollback when a materialized control selection commit fails', async () => {
    const h = createControlSelectionHarness({
      source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 },
    });
    await h.publishInitialCache();
    const beforeDocument = structuredClone(h.engine.document.getDocument());
    const beforeCache = await snapshotLayerCache(h.engine, 'control');
    h.engine.selection.selectAll();
    historyPreparationFaults.layerSnapshot = true;

    expect(() => h.engine.selection.fillSelection()).toThrow('layer snapshot preparation failed');

    expect(h.engine.document.getDocument()).toEqual(beforeDocument);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    expect(h.bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    expect(h.bitmapStore.releaseSuspendedLayer).toHaveBeenCalledOnce();
    const restored = await h.engine.exports.exportLayerPixels('control', { includeDisabled: true });
    expect(restored.status).toBe('ok');
    if (restored.status === 'ok') {
      expect(restored.surface).toBe(beforeCache.surface);
      expect(restored.rect).toEqual(beforeCache.rect);
      expect(restored.guard.cacheVersion).toBe(beforeCache.version);
    }
    h.engine.lifecycle.dispose();
  });

  it('fillSelection on the selected paint layer records one undoable edit + persists', () => {
    const { bitmapStore, engine } = makeEngine(paintDoc());
    engine.selection.selectAll();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.selection.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledWith('paint1');
    // Undo restores (canRedo becomes available).
    engine.history.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('blocks paint without closing a guarded operation', async () => {
    const { engine } = makeEngine(paintDoc());
    engine.selection.selectAll();
    engine.selection.fillSelection();
    const exported = await engine.exports.exportLayerPixels('paint1');
    if (exported.status !== 'ok') {
      throw new Error('expected published paint pixels');
    }
    const cleanupPreview = vi.fn();
    getCanvasOperations(engine).controller.start({
      cleanupPreview,
      guard: exported.guard,
      identity: { kind: 'filter', layerId: 'paint1', projectId: 'p1' },
    });

    engine.selection.fillSelection();

    expect(cleanupPreview).not.toHaveBeenCalled();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'filter' },
      status: 'active',
    });
    engine.lifecycle.dispose();
  });

  it('cache clear cancels a guarded operation and cleans its preview', async () => {
    const { engine } = makeEngine(paintDoc());
    engine.selection.selectAll();
    engine.selection.fillSelection();
    const exported = await engine.exports.exportLayerPixels('paint1');
    if (exported.status !== 'ok') {
      throw new Error('expected published paint pixels');
    }
    const cleanupPreview = vi.fn();
    getCanvasOperations(engine).controller.start({
      cleanupPreview,
      guard: exported.guard,
      identity: { kind: 'filter', layerId: 'paint1', projectId: 'p1' },
    });

    await engine.diagnostics.clearCaches();

    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    engine.lifecycle.dispose();
  });

  it('eraseSelection records one undoable edit (on existing content)', () => {
    const { engine } = makeEngine(paintDoc());
    engine.selection.selectAll();
    // Content-sized: erase only affects EXISTING pixels, so give the layer content
    // first (a fill grows the empty paint cache to the selection). Then erase records
    // its own edit within that extent.
    engine.selection.fillSelection();
    engine.selection.eraseSelection();
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('eraseSelection is a no-op on an EMPTY paint layer (no pixels to erase)', () => {
    const { engine } = makeEngine(paintDoc());
    engine.selection.selectAll();
    engine.selection.eraseSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('fillSelection is a no-op with no selection', () => {
    const { engine } = makeEngine(paintDoc());
    engine.selection.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('fillSelection is a no-op on an image-source layer (deferred to rasterize task)', () => {
    const { engine } = makeEngine(imageSelectedDoc());
    engine.selection.selectAll();
    engine.selection.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('fillSelection is a no-op on a transparency-locked EMPTY layer (source-atop clamps to existing pixels)', () => {
    const doc = paintDoc();
    const target = doc.layers[0];
    if (target?.type === 'raster') {
      target.isTransparencyLocked = true;
    }
    const { engine } = makeEngine(doc);
    engine.selection.selectAll();
    engine.selection.fillSelection();
    // The layer has no existing pixels, so a transparency-locked (source-atop) fill
    // lands nothing — no undoable edit.
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('fillSelection is a no-op on a locked paint layer', () => {
    const { engine } = makeEngine(lockedPaintDoc());
    engine.selection.selectAll();
    engine.selection.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('clearCaches flushes pending bitmap uploads before invalidating (unflushed strokes survive)', async () => {
    const { bitmapStore, engine } = makeEngine(paintDoc());
    // The debug "Clear caches" action must persist any in-flight (debounced) paint
    // upload before it drops the layer caches — otherwise an unflushed stroke is lost.
    await engine.diagnostics.clearCaches();
    expect(bitmapStore.flushPendingUploads).toHaveBeenCalledTimes(1);
    engine.lifecycle.dispose();
  });
});

describe('engine selection: marching ants animation + overlay-only', () => {
  it('a selection redraws the overlay with ants and never composites the document', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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
    overlay.surface.callLog.length = 0;

    engine.selection.selectAll();
    raf.flush();

    // Overlay redrawn (ants stroked); the screen composite never ran.
    const compositeOps = screen.surface.callLog.filter(
      (e) => e.op === 'drawImage' || e.op === 'clearRect' || e.op === 'fillRect'
    );
    expect(compositeOps).toHaveLength(0);
    expect(overlay.surface.callLog.some((e) => e.op === 'stroke')).toBe(true);

    engine.lifecycle.dispose();
  });

  it('stops the ants loop (no pending frames) on deselect and on detach', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

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

    // With a selection the loop keeps rescheduling itself frame after frame.
    engine.selection.selectAll();
    raf.flush();
    raf.flush();
    expect(raf.pendingCount()).toBeGreaterThan(0);

    // Deselect stops it: after draining, nothing reschedules.
    engine.selection.deselect();
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    // Re-select, then detach: the loop must not leak a pending frame.
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
    vi.stubGlobal('Path2D', class FakePath2D {});

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
    // Selection is transient: no reducer traffic from the lasso gesture.
    expect(dispatch).not.toHaveBeenCalled();

    engine.lifecycle.dispose();
  });
});

describe('engine selection: document replace clears the selection', () => {
  it('a wholesale document swap deselects', () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
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
    // A new document revision replaces the mirror.
    setDocument(paintDoc(), 1);
    expect(engine.stores.hasSelection.get()).toBe(false);
    engine.lifecycle.dispose();
  });
});

// ---- text edit session: create/edit commit, cancel, teardown ---------------

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

  const textDoc = (over: Partial<CanvasDocumentContractV2> = {}): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers: [
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
    ],
    selectedLayerId: 'txt1',
    version: 2,
    width: 100,
    ...over,
  });

  const makeEngine = (doc: CanvasDocumentContractV2) => {
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

  it('create-mode commit dispatches ONE addCanvasLayer with the typed content; undo removes it', () => {
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
    engine.history.undo();
    const removes = layerActions().filter((a) => a.type === 'removeCanvasLayers');
    expect(removes).toHaveLength(1);
    engine.lifecycle.dispose();
  });

  it('an empty create-mode commit dispatches nothing (cancel semantics)', () => {
    const { dispatch, engine } = makeEngine(paintDoc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 0, y: 0 });
    dispatch.mockClear();
    engine.layers.commitTextEdit('   ');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('edit-mode commit dispatches ONE updateCanvasLayerSource with the exact inverse', () => {
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
    engine.history.undo();
    const inverse = layerActions().find((a) => a.type === 'updateCanvasLayerSource');
    if (inverse?.type === 'updateCanvasLayerSource' && inverse.source.type === 'text') {
      expect(inverse.source.content).toBe('hello');
    } else {
      throw new Error('expected the inverse to restore the original content');
    }
    engine.lifecycle.dispose();
  });

  it('folds a live style change into the single edit commit', () => {
    const { dispatch, engine, layerActions } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    engine.layers.updateTextEditStyle({ color: '#ff0000', fontSize: 40 });

    dispatch.mockClear();
    engine.layers.commitTextEdit('hello');

    const edits = layerActions().filter((a) => a.type === 'updateCanvasLayerSource');
    expect(edits).toHaveLength(1);
    const forward = edits[0];
    if (forward?.type === 'updateCanvasLayerSource' && forward.source.type === 'text') {
      expect(forward.source.color).toBe('#ff0000');
      expect(forward.source.fontSize).toBe(40);
      expect(forward.source.content).toBe('hello');
    } else {
      throw new Error('expected the folded style change');
    }
    engine.lifecycle.dispose();
  });

  it('an unchanged edit-mode commit dispatches nothing', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    dispatch.mockClear();
    engine.layers.commitTextEdit('hello');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('cancel drops the session with no dispatch', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    dispatch.mockClear();
    engine.layers.cancelTextEdit();
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('does not open an edit session on a locked text layer', () => {
    const { engine } = makeEngine(
      textDoc({
        layers: [
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
        ],
      })
    );
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('a temporary tool switch (space-hold) preserves the open session', () => {
    const { engine } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).not.toBeNull();

    // Space-hold switches to the view tool temporarily; the session must survive.
    (engine.tools.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void)('view', { temporary: true });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    engine.lifecycle.dispose();
  });

  it('a real tool switch away from the text tool tears the session down', () => {
    const { engine } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).not.toBeNull();

    engine.tools.setTool('brush');
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('drops the session on a wholesale document replace', () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
    const { engine, setDocument } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    setDocument(paintDoc(), 1);
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

  // ---- click-elsewhere-to-commit (pointerdown / Escape) ----
  //
  // Regression: `commitTextEdit` only fired on the portal's blur, but the pointer
  // pipeline `preventDefault`s a canvas pointerdown (suppressing that blur), and
  // the mid-gesture guard swallowed any blur that did land. The commit now runs
  // engine-side on pointerdown (before a gesture starts), reading the live portal
  // content via a registered reader; Escape cancels a defocused session via the
  // engine's escape ladder. These drive that engine-side logic directly.

  it('commitOpenTextSession reads the registered content reader and commits the create', () => {
    const { engine, layerActions } = makeEngine(paintDoc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 10, y: 20 });
    engine.layers.setTextEditContentReader(() => 'live typed text');

    expect(engine.layers.commitOpenTextSession()).toBe(true);

    const adds = layerActions().filter((a) => a.type === 'addCanvasLayer');
    expect(adds).toHaveLength(1);
    const add = adds[0];
    if (add?.type === 'addCanvasLayer' && add.layer.type === 'raster' && add.layer.source.type === 'text') {
      expect(add.layer.source.content).toBe('live typed text');
    } else {
      throw new Error('expected an addCanvasLayer with the live reader content');
    }
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('commitOpenTextSession returns false when no session is open', () => {
    const { dispatch, engine } = makeEngine(paintDoc());
    expect(engine.layers.commitOpenTextSession()).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
    engine.lifecycle.dispose();
  });

  it('commitOpenTextSession falls back to the session content when no reader is registered', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.tools.setTool('text');
    engine.layers.openTextEdit('txt1'); // session content = 'hello'
    dispatch.mockClear();

    expect(engine.layers.commitOpenTextSession()).toBe(true);
    // No reader → uses the session's own content ('hello') → unchanged edit → no dispatch.
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.lifecycle.dispose();
  });

  it('a second commit after the session closed is a no-op (one commit per close)', () => {
    const { dispatch, engine, layerActions } = makeEngine(paintDoc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 0, y: 0 });
    engine.layers.setTextEditContentReader(() => 'once');
    expect(engine.layers.commitOpenTextSession()).toBe(true);
    dispatch.mockClear();
    // The portal's onBlur would re-fire commitTextEdit after the pointerdown commit;
    // the session is already null, so it dispatches nothing.
    engine.layers.commitTextEdit('once');
    expect(engine.layers.commitOpenTextSession()).toBe(false);
    expect(layerActions().filter((a) => a.type === 'addCanvasLayer')).toHaveLength(0);
    engine.lifecycle.dispose();
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

describe('contextMenuLayerIdAt (canvas right-click target)', () => {
  const controlLayer = (id: string): CanvasLayerContract => ({
    adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: false,
  });

  // A raster at index 0 with a control layer above it, both covering [0,10]². At
  // the default viewport (zoom 1, no pan) a screen point maps 1:1 to document space.
  const stackedDoc = (): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers: [rasterLayer('raster'), controlLayer('control')],
    selectedLayerId: null,
    version: 2,
    width: 100,
  });

  const makeEngine = (doc: CanvasDocumentContractV2) => {
    const { store } = createFakeStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    return engine;
  };

  it('returns the composite-top layer at the point (control over raster, batch finding N1)', () => {
    const engine = makeEngine(stackedDoc());
    // The raster is earlier in the array but the control composites above it.
    expect(engine.tools.contextMenuLayerIdAt({ x: 5, y: 5 })).toBe('control');
    engine.lifecycle.dispose();
  });

  it('returns null on empty space', () => {
    const engine = makeEngine(stackedDoc());
    expect(engine.tools.contextMenuLayerIdAt({ x: 60, y: 60 })).toBeNull();
    engine.lifecycle.dispose();
  });

  it('returns null while a text-edit session is open (never opens over an in-progress edit)', () => {
    const engine = makeEngine(stackedDoc());
    engine.tools.setTool('text');
    engine.layers.openTextCreate({ x: 5, y: 5 });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    expect(engine.tools.contextMenuLayerIdAt({ x: 5, y: 5 })).toBeNull();
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

  /** A raster layer whose baked world rect exactly covers `rect`. */
  const samRectLayer = (id: string, rect: CanvasDocumentContractV2['bbox']): CanvasRasterLayerContractV2 => ({
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

  const samDocument = (layers: CanvasLayerContract[] = [samLayer('source')]): CanvasDocumentContractV2 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    layers,
    selectedLayerId: layers[0]?.id ?? null,
    version: 2,
    width: 100,
  });

  const createOpaqueSamBackend = () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
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
      bbox?: CanvasDocumentContractV2['bbox'];
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

  it('keeps Select Object visible and inputs intact when the first Process fails to upload, then retries', async () => {
    const bbox = { height: 100, width: 100, x: 0, y: 0 };
    const reactive = createReactiveStore(samDocument());
    const runGraph = vi.fn(() =>
      Promise.resolve({ height: bbox.height, imageName: 'sam-mask.png', origin: 'test', width: bbox.width })
    );
    const uploadIntermediate = vi
      .fn<NonNullable<CanvasEngineOptions['selectObjectDeps']>['uploadIntermediate']>()
      .mockRejectedValueOnce(new Error('upload failed'))
      .mockResolvedValueOnce({ height: bbox.height, imageName: 'sam-source.png', width: bbox.width });
    const engine = createCanvasEngine({
      backend: {
        ...createTestStubRasterBackend(),
        createImageBitmap: () =>
          Promise.resolve({ close: () => undefined, height: bbox.height, width: bbox.width } as unknown as ImageBitmap),
      },
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      selectObjectDeps: {
        runGraph,
        uploadIntermediate,
      },
      store: reactive.store,
    });
    await engine.exports.exportLayerPixels('source');
    const input = { prompt: 'keep this', type: 'prompt' as const };
    expect(getCanvasOperations(engine).startSelectObject('source')).toBe('started');
    getCanvasOperations(engine).updateSelectObjectSession({ input, invert: true });

    await expect(getCanvasOperations(engine).processSelectObjectSession()).resolves.toBe('error');

    expect(runGraph).not.toHaveBeenCalled();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({ phase: 'error', status: 'active' });
    expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({
      hasPreview: false,
      input,
      invert: true,
    });

    await expect(getCanvasOperations(engine).processSelectObjectSession()).resolves.toBe('published');
    expect(runGraph).toHaveBeenCalledOnce();
    expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({
      hasPreview: true,
      input,
      invert: true,
      status: 'ready',
    });
    engine.lifecycle.dispose();
  });

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
    bbox: CanvasDocumentContractV2['bbox'] = { height: 100, width: 100, x: 0, y: 0 }
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

  const invalidateSelectObjectCommit = async (
    engine: CanvasEngine,
    kind: 'input' | 'reset' | 'process'
  ): Promise<void> => {
    if (kind === 'input') {
      getCanvasOperations(engine).updateSelectObjectSession({ input: { prompt: 'new input', type: 'prompt' } });
    } else if (kind === 'reset') {
      getCanvasOperations(engine).resetSelectObjectSession();
    } else {
      await getCanvasOperations(engine).processSelectObjectSession();
    }
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

  it('resets edited inputs and point mode immediately after launch without requiring a preview guard', async () => {
    const h = await createSamHarness();
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: {
        bbox: { height: 10, width: 10, x: 1, y: 2 },
        excludePoints: [{ x: 4, y: 5 }],
        includePoints: [{ x: 2, y: 3 }],
        type: 'visual',
      },
      invert: true,
      pointLabel: 'exclude',
    });

    expect(getCanvasOperations(h.engine).resetSelectObjectSession()).toBe('updated');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: null,
      hasPreview: false,
      input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
      invert: false,
      pointLabel: 'include',
      status: 'ready',
    });
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({ status: 'active' });
    h.engine.lifecycle.dispose();
  });

  it('resets after a first failed Process while preserving lifecycle ownership', async () => {
    const bbox = { height: 100, width: 100, x: 0, y: 0 };
    const reactive = createReactiveStore(samDocument());
    const engine = createCanvasEngine({
      backend: withSamPreviewBitmapDimensions(createTestStubRasterBackend()),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      selectObjectDeps: {
        runGraph: () => Promise.reject(new Error('queue failed')),
        uploadIntermediate: () => Promise.resolve({ height: 100, imageName: 'source.png', width: 100 }),
      },
      store: reactive.store,
    });
    await engine.exports.exportLayerPixels('source');
    expect(getCanvasOperations(engine).startSelectObject('source')).toBe('started');
    getCanvasOperations(engine).updateSelectObjectSession({ input: { prompt: 'cat', type: 'prompt' }, invert: true });
    await expect(getCanvasOperations(engine).processSelectObjectSession()).resolves.toBe('error');
    expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({
      error: { code: 'queue' },
      status: 'error',
    });

    expect(getCanvasOperations(engine).resetSelectObjectSession()).toBe('updated');

    expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({
      error: null,
      hasPreview: false,
      input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
      invert: false,
      pointLabel: 'include',
      status: 'ready',
    });
    expect(getCanvasOperations(engine).controller.getSnapshot()).toMatchObject({ status: 'active' });
    expect(engine.document.getDocument()?.bbox).toEqual(bbox);
    engine.lifecycle.dispose();
  });

  it('reuses one uploaded source image across fresh engine guards for point, invert, and refinement changes', async () => {
    const uploadIntermediate = vi.fn(() => Promise.resolve({ height: 100, imageName: 'source.png', width: 100 }));
    const runGraph = vi.fn(() => Promise.resolve({ height: 100, imageName: 'mask.png', origin: 'test', width: 100 }));
    const h = await createSamHarness({ runGraph, uploadIntermediate });
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 2, y: 2 }], type: 'visual' },
    });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');

    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 3, y: 3 }], type: 'visual' },
    });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');
    getCanvasOperations(h.engine).updateSelectObjectSession({ invert: true });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');
    getCanvasOperations(h.engine).updateSelectObjectSession({ applyPolygonRefinement: true });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');

    expect(uploadIntermediate).toHaveBeenCalledOnce();
    expect(runGraph).toHaveBeenCalledTimes(4);
    h.engine.lifecycle.dispose();
  });

  it('keeps the selection when starting on the selected layer and selects a non-selected source', async () => {
    const h = await createSamHarness({ layers: [samLayer('first'), samLayer('second', 30)] });

    expect(getCanvasOperations(h.engine).startSelectObject('first')).toBe('started');
    expect(h.store.dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'setCanvasSelectedLayer' }));

    expect(getCanvasOperations(h.engine).startSelectObject('second')).toBe('started');
    expect(h.store.dispatch).toHaveBeenCalledWith({ id: 'second', type: 'setCanvasSelectedLayer' });
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

  it('builds the prompt path and reports a decoded preview', async () => {
    const runGraph = vi.fn<NonNullable<CanvasEngineOptions['selectObjectDeps']>['runGraph']>((options) => {
      expect(options.graph.nodes['sam-detect']).toMatchObject({ prompt: 'red car', type: 'grounding_dino' });
      return Promise.resolve({ height: 100, imageName: 'prompt-mask.png', origin: 'test', width: 100 });
    });
    const h = await createSamHarness({ runGraph });
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({ input: { prompt: '  red car  ', type: 'prompt' } });

    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({ hasPreview: true, status: 'ready' });
    h.engine.lifecycle.dispose();
  });

  it('rejects SAM metadata dimensions that do not match the decoded preview rect', async () => {
    const h = await createSamHarness({
      runGraph: () => Promise.resolve({ height: 99, imageName: 'bad-mask.png', origin: 'test', width: 100 }),
    });
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({ input: { prompt: 'cat', type: 'prompt' } });

    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('error');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: { code: 'output-dimension', detail: 'SAM output dimensions 100x99 do not match 100x100.' },
      hasPreview: false,
      status: 'error',
    });
    h.engine.lifecycle.dispose();
  });

  it('reports preview decode failures without publishing a stale surface', async () => {
    const h = await createSamHarness({ imageResolver: () => Promise.reject(new Error('preview decode failed')) });
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({ input: { prompt: 'cat', type: 'prompt' } });

    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('error');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: { code: 'decode', detail: 'preview decode failed' },
      hasPreview: false,
      status: 'error',
    });
    h.engine.lifecycle.dispose();
  });

  it.each([
    { decoded: { height: 100, width: 99 }, label: 'smaller' },
    { decoded: { height: 100, width: 101 }, label: 'larger' },
    { decoded: { height: 100, width: Number.NaN }, label: 'non-finite' },
    { decoded: { height: 100, width: 99.5 }, label: 'non-integer' },
  ])('rejects a $label decoded bitmap before surface allocation and closes it', async ({ decoded }) => {
    const close = vi.fn();
    const base = createTestStubRasterBackend();
    let didDecode = false;
    const previewSurfaceAllocations = vi.fn();
    const backend = {
      ...base,
      createImageBitmap: () => {
        didDecode = true;
        return Promise.resolve({ ...decoded, close } as unknown as ImageBitmap);
      },
      createSurface: (width: number, height: number) => {
        if (didDecode) {
          previewSurfaceAllocations(width, height);
        }
        return base.createSurface(width, height);
      },
    };
    const h = await createSamHarness({ backend });
    getCanvasOperations(h.engine).startSelectObject('source');
    getCanvasOperations(h.engine).updateSelectObjectSession({ input: { prompt: 'cat', type: 'prompt' } });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('error');

    expect(close).toHaveBeenCalledOnce();
    expect(previewSurfaceAllocations).not.toHaveBeenCalled();
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: {
        code: 'output-dimension',
        detail: `Decoded Select Object preview dimensions ${String(decoded.width)}x${decoded.height} do not match SAM output 100x100 and preview rect 100x100.`,
      },
      hasPreview: false,
      status: 'error',
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
    const replaced = h.engine.document.getDocument()!.layers.find((layer) => layer.id === 'source')!;
    expect(replaced).toMatchObject({
      source: {
        bitmap: { height: 100, imageName: 'sam-mask.png', width: 100 },
        offset: { x: 0, y: 0 },
        type: 'paint',
      },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(h.engine.document.getDocument()!.layers).toHaveLength(1);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(h.engine.stores.activeTool.get()).toBe('view');

    expect(h.engine.stores.canUndo.get()).toBe(true);
    h.engine.history.undo();
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.history.redo();
    expect(h.engine.document.getDocument()!.layers.find((layer) => layer.id === 'source')).toEqual(replaced);
    h.engine.lifecycle.dispose();
  });

  it('blocks Select Object actions called directly after an external lock and still allows Cancel', async () => {
    const h = await createCommittingSamHarness();
    const makeDurable = vi.fn(() => Promise.resolve());
    const before = getCanvasOperations(h.engine).stores.samSession.get();

    h.engine.tools.setInteractionLocked(true);

    expect(
      getCanvasOperations(h.engine).updateSelectObjectSession({ input: { prompt: 'changed', type: 'prompt' } })
    ).toBe('blocked');
    expect(getCanvasOperations(h.engine).resetSelectObjectSession()).toBe('blocked');
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('blocked');
    await expect(getCanvasOperations(h.engine).applySelectObjectSession(makeDurable)).resolves.toEqual({
      status: 'locked',
    });
    await expect(getCanvasOperations(h.engine).saveSelectObjectSession('raster', makeDurable)).resolves.toEqual({
      status: 'locked',
    });
    expect(makeDurable).not.toHaveBeenCalled();
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toEqual(before);

    getCanvasOperations(h.engine).cancelSelectObjectSession();
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    h.engine.tools.setInteractionLocked(false);
    expect(h.engine.stores.activeTool.get()).toBe('view');
    expect(getCanvasOperations(h.engine).updateSelectObjectSession({ invert: false })).toBe('stale');
    expect(getCanvasOperations(h.engine).resetSelectObjectSession()).toBe('stale');
    h.engine.lifecycle.dispose();
  });

  it('blocks Select Object save mutation when an external lock begins during durability', async () => {
    const h = await createCommittingSamHarness();
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => durable.promise);
    await vi.waitFor(() => expect(getCanvasOperations(h.engine).stores.samSession.get()?.status).toBe('committing'));

    h.engine.tools.setInteractionLocked(true);
    durable.resolve();

    await expect(pending).resolves.toEqual({ status: 'locked' });
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({ hasPreview: true, status: 'ready' });
    getCanvasOperations(h.engine).cancelSelectObjectSession();
    h.engine.lifecycle.dispose();
  });

  it('blocks Select Object apply mutation when an external lock begins during final decode', async () => {
    const finalDecode = createDeferred<Blob>();
    let resolutions = 0;
    const h = await createCommittingSamHarness(() => {
      resolutions += 1;
      return resolutions === 1 ? Promise.resolve(new Blob()) : finalDecode.promise;
    });
    const pending = getCanvasOperations(h.engine).applySelectObjectSession(() => Promise.resolve());
    await vi.waitFor(() => expect(resolutions).toBe(2));

    h.engine.tools.setInteractionLocked(true);
    finalDecode.resolve(new Blob());

    await expect(pending).resolves.not.toMatchObject({ status: 'committed' });
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({ hasPreview: true, status: 'ready' });
    getCanvasOperations(h.engine).cancelSelectObjectSession();
    h.engine.lifecycle.dispose();
  });

  it('interrupts Select Object processing on an external lock without losing inputs or operation', async () => {
    const graph = createDeferred<{ height: number; imageName: string; origin: string; width: number }>();
    const runGraph = vi.fn(() => graph.promise);
    const h = await createSamHarness({ runGraph });
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');
    const input = { prompt: 'keep this prompt', type: 'prompt' as const };
    expect(getCanvasOperations(h.engine).updateSelectObjectSession({ input, invert: true })).toBe('updated');
    const pending = getCanvasOperations(h.engine).processSelectObjectSession();
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.status).toBe('preparing-source');
    await vi.waitFor(() => expect(runGraph).toHaveBeenCalledOnce());

    h.engine.tools.setInteractionLocked(true);
    graph.resolve({ height: 100, imageName: 'late.png', origin: 'test', width: 100 });

    await expect(pending).resolves.toBe('stale');
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      hasPreview: false,
      input,
      invert: true,
      status: 'ready',
    });
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({ phase: 'ready', status: 'active' });
    getCanvasOperations(h.engine).cancelSelectObjectSession();
    h.engine.lifecycle.dispose();
  });

  it.each(['input', 'reset', 'process'] as const)(
    'does not apply or close when %s invalidates ownership during final selection decode',
    async (kind) => {
      const finalDecode = createDeferred<Blob>();
      let resolutions = 0;
      const h = await createCommittingSamHarness(() => {
        resolutions += 1;
        return resolutions === 1 ? Promise.resolve(new Blob()) : finalDecode.promise;
      });
      const pending = getCanvasOperations(h.engine).applySelectObjectSession(() => Promise.resolve());
      await vi.waitFor(() => expect(resolutions).toBe(2));

      await invalidateSelectObjectCommit(h.engine, kind);
      finalDecode.resolve(new Blob());

      await expect(pending).resolves.not.toMatchObject({ status: 'committed' });
      expect(h.engine.document.getDocument()).toEqual(h.document);
      expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
      h.engine.lifecycle.dispose();
    }
  );

  it('does not invalidate a pending save when preview isolation changes', async () => {
    const h = await createCommittingSamHarness();
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('inpaint_mask', () => durable.promise);

    getCanvasOperations(h.engine).updateSelectObjectSession({ isolatedPreview: false });
    durable.resolve();

    await expect(pending).resolves.toMatchObject({ status: 'committed' });
    expect(h.engine.document.getDocument()?.layers[0]?.type).toBe('inpaint_mask');
    h.engine.lifecycle.dispose();
  });

  it('Reset clears inputs, preview, and error while keeping the operation active', async () => {
    const h = await createCommittingSamHarness();
    getCanvasOperations(h.engine).updateSelectObjectSession({ input: { prompt: 'cat', type: 'prompt' }, invert: true });

    getCanvasOperations(h.engine).resetSelectObjectSession();

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: null,
      hasPreview: false,
      input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
      invert: false,
      status: 'ready',
    });
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({ status: 'active' });
    h.engine.lifecycle.dispose();
  });

  it.each([
    ['raster', 'Segmented Object'],
    ['control', 'Segmented Object Control'],
    ['inpaint_mask', null],
    ['regional_guidance', null],
  ] as const)(
    'promotes then atomically saves the guarded preview as %s at the top of its group, keeping the session',
    async (target, copyName) => {
      const calls: string[] = [];
      const h = await createCommittingSamHarness();

      const result = await getCanvasOperations(h.engine).saveSelectObjectSession(target, () => {
        calls.push('durable');
        return Promise.resolve();
      });

      expect(result.status).toBe('committed');
      expect(calls[0]).toBe('durable');
      expect(h.engine.document.getDocument()?.layers[0]?.type).toBe(target);
      expect(h.engine.document.getDocument()?.layers[1]?.id).toBe('source');
      const created = h.engine.document.getDocument()?.layers[0];
      if (!created) {
        throw new Error('expected a saved object layer');
      }
      if (created.type === 'raster' || created.type === 'control') {
        expect(created.name).toBe(copyName);
        expect(created.source).toMatchObject({ offset: { x: 0, y: 0 }, type: 'paint' });
      } else {
        expect(created.mask).toMatchObject({ offset: { x: 0, y: 0 } });
      }
      expect(created.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });

      // Save As keeps the session and its preview open; only Apply, the
      // Selection target, and Cancel end it.
      expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
        hasPreview: true,
        status: 'ready',
      });
      expect(h.engine.stores.activeTool.get()).toBe('sam');
      getCanvasOperations(h.engine).cancelSelectObjectSession();

      expect(h.engine.stores.canUndo.get()).toBe(true);
      h.engine.history.undo();
      expect(h.engine.document.getDocument()?.layers.map((layer) => layer.id)).toEqual(['source']);
      h.engine.history.redo();
      expect(h.engine.document.getDocument()?.layers[0]?.type).toBe(target);
      h.engine.lifecycle.dispose();
    }
  );

  it('keeps the session open across saves so a second save also succeeds without touching the source', async () => {
    const h = await createCommittingSamHarness();

    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => Promise.resolve())
    ).resolves.toMatchObject({
      status: 'committed',
    });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({ hasPreview: true, status: 'ready' });

    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => Promise.resolve())
    ).resolves.toMatchObject({
      status: 'committed',
    });

    expect(h.engine.document.getDocument()?.layers.map((layer) => layer.name)).toEqual([
      'Segmented Object',
      'Segmented Object',
      'source',
    ]);
    expect(h.engine.document.getDocument()?.layers[2]).toEqual(h.document.layers[0]);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({ hasPreview: true, status: 'ready' });
    expect(h.engine.stores.activeTool.get()).toBe('sam');
    h.engine.lifecycle.dispose();
  });

  it('saves the exact non-zero source rect origin and dimensions', async () => {
    const rect = { height: 30, width: 40, x: -11, y: 7 };
    const h = await createCommittingSamHarness(undefined, undefined, rect);

    const result = await getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => Promise.resolve());

    expect(result.status).toBe('committed');
    const created = h.engine.document.getDocument()?.layers[0];
    expect(created?.type).toBe('raster');
    if (created?.type === 'raster') {
      expect(created.source).toMatchObject({
        bitmap: { height: rect.height, width: rect.width },
        offset: { x: rect.x, y: rect.y },
      });
    }
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

  it.each(['input', 'reset', 'process'] as const)(
    'does not save or close when %s invalidates ownership during durability promotion',
    async (kind) => {
      const h = await createCommittingSamHarness();
      const durable = createDeferred<void>();
      const pending = getCanvasOperations(h.engine).saveSelectObjectSession('regional_guidance', () => durable.promise);

      await invalidateSelectObjectCommit(h.engine, kind);
      durable.resolve();

      await expect(pending).resolves.not.toMatchObject({ status: 'committed' });
      expect(h.engine.document.getDocument()).toEqual(h.document);
      expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
      h.engine.lifecycle.dispose();
    }
  );

  it.each([
    ['raster', 'input'],
    ['raster', 'reset'],
    ['raster', 'process'],
    ['control', 'input'],
    ['control', 'reset'],
    ['control', 'process'],
  ] as const)('does not commit %s when %s invalidates ownership during final decode', async (target, kind) => {
    const finalDecode = createDeferred<Blob>();
    let resolutions = 0;
    const h = await createCommittingSamHarness(() => {
      resolutions += 1;
      return resolutions === 1 ? Promise.resolve(new Blob()) : finalDecode.promise;
    });
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession(target, () => Promise.resolve());
    await vi.waitFor(() => expect(resolutions).toBe(2));

    await invalidateSelectObjectCommit(h.engine, kind);
    finalDecode.resolve(new Blob());

    await expect(pending).resolves.not.toMatchObject({ status: 'committed' });
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
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

  it('does not close a replacement session when the old selection save finishes authoritatively', async () => {
    const document = samDocument([samLayer('first'), samLayer('second', 30)]);
    const reactive = createReducerBackedStore(document);
    const { backend } = createOpaqueSamBackend();
    const engine = createCanvasEngine({
      backend: withSamPreviewBitmapDimensions(backend),
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
    let replaced = false;
    const unsubscribe = engine.stores.hasSelection.subscribe(() => {
      if (!replaced && engine.stores.hasSelection.get()) {
        replaced = true;
        getCanvasOperations(engine).startSelectObject('second');
      }
    });

    await expect(
      getCanvasOperations(engine).saveSelectObjectSession('selection', () => Promise.resolve())
    ).resolves.toEqual({
      status: 'selected',
    });

    expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({ error: null });
    unsubscribe();
    engine.lifecycle.dispose();
  });

  it.each(['apply', 'save'] as const)(
    'preserves a replacement started by the SAM subscriber when old %s succeeds',
    async (kind) => {
      const document = samDocument([samLayer('first'), samLayer('second', 30)]);
      const reactive = createReducerBackedStore(document);
      const { backend } = createOpaqueSamBackend();
      const engine = createCanvasEngine({
        backend: withSamPreviewBitmapDimensions(backend),
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
      let sawCommitting = false;
      let replaced = false;
      const unsubscribe = getCanvasOperations(engine).stores.samSession.subscribe(() => {
        const snapshot = getCanvasOperations(engine).stores.samSession.get();
        if (snapshot?.status === 'committing') {
          sawCommitting = true;
        } else if (sawCommitting && !replaced) {
          replaced = true;
          expect(getCanvasOperations(engine).startSelectObject('second')).toBe('started');
        }
      });

      const result =
        kind === 'apply'
          ? await getCanvasOperations(engine).applySelectObjectSession(() => Promise.resolve())
          : await getCanvasOperations(engine).saveSelectObjectSession('inpaint_mask', () => Promise.resolve());

      expect(result.status).toBe('committed');
      expect(replaced).toBe(true);
      expect(getCanvasOperations(engine).stores.samSession.get()).toMatchObject({ error: null });
      expect(engine.stores.activeTool.get()).toBe('sam');
      unsubscribe();
      engine.lifecycle.dispose();
    }
  );

  it('keeps the session and exact preview retryable when durability fails', async () => {
    const h = await createCommittingSamHarness();
    const previewInput = getCanvasOperations(h.engine).stores.samSession.get()?.input;

    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('raster', () =>
        Promise.reject(new Error('promotion failed'))
      )
    ).resolves.toEqual({ message: 'promotion failed', status: 'failed' });

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: { code: 'unknown', detail: 'promotion failed' },
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

  it('keeps the session and preview retryable when the structural save is rejected', async () => {
    const h = await createCommittingSamHarness();
    const previewInput = getCanvasOperations(h.engine).stores.samSession.get()?.input;
    h.dispatch.mockImplementationOnce(() => {
      throw new Error('commit rejected');
    });

    await expect(
      getCanvasOperations(h.engine).saveSelectObjectSession('inpaint_mask', () => Promise.resolve())
    ).resolves.toEqual({
      message: 'commit rejected',
      status: 'failed',
    });

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toMatchObject({
      error: { code: 'unknown', detail: 'commit rejected' },
      hasPreview: true,
    });
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.input).toEqual(previewInput);
    expect(h.engine.document.getDocument()).toEqual(h.document);
    h.engine.lifecycle.dispose();
  });

  it('does not commit a preview that becomes stale during durability promotion', async () => {
    const h = await createCommittingSamHarness();
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('control', () => durable.promise);
    h.dispatch({ id: 'source', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
    durable.resolve();

    await expect(pending).resolves.toEqual({ status: 'stale' });
    expect(h.engine.document.getDocument()?.layers).toHaveLength(1);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('does not treat a reentrant unrelated layer mutation as the owned Save insertion', async () => {
    const document = samDocument([samLayer('source')]);
    const reducer = createReducerBackedStore(document);
    let engine!: CanvasEngine;
    let reentered = false;
    let sessionDuringReentry: ReturnType<ReturnType<typeof getCanvasOperations>['stores']['samSession']['get']> = null;
    reducer.store.subscribe(() => {
      const current = reducer.store.getState().projects[0]?.canvas.document;
      if (!reentered && current?.layers.length === 2) {
        reentered = true;
        reducer.store.dispatch({ id: 'source', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
        sessionDuringReentry = getCanvasOperations(engine).stores.samSession.get();
      }
    });
    engine = createCanvasEngine({
      backend: withSamPreviewBitmapDimensions(createTestStubRasterBackend()),
      bitmapStore: createSpyBitmapStore(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      selectObjectDeps: {
        runGraph: () => Promise.resolve({ height: 100, imageName: 'sam-mask.png', origin: 'test', width: 100 }),
        uploadIntermediate: () => Promise.resolve({ height: 100, imageName: 'sam-source.png', width: 100 }),
      },
      store: reducer.store,
    });
    await engine.exports.exportLayerPixels('source');
    expect(getCanvasOperations(engine).startSelectObject('source')).toBe('started');
    getCanvasOperations(engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 5 }], type: 'visual' },
    });
    await expect(getCanvasOperations(engine).processSelectObjectSession()).resolves.toBe('published');

    await expect(
      getCanvasOperations(engine).saveSelectObjectSession('raster', () => Promise.resolve())
    ).resolves.toMatchObject({
      status: 'committed',
    });

    expect(reentered).toBe(true);
    expect(sessionDuringReentry).toBeNull();
    expect(getCanvasOperations(engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(engine.document.getDocument()?.layers.find((layer) => layer.id === 'source')?.opacity).toBe(0.5);
    engine.lifecycle.dispose();
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
      h.setDocument({ ...document, layers: [{ ...document.layers[0]!, opacity: 0.5 }] });
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

  it('does not commit when Cancel closes the operation during durability promotion', async () => {
    const h = await createCommittingSamHarness();
    const durable = createDeferred<void>();
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('regional_guidance', () => durable.promise);

    getCanvasOperations(h.engine).cancelSelectObjectSession();
    durable.resolve();

    await expect(pending).resolves.toEqual({ status: 'stale' });
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('aborts a raster save when Cancel lands during final image decoding', async () => {
    const finalDecode = createDeferred<Blob>();
    let resolutions = 0;
    const h = await createCommittingSamHarness(() => {
      resolutions += 1;
      return resolutions === 1 ? Promise.resolve(new Blob()) : finalDecode.promise;
    });
    const pending = getCanvasOperations(h.engine).saveSelectObjectSession('raster', () => Promise.resolve());
    await vi.waitFor(() => expect(resolutions).toBe(2));

    getCanvasOperations(h.engine).cancelSelectObjectSession();
    finalDecode.resolve(new Blob());

    await expect(pending).resolves.toEqual({ status: 'aborted' });
    expect(h.engine.document.getDocument()).toEqual(h.document);
    expect(h.engine.stores.canUndo.get()).toBe(false);
    h.engine.lifecycle.dispose();
  });

  it('processes a sole bbox near-edge point after canonicalizing it to the last bbox pixel', async () => {
    const runGraph = vi.fn(() =>
      Promise.resolve({ height: 100, imageName: 'sam-mask.png', origin: 'test', width: 100 })
    );
    const h = await createSamHarness({ runGraph });
    getCanvasOperations(h.engine).startSelectObject('source');

    h.overlay.fire('pointerdown', pointerAt(99.8, 99.7));
    h.overlay.fire('pointerup', pointerAt(99.8, 99.7, { buttons: 0 }));

    expect(getCanvasOperations(h.engine).stores.samSession.get()?.input.includePoints).toEqual([{ x: 99, y: 99 }]);
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');
    expect(runGraph).toHaveBeenCalledOnce();
    h.engine.lifecycle.dispose();
  });

  it('uses the half-open generation bbox as the only point domain', async () => {
    const h = await createSamHarness();
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');

    h.overlay.fire('pointerdown', pointerAt(100, 50));
    h.overlay.fire('pointerup', pointerAt(100, 50, { buttons: 0 }));
    h.overlay.fire('pointerdown', pointerAt(-0.01, 50));
    h.overlay.fire('pointerup', pointerAt(-0.01, 50, { buttons: 0 }));
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.input.includePoints).toEqual([]);

    h.overlay.fire('pointerdown', pointerAt(99.99, 99.99));
    h.overlay.fire('pointerup', pointerAt(99.99, 99.99, { buttons: 0 }));
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.input.includePoints).toEqual([{ x: 99, y: 99 }]);
    h.engine.lifecycle.dispose();
  });

  it('keeps viewport changes current but invalidates on cache clears', async () => {
    const h = await createSamHarness();
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');

    h.engine.viewport.getViewport().panBy({ x: 20, y: -10 });
    h.engine.viewport.getViewport().zoomAtPoint(2, { x: 50, y: 50 });
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object' },
      status: 'active',
    });

    await h.engine.diagnostics.clearCaches();
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    h.engine.lifecycle.dispose();
  });

  it('survives bbox moves, layer reorders, and other-layer edits but cancels on a source-layer edit', async () => {
    const h = await createSamHarness({ layers: [samLayer('source'), samLayer('other', 30)] });
    expect(getCanvasOperations(h.engine).startSelectObject('source')).toBe('started');

    const document = h.engine.document.getDocument()!;
    h.setDocument({ ...document, bbox: { height: 80, width: 90, x: 3, y: 4 } });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();

    const reordered = h.engine.document.getDocument()!;
    h.setDocument({ ...reordered, layers: [...reordered.layers].reverse() });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();

    const withOtherEdited = h.engine.document.getDocument()!;
    h.setDocument({
      ...withOtherEdited,
      layers: withOtherEdited.layers.map((layer) => (layer.id === 'other' ? { ...layer, opacity: 0.5 } : layer)),
    });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({
      identity: { kind: 'select-object', layerId: 'source' },
      status: 'active',
    });

    const withSourceEdited = h.engine.document.getDocument()!;
    h.setDocument({
      ...withSourceEdited,
      layers: withSourceEdited.layers.map((layer) => (layer.id === 'source' ? { ...layer, opacity: 0.25 } : layer)),
    });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(h.engine.stores.activeTool.get()).toBe('view');
    h.engine.lifecycle.dispose();
  });

  it('does not start from a stale cache whose replacement source is still decoding', async () => {
    const nextImage = createDeferred<Blob>();
    const outside = {
      ...samLayer('moving-source'),
      source: {
        bitmap: { height: 10, imageName: 'outside.png', width: 10 },
        offset: { x: 150, y: 0 },
        type: 'paint' as const,
      },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    };
    const h = await createSamHarness({
      imageResolver: (imageName) =>
        imageName === 'inside.png' ? nextImage.promise : Promise.resolve(new Blob(['outside'])),
      layers: [outside],
    });
    const document = h.engine.document.getDocument()!;
    h.setDocument({
      ...document,
      layers: [
        {
          ...outside,
          source: { image: { height: 10, imageName: 'inside.png', width: 10 }, type: 'image' },
        },
      ],
    });

    expect(getCanvasOperations(h.engine).startSelectObject('moving-source')).toBe('not-ready');
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });

    nextImage.resolve(new Blob(['inside']));
    await expect(h.engine.exports.exportLayerPixels('moving-source')).resolves.toMatchObject({ status: 'ok' });
    expect(getCanvasOperations(h.engine).startSelectObject('moving-source')).toBe('started');
    h.engine.lifecycle.dispose();
  });

  it('refuses disabled and authoritatively empty source layers without rasterizing', async () => {
    const imageResolver = vi.fn(() => Promise.reject(new Error('disabled layer must not rasterize')));
    const empty = {
      ...samLayer('empty'),
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' as const },
    };
    const disabled = {
      ...samLayer('disabled'),
      isEnabled: false,
      source: { image: { height: 10, imageName: 'disabled.png', width: 10 }, type: 'image' as const },
    };
    const h = await createSamHarness({ imageResolver, layers: [disabled, empty] });

    expect(getCanvasOperations(h.engine).startSelectObject('disabled')).toBe('disabled');
    expect(getCanvasOperations(h.engine).startSelectObject('empty')).toBe('not-ready');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(imageResolver).not.toHaveBeenCalled();
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
    expect(h.engine.document.getDocument()!.layers.every((layer) => layer.isEnabled)).toBe(true);
    h.engine.lifecycle.dispose();
  });

  it('isolates the transformed source layer with display semantics while excluding other layers', async () => {
    const backend = createRecordingRasterBackend();
    backend.createImageBitmap = vi.fn(() => Promise.resolve(recordingBitmap('decoded', 20, 15)));
    // Paint bitmap 10x10 at offset (-4, 6), scaled 2x1.5 and translated (12, 8):
    // baked world rect {4, 17, 20, 15}.
    const raster: CanvasLayerContract = {
      ...samLayer('adjusted-raster'),
      adjustments: { brightness: 0.2, contrast: -0.1, saturation: 0.3 },
      blendMode: 'multiply',
      opacity: 0.4,
      source: {
        bitmap: { height: 10, imageName: 'sparse.png', width: 10 },
        offset: { x: -4, y: 6 },
        type: 'paint',
      },
      transform: { rotation: 0, scaleX: 2, scaleY: 1.5, x: 12, y: 8 },
    };
    const control: CanvasLayerContract = {
      adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'screen',
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control',
      opacity: 0.65,
      source: { image: { height: 12, imageName: 'control.png', width: 14 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 30, y: 20 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const mask: CanvasInpaintMaskLayerContract = {
      ...samLayer('mask'),
      mask: { bitmap: { height: 20, imageName: 'mask.png', width: 20 }, fill: { color: '#fff', style: 'solid' } },
      type: 'inpaint_mask',
    };
    const h = await createSamHarness({
      backend,
      layers: [mask, control, raster],
      runGraph: () => Promise.resolve({ height: 15, imageName: 'sam-mask.png', origin: 'test', width: 20 }),
      uploadIntermediate: () => Promise.resolve({ height: 15, imageName: 'sam-source.png', width: 20 }),
    });
    h.engine.stores.checkerboard.set(false);
    expect(getCanvasOperations(h.engine).startSelectObject('adjusted-raster')).toBe('started');
    expect(getCanvasOperations(h.engine).stores.samSession.get()?.sourceRect).toEqual({
      height: 15,
      width: 20,
      x: 4,
      y: 17,
    });
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 5, y: 18 }], type: 'visual' },
    });
    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');
    h.screen.surface.callLog.length = 0;
    h.raf.flush();

    const draws = h.screen.surface.callLog.filter((entry) => entry.op === 'drawImage');
    expect(draws).toHaveLength(1);
    expect(h.screen.surface.callLog).toEqual(
      expect.arrayContaining([
        { args: ['globalAlpha', 0.4], op: 'set' },
        { args: ['globalCompositeOperation', 'multiply'], op: 'set' },
        { args: [4, 17, 20, 15], op: 'rect' },
      ])
    );
    expect(h.screen.surface.callLog).not.toContainEqual({ args: ['globalAlpha', 0.65], op: 'set' });
    expect(h.screen.surface.callLog).not.toContainEqual({ args: ['globalCompositeOperation', 'screen'], op: 'set' });
    expect(adjustedSurfaceCacheGets).toContain('adjusted-raster');
    h.engine.lifecycle.dispose();
  });

  it('encodes exactly the baked source layer surface, excluding other layers and previews', async () => {
    const backend = createRecordingRasterBackend();
    const encoded: StubRasterSurface[] = [];
    backend.encodeSurface = vi.fn((surface) => {
      encoded.push(surface as StubRasterSurface);
      return Promise.resolve(new Blob(['baked-layer'], { type: 'image/png' }));
    });
    backend.createImageBitmap = vi.fn(() => Promise.resolve(recordingBitmap('decoded', 12, 24)));
    // Paint bitmap 6x8 at offset (-4, 6), scaled 2x3 and translated (20, 30):
    // baked world rect {12, 48, 12, 24}.
    const raster: CanvasLayerContract = {
      ...samLayer('raster'),
      source: {
        bitmap: { height: 8, imageName: 'raster.png', width: 6 },
        offset: { x: -4, y: 6 },
        type: 'paint',
      },
      transform: { rotation: 0, scaleX: 2, scaleY: 3, x: 20, y: 30 },
    };
    const control: CanvasLayerContract = {
      adapter: { beginEndStepPct: [0, 0.75], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'normal',
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control',
      opacity: 1,
      source: { image: { height: 10, imageName: 'control.png', width: 10 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 40, y: 50 },
      type: 'control',
      withTransparencyEffect: false,
    };
    const mask: CanvasInpaintMaskLayerContract = {
      ...samLayer('mask'),
      mask: { bitmap: { height: 10, imageName: 'mask.png', width: 10 }, fill: { color: '#fff', style: 'solid' } },
      type: 'inpaint_mask',
    };
    const h = await createSamHarness({
      backend,
      layers: [mask, control, raster],
      runGraph: () => Promise.resolve({ height: 24, imageName: 'sam-mask.png', origin: 'test', width: 12 }),
      uploadIntermediate: () => Promise.resolve({ height: 24, imageName: 'sam-source.png', width: 12 }),
    });
    h.setDocument({ ...h.engine.document.getDocument()!, bbox: { height: 70, width: 80, x: 10, y: 15 } });
    const rasterExport = await h.engine.exports.exportLayerPixels('raster');
    const controlExport = await h.engine.exports.exportLayerPixels('control');
    if (rasterExport.status !== 'ok' || controlExport.status !== 'ok') {
      throw new Error('expected authoritative layer caches');
    }
    await expect(
      h.engine.previews.setGuardedFilterPreview(
        'raster',
        { imageName: 'filter-preview.png', rect: rasterExport.rect },
        rasterExport.guard
      )
    ).resolves.toBe('shown');
    h.engine.previews.setStagedPreview({ imageName: 'staged.png' });
    await flushMicrotasks();
    expect(getCanvasOperations(h.engine).startSelectObject('raster')).toBe('started');
    getCanvasOperations(h.engine).updateSelectObjectSession({
      input: { bbox: null, excludePoints: [], includePoints: [{ x: 15, y: 50 }], type: 'visual' },
    });

    await expect(getCanvasOperations(h.engine).processSelectObjectSession()).resolves.toBe('published');

    const baked = encoded.find((surface) => surface.width === 12 && surface.height === 24);
    expect(baked).toBeDefined();
    const log = baked!.callLog;
    const draws = log.filter((entry) => entry.op === 'drawImage');
    expect(draws).toHaveLength(1);
    expect(backend.drawSourcesFor(baked!)).toEqual([backend.surfaceId(rasterExport.surface as StubRasterSurface)]);
    expect(draws.map((entry) => entry.args.slice(1))).toEqual([[-4, 6]]);
    expect(log.filter((entry) => entry.op === 'setTransform').map((entry) => entry.args)).toEqual(
      expect.arrayContaining([[2, 0, 0, 3, 8, -18]])
    );
    h.engine.lifecycle.dispose();
  });

  it('tears down the session when its captured source contract changes', async () => {
    const h = await createSamHarness();
    getCanvasOperations(h.engine).startSelectObject('source');

    h.setDocument({
      ...h.engine.document.getDocument()!,
      layers: [{ ...h.engine.document.getDocument()!.layers[0]!, opacity: 0.5 }],
    });

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(h.engine.stores.activeTool.get()).toBe('view');
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

  it('lets Escape cancel a gesture before a later Escape cancels the operation', async () => {
    const h = await createSamHarness();
    getCanvasOperations(h.engine).startSelectObject('source');
    h.overlay.fire('pointerdown', pointerAt(5, 5));
    h.overlay.fire('pointermove', pointerAt(15, 15));
    h.overlay.fire('pointercancel', pointerAt(15, 15));

    h.engine.tools.handleEscapePriority({ gestureWasActive: true });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).not.toBeNull();

    h.engine.tools.handleEscapePriority({ gestureWasActive: false });
    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    h.engine.lifecycle.dispose();
  });

  it('keeps the Select Object operation active while Space temporarily switches to view', async () => {
    const h = await createSamHarness();
    getCanvasOperations(h.engine).startSelectObject('source');
    h.overlay.fire('pointerenter', {});

    h.fireKey('keydown', 'Space', ' ');

    expect(h.engine.stores.activeTool.get()).toBe('view');
    expect(getCanvasOperations(h.engine).controller.getSnapshot()).toMatchObject({ status: 'active' });
    h.fireKey('keyup', 'Space', ' ');
    expect(h.engine.stores.activeTool.get()).toBe('sam');
    h.engine.lifecycle.dispose();
  });

  it('does not restore sessionless sam when Escape closes the session during a Space hold', async () => {
    const h = await createSamHarness();
    getCanvasOperations(h.engine).startSelectObject('source');
    h.overlay.fire('pointerenter', {});
    h.fireKey('keydown', 'Space', ' ');
    expect(h.engine.stores.activeTool.get()).toBe('view');

    h.fireKey('keydown', 'Escape', 'Escape');
    h.fireKey('keyup', 'Space', ' ');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(h.engine.stores.activeTool.get()).toBe('view');
    h.engine.lifecycle.dispose();
  });

  it('does not restore sessionless sam when source invalidation closes the session during a Space hold', async () => {
    const h = await createSamHarness();
    getCanvasOperations(h.engine).startSelectObject('source');
    h.overlay.fire('pointerenter', {});
    h.fireKey('keydown', 'Space', ' ');

    h.setDocument({
      ...h.engine.document.getDocument()!,
      layers: [{ ...h.engine.document.getDocument()!.layers[0]!, opacity: 0.5 }],
    });
    h.fireKey('keyup', 'Space', ' ');

    expect(getCanvasOperations(h.engine).stores.samSession.get()).toBeNull();
    expect(h.engine.stores.activeTool.get()).toBe('view');
    h.engine.lifecycle.dispose();
  });
});
