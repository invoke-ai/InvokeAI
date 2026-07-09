import type * as AdjustedSurfaceCacheModule from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import type {
  RasterCallLogEntry,
  StubRasterBackend,
  StubRasterSurface,
} from '@workbench/canvas-engine/render/raster.testStub';
import type { ToolId } from '@workbench/canvas-engine/types';
import type {
  CanvasDocumentContractV2,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
  CanvasStateContractV2,
  WorkbenchState,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/render/compositor';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState';
import { afterEach, describe, expect, it, type Mock, vi } from 'vitest';

import type { BitmapStore } from './document/bitmapStore';
import type { EngineStore } from './engine';
import type { StrokeCommittedEvent } from './tools/tool';

import { createBitmapStore } from './document/bitmapStore';
import { mergeDownMatrix } from './document/mergeDown';
import { createCanvasEngine } from './engine';

// Records every layer id the engine tells its adjusted-surface cache to drop, so
// the layer-removal cleanup wiring (Task 39, finding 2) can be asserted without
// exposing the cache. The factory wraps the real implementation, preserving all
// behaviour and only spying on `delete`.
const adjustedSurfaceCacheDeletes = vi.hoisted(() => [] as string[]);

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
          cache.delete(layerId);
        },
      };
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

const recordingBitmap = (id: string): ImageBitmap =>
  ({ __recordingId: `bitmap-${id}`, close: vi.fn(), height: 10, width: 10 }) as unknown as ImageBitmap;

/** Flushes pending microtasks (promise chains) without depending on fake timers. */
const flushMicrotasks = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('createCanvasEngine', () => {
  it('mirrors the reducer-owned document on creation', () => {
    const { doc, engine } = createEngine();
    expect(engine.getDocument()).toBe(doc);
    engine.dispose();
  });

  it('exposes an initial viewport and stores', () => {
    const { engine } = createEngine();
    expect(engine.getViewport().getZoom()).toBe(1);
    expect(engine.stores.activeTool.get()).toBe('view');
    expect(engine.stores.viewportReady.get()).toBe(false);
    engine.dispose();
  });

  it('exportLayerPixels rasterizes a visible layer and returns its cache surface plus content rect', async () => {
    const { engine } = createEngine();

    const result = await engine.exportLayerPixels('a');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.rect).toEqual({ height: 10, width: 10, x: 0, y: 0 });
      expect(result.surface.width).toBe(10);
      expect(result.surface.height).toBe(10);
      expect((result.surface as StubRasterSurface).callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    }
    engine.dispose();
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

    expect(await engine.exportLayerPixels('hidden')).toEqual({ status: 'disabled' });
    expect((await engine.exportLayerPixels('hidden', { includeDisabled: true })).status).toBe('ok');
    engine.dispose();
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

    engine.attach(screen.element, overlay.element);
    raf.flush();

    const exported = engine.exportLayerPixels('a');
    expect(imageResolver).toHaveBeenCalledTimes(1);

    pendingResolve.resolve(new Blob());
    expect((await exported).status).toBe('ok');
    engine.dispose();
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

    const exportA = engine.exportLayerPixels('L');
    setDocument({ ...document, layers: [rasterLayer('L', { imageName: 'B' })] });
    const exportB = engine.exportLayerPixels('L');

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
    engine.dispose();
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

    const oldExport = engine.exportLayerPixels('L');
    setDocument({ ...document, layers: [rasterLayer('L', { imageName: 'B' })] }, 1);
    const newExport = engine.exportLayerPixels('L');

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
    engine.dispose();
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

    const [a, b] = await Promise.all([engine.exportLayerPixels('L'), engine.exportLayerPixels('L')]);

    expect(imageResolver).toHaveBeenCalledTimes(1);
    expect(a.status).toBe('ok');
    expect(b.status).toBe('ok');
    engine.dispose();
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

    const exported = engine.exportLayerPixels('L');
    setDocument({ ...document, layers: [{ ...layer, isEnabled: false }] });
    pending.resolve(new Blob());

    expect(await exported).toEqual({ status: 'disabled' });
    engine.dispose();
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

    engine.attach(screen.element, overlay.element);
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
    expect(await engine.exportLayerPixels('empty')).toEqual({ status: 'empty' });
    engine.dispose();
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

    expect(await engine.exportLayerPixels('shape')).toEqual({ status: 'not-ready' });
    expect((await engine.exportLayerPixels('shape')).status).toBe('ok');
    engine.dispose();
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

    const result = await engine.exportBakedLayerPixels('a');

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
    engine.dispose();
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

    const raw = await engine.exportLayerPixels('a');
    const baked = await engine.exportBakedLayerPixels('a');

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
    engine.dispose();
  });

  it('exportBakedLayerBlob encodes the baked layer surface as PNG', async () => {
    const { engine } = createEngine();

    const result = await engine.exportBakedLayerBlob('a');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.rect).toEqual({ height: 10, width: 10, x: 0, y: 0 });
      expect(result.blob.type).toBe('image/png');
      expect(await result.blob.text()).toBe('stub-surface-10x10');
    }
    engine.dispose();
  });

  it('returns a current LayerExportGuard from local, baked-pixel, and baked-blob exports', async () => {
    const { doc, engine } = createEngine();

    const local = await engine.exportLayerPixels('a');
    const baked = await engine.exportBakedLayerPixels('a');
    const blob = await engine.exportBakedLayerBlob('a');

    expect(local.status).toBe('ok');
    expect(baked.status).toBe('ok');
    expect(blob.status).toBe('ok');
    if (local.status !== 'ok' || baked.status !== 'ok' || blob.status !== 'ok') {
      throw new Error('expected successful exports');
    }
    for (const result of [local, baked, blob]) {
      expect(result.guard).toMatchObject({ layer: doc.layers[0], layerId: 'a', projectId: 'p1' });
      expect(engine.isLayerExportGuardCurrent(result.guard)).toBe(true);
    }
    engine.dispose();
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
    const exported = await engine.exportLayerPixels('a');
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }

    setDocument({ ...document, layers: [{ ...document.layers[0]!, opacity: 0.5 }] });

    expect(engine.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    engine.dispose();
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
    const exported = await engine.exportLayerPixels('a');
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }

    setDocument({ ...document, layers: document.layers }, 1);

    expect(engine.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    engine.dispose();
  });

  it('invalidates a LayerExportGuard when another project becomes active', async () => {
    const document = makeDoc();
    const { setActiveProjectId, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    const exported = await engine.exportLayerPixels('a');
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }

    setActiveProjectId('p2');

    expect(engine.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    engine.dispose();
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

    const pending = engine.exportBakedLayerBlob('a');
    await vi.waitFor(() => expect(encodeSurface).toHaveBeenCalledOnce());
    setDocument({ ...document, layers: [{ ...document.layers[0]!, opacity: 0.5 }] });
    encoded.resolve(new Blob(['stale'], { type: 'image/png' }));

    expect(await pending).toEqual({ status: 'not-ready' });
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const exported = await engine.exportLayerPixels(layer.id);
    expect(exported.status).toBe('ok');
    if (exported.status !== 'ok') {
      throw new Error('expected successful export');
    }

    engine.setTool('brush');
    overlay.fire('pointerdown', pointerAt(5, 5));
    overlay.fire('pointermove', pointerAt(10, 10));
    overlay.fire('pointerup', pointerAt(10, 10, { buttons: 0 }));

    expect(engine.isLayerExportGuardCurrent(exported.guard)).toBe(false);
    engine.dispose();
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
    const beforeContract = structuredClone(engine.getDocument()!.layers[0]!);
    const beforeExport = await engine.exportLayerPixels('a');
    expect(beforeExport.status).toBe('ok');
    if (beforeExport.status !== 'ok') {
      throw new Error('expected original cache pixels');
    }
    const thumbnailListener = vi.fn();
    engine.stores.thumbnailVersion.subscribeKey('a', thumbnailListener);
    bitmapStore.markLayerDirty.mockClear();

    expect(await engine.cropLayerToBbox('a')).toEqual({ status: 'cropped' });

    const afterContract = engine.getDocument()!.layers[0]!;
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
    const forwardExport = await engine.exportLayerPixels('a');
    expect(forwardExport.status).toBe('ok');
    if (forwardExport.status !== 'ok') {
      throw new Error('expected cropped cache pixels');
    }
    expect(forwardExport.rect).toEqual({ height: 5, width: 7, x: 8, y: 5 });
    const forwardSources = backend.drawSourcesFor(forwardExport.surface as StubRasterSurface);

    engine.undo();
    expect(engine.getDocument()!.layers[0]).toEqual(beforeContract);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledTimes(2);
    expect(thumbnailListener).toHaveBeenCalledTimes(2);
    const undoExport = await engine.exportLayerPixels('a');
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

    engine.redo();
    expect(engine.getDocument()!.layers[0]).toEqual(afterContract);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledTimes(3);
    expect(thumbnailListener).toHaveBeenCalledTimes(3);
    const redoExport = await engine.exportLayerPixels('a');
    expect(redoExport.status).toBe('ok');
    if (redoExport.status !== 'ok') {
      throw new Error('expected redone cache pixels');
    }
    expect(backend.drawSourcesFor(redoExport.surface as StubRasterSurface)).toEqual(forwardSources);
    engine.dispose();
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

    expect(await engine.cropLayerToBbox('control')).toEqual({ status: 'cropped' });
    const cropped = engine.getDocument()!.layers[0];
    expect(cropped).toMatchObject({
      adapter: control.adapter,
      source: { bitmap: null, offset: { x: 2, y: 3 }, type: 'paint' },
      type: 'control',
      withTransparencyEffect: true,
    });
    expect(cropped && 'filter' in cropped).toBe(false);
    engine.dispose();
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

    expect(await engine.cropLayerToBbox('mask')).toEqual({ status: 'cropped' });
    expect(engine.getDocument()!.layers[0]).toMatchObject({
      denoiseLimit: 0.45,
      mask: { bitmap: null, fill: layer.mask.fill, offset: { x: 2, y: 3 } },
      noiseLevel: 0.25,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    });
    engine.dispose();
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

    expect(await engine.cropLayerToBbox('region')).toEqual({ status: 'cropped' });
    expect(engine.getDocument()!.layers[0]).toMatchObject({
      autoNegative: false,
      mask: { bitmap: null, fill: layer.mask.fill, offset: { x: 2, y: 3 } },
      negativePrompt: 'negative',
      positivePrompt: 'positive',
      referenceImages: layer.referenceImages,
      type: 'regional_guidance',
    });
    engine.dispose();
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
    const before = engine.getDocument();

    expect(await engine.cropLayerToBbox('a')).toEqual({ status: 'empty' });
    expect(engine.getDocument()).toBe(before);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    engine.dispose();
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

    expect(await engine.cropLayerToBbox('missing')).toEqual({ status: 'missing' });
    expect(await engine.cropLayerToBbox('locked')).toEqual({ status: 'locked' });
    expect(await engine.cropLayerToBbox('polygon')).toEqual({ status: 'unsupported' });
    expect(await engine.cropLayerToBbox('pending')).toEqual({ status: 'not-ready' });
    engine.dispose();
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

    const crop = engine.cropLayerToBbox('a');
    setDocument({ ...document, layers: [rasterLayer('a', { imageName: 'B' })] });
    pending.resolve(new Blob());

    expect(await crop).toEqual({ status: 'not-ready' });
    expect(store.dispatch).not.toHaveBeenCalled();
    expect(bitmapStore.markLayerDirty).not.toHaveBeenCalled();
    engine.dispose();
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
    busyEngine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    busyEngine.setTool('brush');
    overlay.fire('pointerdown', pointerAt(2, 2));
    expect(await busyEngine.cropLayerToBbox('paint')).toEqual({ status: 'busy' });
    overlay.fire('pointerup', pointerAt(2, 2, { buttons: 0 }));
    busyEngine.dispose();
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
    expect(await failedEngine.cropLayerToBbox('a')).toEqual({
      message: 'crop allocation failed',
      status: 'failed',
    });
    failedEngine.dispose();
  });

  it('copyLayerToRaster does not mutate another project after its export resolves', async () => {
    const pending = createDeferred<Blob>();
    const document = makeDoc();
    const { setActiveProjectId, store } = createReactiveStore(document);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => pending.promise,
      projectId: 'p1',
      store,
    });

    const copy = engine.copyLayerToRaster('a');
    setActiveProjectId('p2');
    pending.resolve(new Blob());

    expect(await copy).toBeNull();
    expect(store.dispatch).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('copyLayerToRaster adds a baked paint copy directly above the source layer', async () => {
    const doc = { ...makeDoc(), layers: [rasterLayer('top'), rasterLayer('a')] };
    const { store } = createFakeStore(doc);
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
      projectId: 'p1',
      store,
    });

    const newId = await engine.copyLayerToRaster('a');

    expect(newId).toMatch(/^layer-/);
    const add = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .find((action) => action.type === 'addCanvasLayer');
    expect(add).toBeDefined();
    if (add?.type === 'addCanvasLayer' && add.layer.type === 'raster') {
      expect(add.index).toBe(1);
      expect(add.layer.id).toBe(newId);
      expect(add.layer.name).toBe('a copy');
      expect(add.layer.source).toEqual({ bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' });
      expect(add.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    } else {
      throw new Error('expected addCanvasLayer with raster copy');
    }
    expect(
      surfaces.some(
        (surface) =>
          surface.width === 10 && surface.height === 10 && surface.callLog.some((entry) => entry.op === 'drawImage')
      )
    ).toBe(true);
    engine.dispose();
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

    expect(await engine.copyLayerToRaster('empty')).toBeNull();
    expect(dispatch).not.toHaveBeenCalled();
    engine.dispose();
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

    const newId = await engine.copyLayerToRaster('mask');
    expect(engine.getDocument()!.layers.some((layer) => layer.id === newId)).toBe(true);

    engine.undo();
    expect(engine.getDocument()!.layers.some((layer) => layer.id === newId)).toBe(false);
    engine.redo();
    expect(engine.getDocument()!.layers.some((layer) => layer.id === newId)).toBe(true);
    engine.dispose();
  });

  it('setTool switches the active tool and updates the store', () => {
    const { engine } = createEngine();
    const listener = vi.fn();
    engine.stores.activeTool.subscribe(listener);

    engine.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');
    expect(listener).toHaveBeenCalledTimes(1);

    // Setting the same tool again is a no-op.
    engine.setTool('brush');
    expect(listener).toHaveBeenCalledTimes(1);
    engine.dispose();
  });

  it('interaction lock forces view and refuses non-view tools until unlocked', () => {
    const { engine } = createEngine();

    engine.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');

    engine.setInteractionLocked(true);
    expect(engine.stores.activeTool.get()).toBe('view');

    engine.setTool('bbox');
    expect(engine.stores.activeTool.get()).toBe('view');

    engine.setTool('colorPicker');
    expect(engine.stores.activeTool.get()).toBe('view');

    engine.setInteractionLocked(false);
    engine.setTool('bbox');
    expect(engine.stores.activeTool.get()).toBe('bbox');

    engine.dispose();
  });

  it('registers the brush and eraser tools', () => {
    const { engine } = createEngine();
    engine.setTool('brush');
    expect(engine.stores.activeTool.get()).toBe('brush');
    engine.setTool('eraser');
    expect(engine.stores.activeTool.get()).toBe('eraser');
    engine.dispose();
  });

  it('onStrokeCommitted returns an unsubscribe and never fires without a stroke', () => {
    const { engine } = createEngine();
    const listener = vi.fn();
    const unsubscribe = engine.onStrokeCommitted(listener);
    expect(typeof unsubscribe).toBe('function');
    unsubscribe();
    expect(listener).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('resize records the viewport size and clamps the device-pixel ratio', () => {
    const { engine } = createEngine();
    engine.resize(800, 600, 4);
    expect(engine.getViewport().getViewportSize()).toEqual({ height: 600, width: 800 });
    expect(engine.getViewport().getDpr()).toBe(2);
    engine.dispose();
  });

  it('dispose removes the store subscription', () => {
    const { engine, unsubscribe } = createEngine();
    expect(unsubscribe).not.toHaveBeenCalled();
    engine.dispose();
    expect(unsubscribe).toHaveBeenCalledTimes(1);
  });

  it('dispose is idempotent', () => {
    const { engine, unsubscribe } = createEngine();
    engine.dispose();
    engine.dispose();
    expect(unsubscribe).toHaveBeenCalledTimes(1);
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
    engine.attach(screen.element, overlay.element);

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

    engine.dispose();
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
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);
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

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);

    // Run the initial attach frame so the surface holds a composited frame, then
    // isolate what the resize alone draws.
    raf.flush();
    screen.surface.callLog.length = 0;

    // A single resize event (as a ResizeObserver fires mid panel-drag). Sizing a
    // `<canvas>` backing store CLEARS it; the fix recomposites synchronously so the
    // cleared surface is repainted before the browser paints (no blank frame).
    // No `raf.flush()` follows — so any draw seen here happened IN-TASK.
    engine.resize(800, 600, 1);

    // Backing store was resized in the same call...
    expect(screen.element.width).toBe(800);
    expect(screen.element.height).toBe(600);
    // ...and the composite ran SYNCHRONOUSLY (drew into the surface without a rAF
    // flush). `compositeDocument` always clears the target, so a `clearRect` here
    // proves the recomposite happened in-task rather than being deferred to rAF.
    const composited = screen.surface.callLog.some((entry) => entry.op === 'clearRect');
    expect(composited).toBe(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    screen.surface.callLog.length = 0;

    engine.resize(800, 600, 1);

    // The synchronous composite ran...
    expect(screen.surface.callLog.some((entry) => entry.op === 'clearRect')).toBe(true);
    // ...and setViewportSize's viewport-subscription `{ view: true }` invalidate was
    // suppressed, so NO rAF frame is queued to recomposite the identical content.
    expect(raf.pendingCount()).toBe(0);

    // Draining any frame anyway must not produce a second composite.
    screen.surface.callLog.length = 0;
    raf.flush();
    expect(screen.surface.callLog.some((entry) => entry.op === 'clearRect')).toBe(false);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    screen.surface.callLog.length = 0;

    engine.resize(320, 240, 2);

    // The screen (document composite) surface — not just the overlay — was redrawn.
    expect(screen.surface.callLog.some((entry) => entry.op === 'clearRect')).toBe(true);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);

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
    expect(resolver).toHaveBeenNthCalledWith(1, 'a');

    // An edit lands mid-flight: same layer id, new object reference, new source.
    setDocument({ ...doc, layers: [rasterLayer('a', { imageName: 'a-v2' })] });

    // A frame runs while the first rasterize is still in flight. The source no
    // longer matches that job, so a fresh isolated job starts immediately.
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'a-v2');

    // The newer rasterize wins and publishes while the old decode is pending.
    deferreds.get('a-v2')!.resolve(new Blob());
    await flushMicrotasks();
    raf.flush();
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    expect(engine.stores.thumbnailVersion.get('a')).toBe(1);

    // The older decode resolves afterwards. It may finish its isolated scratch
    // draw, but it must neither publish nor notify subscribers.
    deferreds.get('a')!.resolve(new Blob());
    await flushMicrotasks();
    raf.flush();
    expect(thumbnailListener).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenCalledTimes(2);

    unsubscribe();
    engine.dispose();
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

/** A minimal in-memory bitmap store: records dirty-marks, never touches the network. */
const createSpyBitmapStore = (): BitmapStore & {
  markLayerDirty: Mock<(layerId: string) => void>;
  reset: Mock<() => void>;
} => ({
  discardLayer: vi.fn(),
  dispose: vi.fn(),
  flushPendingUploads: vi.fn(() => Promise.resolve()),
  isSelfEcho: () => false,
  markLayerDirty: vi.fn<(layerId: string) => void>(),
  reset: vi.fn<() => void>(),
});

/** A fake canvas that also lets a test fire pointer events at the engine's listeners. */
const createInputCanvas = (
  width = 100,
  height = 100
): { element: HTMLCanvasElement; fire: (type: string, event: Partial<PointerEvent>) => void } => {
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
  return { element, fire };
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
    engine.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

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
    engine.undo();
    const undoPut = putImageDataCalls(surfaces).find((call) => call.image === event.beforeImageData);
    expect(undoPut).toBeDefined();
    expect(undoPut!.x).toBe(0);
    expect(undoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    expect(bitmapStore.markLayerDirty.mock.calls.length).toBeGreaterThan(dirtyAfterStroke);

    // Redo: putImageData(after) restores the post-stroke pixels.
    engine.redo();
    const redoPut = putImageDataCalls(surfaces).find((call) => call.image === event.afterImageData);
    expect(redoPut).toBeDefined();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.dispose();
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
    engine.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

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
    engine.undo();
    const undoPut = putImageDataCalls(surfaces).find((call) => call.image === event.beforeImageData);
    expect(undoPut).toBeDefined();
    expect(undoPut!.x).toBe(0);
    expect(undoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    // Redo writes the EXACT post-stroke ImageData back — a lossless round-trip.
    engine.redo();
    const redoPut = putImageDataCalls(surfaces).find((call) => call.image === event.afterImageData);
    expect(redoPut).toBeDefined();
    expect(redoPut!.x).toBe(0);
    expect(redoPut!.y).toBe(0);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    expect(engine.stores.canUndo.get()).toBe(true);

    // A dims change triggers onDocumentReplaced → history.clear().
    const replaced = { ...paintDoc(), height: 200, width: 200 };
    setDocument(replaced);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');
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

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);

    // Initial rasterize of the v1 source.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'src-v1');

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
    expect(resolver).toHaveBeenNthCalledWith(2, 'src-v2');

    engine.dispose();
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
    engine.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

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
    engine.undo();
    engine.redo();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(putImageDataCalls(surfaces).length).toBe(putsMidGesture);
    // In particular, the first stroke's before pixels were never injected.
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.beforeImageData)).toBe(false);

    // End the gesture; now undo works and writes the newest stroke's before pixels.
    overlay.fire('pointerup', pointerAt(60, 60, { buttons: 0 }));
    expect(strokes).toHaveLength(2);
    engine.undo();
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[1]!.beforeImageData)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.dispose();
  });
});

// ---- commitStructural: UI-initiated structural edits on the canvas history ----

describe('commitStructural', () => {
  const forward: WorkbenchAction = { id: 'a', type: 'setCanvasSelectedLayer' };
  const inverse: WorkbenchAction = { id: null, type: 'setCanvasSelectedLayer' };

  it('dispatches forward immediately and records a reversible history entry', () => {
    const { store } = createFakeStore(makeDoc());
    const dispatch = store.dispatch as Mock;
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });

    engine.commitStructural('Select layer', forward, inverse);
    // Forward dispatched once; the edit is now undoable but not redoable.
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenNthCalledWith(1, forward);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    // Undo dispatches the inverse and flips the stacks.
    engine.undo();
    expect(dispatch).toHaveBeenNthCalledWith(2, inverse);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    // Redo re-dispatches the forward.
    engine.redo();
    expect(dispatch).toHaveBeenNthCalledWith(3, forward);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);

    engine.dispose();
  });
});

// ---- drawLayerThumbnail: cache-backed layer previews --------------------

const createThumbnailTarget = (): {
  calls: { args: unknown[]; op: string }[];
  target: HTMLCanvasElement;
} => {
  const calls: { args: unknown[]; op: string }[] = [];
  const ctx = {
    clearRect: (...args: unknown[]) => calls.push({ args, op: 'clearRect' }),
    drawImage: (...args: unknown[]) => calls.push({ args, op: 'drawImage' }),
  };
  const target = { getContext: () => ctx, height: 0, width: 0 } as unknown as HTMLCanvasElement;
  return { calls, target };
};

describe('drawLayerThumbnail', () => {
  it('returns false and draws nothing when the layer has no cache', () => {
    const { engine } = createEngine();
    const { calls, target } = createThumbnailTarget();
    expect(engine.drawLayerThumbnail('missing', target, 96)).toBe(false);
    expect(calls).toHaveLength(0);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    // One frame creates the layer cache entry (10x10).
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const { calls, target } = createThumbnailTarget();
    expect(engine.drawLayerThumbnail('a', target, 96)).toBe(true);
    // 10x10 never upscales, so the target keeps the source dimensions.
    expect(target.width).toBe(10);
    expect(target.height).toBe(10);
    expect(calls.map((call) => call.op)).toEqual(['clearRect', 'drawImage']);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    // One frame builds both layer caches; await the async bitmap decode so both
    // caches are READY (merge refuses stale/in-flight caches — finding 20).
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const surfacesBeforeMerge = surfaces.length;
    expect(engine.mergeLayerDown('upper')).toBe(true);

    // The reducer is asked to collapse the two layers into a paint layer.
    const mergeCall = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
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

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();

    expect(engine.mergeLayerDown('below')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );

    engine.dispose();
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
      engine.attach(screen.element, overlay.element);
      raf.flush();

      expect(engine.mergeLayerDown('upper')).toBe(false);
      expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
        false
      );

      engine.dispose();
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
      engine.attach(screen.element, overlay.element);
      // Await the kept layer's bitmap decode so its cache is READY before merge
      // (the empty operand needs no decode; merge refuses stale/in-flight caches).
      raf.flush();
      await flushMicrotasks();
      raf.flush();

      const surfacesBeforeMerge = surfaces.length;
      expect(engine.mergeLayerDown('upper')).toBe(true);

      // Single-dispatch collapse still happens (undo semantics unchanged).
      expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
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

      engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const surfacesBeforeMerge = surfaces.length;
    expect(engine.mergeLayerDown('upper')).toBe(true);

    // The collapse still dispatches (the upper layer is removed).
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
      true
    );
    // No merged surface was allocated: a 0×0 union surface would throw, and there
    // are no pixels to composite.
    expect(surfaces.length).toBe(surfacesBeforeMerge);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();

    expect(engine.mergeLayerDown('upper')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );
    // Document unchanged: still two layers, each with its original type — no
    // mask was blitted into, and no mask was clobbered into a raster layer.
    expect(engine.getDocument()!.layers.map((l) => l.type)).toEqual(doc.layers.map((l) => l.type));

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();

    expect(engine.mergeLayerDown('upper')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );
    expect(engine.getDocument()!.layers.map((l) => l.type)).toEqual(['inpaint_mask', 'inpaint_mask']);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
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

    expect(await engine.booleanMergeRasterLayers('upper', operation)).toBe('merged');

    const resultSurface = surfaces.find((surface) =>
      surface.callLog.some(
        (entry) => entry.op === 'set' && entry.args[0] === 'globalCompositeOperation' && entry.args[1] === composite
      )
    );
    expect(resultSurface).toBeDefined();
    expect(resultSurface!.callLog.filter((entry) => entry.op === 'drawImage')).toHaveLength(2);

    const merged = engine.getDocument()!;
    const result = merged.layers.find((layer) => layer.id !== 'upper' && layer.id !== 'below');
    expect(result).toMatchObject({
      isEnabled: true,
      source: { bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(merged.layers.find((layer) => layer.id === 'upper')?.isEnabled).toBe(false);
    expect(merged.layers.find((layer) => layer.id === 'below')?.isEnabled).toBe(false);

    engine.undo();
    expect(engine.getDocument()!.layers.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      ['upper', true],
      ['below', true],
    ]);

    engine.redo();
    expect(engine.getDocument()!.layers.map((layer) => [layer.id, layer.isEnabled])).toEqual([
      [result!.id, true],
      ['upper', false],
      ['below', false],
    ]);
    engine.dispose();
  });

  it('does not publish a boolean merge into another active project after awaiting exports', async () => {
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    (store.dispatch as Mock).mockClear();

    const merge = engine.booleanMergeRasterLayers('upper', 'intersect');
    setActiveProjectId('p2');

    expect(await merge).toBe('not-ready');
    expect(store.dispatch).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('refuses the operation until both raster caches are ready', async () => {
    const { engine, raf } = setup();
    raf.flush();

    expect(await engine.booleanMergeRasterLayers('upper', 'intersect')).toBe('not-ready');
    expect(engine.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'below']);
    engine.dispose();
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

    expect(await engine.booleanMergeRasterLayers('upper', 'exclude')).toBe('unsupported');
    expect(await engine.booleanMergeRasterLayers('missing', 'exclude')).toBe('missing');
    expect(engine.getDocument()!.layers).toEqual(doc.layers);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');
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

    expect(await engine.booleanMergeRasterLayers('upper', 'intersect')).toBe('merged');
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    return { engine, raf, surfaces };
  };

  it('composites visible raster content through mask alpha and inserts one undoable raster layer', async () => {
    const { engine, raf, surfaces } = setup();
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    const result = await engine.extractMaskedArea('mask');
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

    const extracted = engine.getDocument()!.layers.find((layer) => layer.id === result.layerId);
    expect(extracted).toMatchObject({
      isEnabled: true,
      source: { bitmap: null, offset: { x: 15, y: 25 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    expect(engine.getDocument()!.layers.find((layer) => layer.id === 'mask')).toEqual(maskedDoc().layers[0]);
    expect(engine.getDocument()!.layers.find((layer) => layer.id === 'upper')?.isEnabled).toBe(true);
    expect(engine.getDocument()!.layers.find((layer) => layer.id === 'below')?.isEnabled).toBe(true);

    engine.undo();
    expect(engine.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(false);
    engine.redo();
    expect(engine.getDocument()!.layers.some((layer) => layer.id === result.layerId)).toBe(true);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    const rasterExport = await engine.exportLayerPixels(raster.id);
    const controlExport = await engine.exportLayerPixels(control.id);
    expect(rasterExport.status).toBe('ok');
    expect(controlExport.status).toBe('ok');
    if (rasterExport.status !== 'ok' || controlExport.status !== 'ok') {
      throw new Error('fixture layers did not rasterize');
    }

    expect((await engine.extractMaskedArea(mask.id)).status).toBe('extracted');

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
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect((await engine.extractMaskedArea('mask')).status).toBe('extracted');
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    (store.dispatch as Mock).mockClear();

    const extraction = engine.extractMaskedArea('mask');
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
    engine.dispose();
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

    expect(await engine.extractMaskedArea('mask')).toEqual({ status: 'unsupported' });
    engine.dispose();
  });

  it('refuses extraction until the mask and contributor caches are ready', async () => {
    const { engine, raf } = setup();
    raf.flush();

    expect(await engine.extractMaskedArea('mask')).toEqual({ status: 'not-ready' });
    expect(engine.getDocument()!.layers.map((layer) => layer.id)).toEqual(['mask', 'upper', 'below']);
    engine.dispose();
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

    expect(await engine.extractMaskedArea('missing')).toEqual({ status: 'missing' });
    expect(await engine.extractMaskedArea('upper')).toEqual({ status: 'unsupported' });
    expect(await engine.extractMaskedArea('mask')).toEqual({ status: 'not-ready' });
    expect(engine.getDocument()!.layers).toEqual(doc.layers);
    engine.dispose();

    const emptyDoc = maskedDoc();
    emptyDoc.layers[0] = maskLayer('mask');
    const emptyHarness = createReducerBackedStore(emptyDoc);
    const emptyEngine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: emptyHarness.projectId,
      store: emptyHarness.store,
    });
    expect(await emptyEngine.extractMaskedArea('mask')).toEqual({ status: 'empty' });
    emptyEngine.dispose();
  });
});

// ---- mergeVisibleRasterLayers: whole-fold pre-flight + interleave-safe fold ----

/**
 * An EngineStore backed by the REAL workbench reducer, so the fold's own
 * dispatches (reorder + mergeCanvasLayersDown) actually advance the document the
 * mirror re-reads between steps — the no-op mock store cannot exercise a
 * multi-step fold.
 */
const createReducerBackedStore = (
  document: CanvasDocumentContractV2
): { dispatch: Mock<(action: WorkbenchAction) => void>; projectId: string; store: EngineStore } => {
  let state = createInitialWorkbenchState();
  const projectId = state.projects[0]!.id;
  state = workbenchReducer(state, { document, type: 'replaceCanvasDocument' });
  const listeners = new Set<() => void>();
  const dispatch = vi.fn((action: WorkbenchAction) => {
    state = workbenchReducer(state, action);
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
      subscribe: (listener) => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
    },
  };
};

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
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    const screen = createFakeCanvas();
    const overlay = createFakeCanvas();
    engine.attach(screen.element, overlay.element);
    return { engine, raf, store };
  };

  it('refuses the WHOLE fold before any pixel work while a participant cache is not ready', async () => {
    const { engine, raf } = setup(interleavedDoc());
    // No frame has run yet: both paint bitmaps are undecoded (no ready caches).
    expect(engine.mergeVisibleRasterLayers()).toBe('not-ready');
    // Nothing happened — no reorder, no merge, no partial fold.
    expect(engine.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'mid-mask', 'below']);

    // Once the decodes land, the SAME call succeeds.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(engine.mergeVisibleRasterLayers()).toBe('merged');
    expect(engine.getDocument()!.layers.map((layer) => layer.id)).toEqual(['mid-mask', 'below']);

    engine.dispose();
  });

  it('folds visible rasters across an interleaved mask (reorder + merge), then reports nothing left', async () => {
    const { engine, raf } = setup(interleavedDoc());
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    expect(engine.mergeVisibleRasterLayers()).toBe('merged');

    // The upper raster merged INTO the lower one across the mask; the mask
    // survives untouched and the merged layer keeps the lower id + raster type.
    const layers = engine.getDocument()!.layers;
    expect(layers.map((layer) => layer.id)).toEqual(['mid-mask', 'below']);
    expect(layers.map((layer) => layer.type)).toEqual(['inpaint_mask', 'raster']);

    // Re-running has nothing to do (one raster left).
    expect(engine.mergeVisibleRasterLayers()).toBe('nothing');

    engine.dispose();
  });

  it('folds an all-empty run trivially instead of stalling on a half-merged stack (F4)', async () => {
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

    expect(engine.getDocument()!.layers.map((layer) => layer.id)).toEqual(['upper', 'below']);
    // The topmost (only) run is two empty rasters: the fold trivially collapses them
    // into one empty layer rather than reporting a silent no-op.
    expect(engine.mergeVisibleRasterLayers()).toBe('merged');
    expect(engine.getDocument()!.layers.map((layer) => layer.id)).toEqual(['below']);
    expect(engine.mergeVisibleRasterLayers()).toBe('nothing');

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    return { backend, dispatch, engine, raf, setDocument, surfaces };
  };

  const convertCalls = (dispatch: Mock): WorkbenchAction[] =>
    dispatch.mock.calls.map((call) => call[0] as WorkbenchAction).filter((a) => a.type === 'convertCanvasLayer');

  it('bakes to a CONTENT-sized paint layer at identity and dispatches convertCanvasLayer with the offset', () => {
    const { dispatch, engine, surfaces } = setup(shapeLayerDoc());
    const before = surfaces.length;

    expect(engine.rasterizeLayer('shape1')).toBe(true);

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

    engine.dispose();
  });

  it('undo re-converts to the ORIGINAL parametric source (no pixel snapshot)', () => {
    const { dispatch, engine } = setup(shapeLayerDoc());
    engine.rasterizeLayer('shape1');

    engine.undo();
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
    engine.redo();
    const afterRedo = convertCalls(dispatch);
    expect(afterRedo).toHaveLength(3);
    if (afterRedo[2]?.type === 'convertCanvasLayer' && afterRedo[2].layer.type === 'raster') {
      expect(afterRedo[2].layer.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });
    }
    engine.dispose();
  });

  it('redo re-bakes from params rather than pinning the doc-sized surface (byte-budget honesty)', () => {
    // Regression: the history entry declared bytes:256 but captured the doc-sized
    // `baked` surface in its redo closure, so repeated rasterizes could retain
    // gigabytes invisible to HISTORY_BYTE_BUDGET. Redo must re-bake instead.
    const { engine, surfaces } = setup(shapeLayerDoc());

    expect(engine.rasterizeLayer('shape1')).toBe(true);
    const afterApply = surfaces.length;

    engine.undo();
    // Undo only re-converts (a dispatch) — it bakes nothing.
    expect(surfaces.length).toBe(afterApply);

    engine.redo();
    // Redo allocates FRESH surfaces: it re-baked from params (content-sized to the
    // transformed shape bounds, 60×40) rather than reusing a surface pinned by the
    // entry.
    expect(surfaces.length).toBeGreaterThan(afterApply);
    const rebaked = surfaces.at(-1);
    expect(rebaked?.width).toBe(60);
    expect(rebaked?.height).toBe(40);
    expect(rebaked?.callLog.some((e) => e.op === 'drawImage')).toBe(true);

    engine.dispose();
  });

  it('rasterizes a gradient layer', () => {
    const doc = shapeLayerDoc();
    doc.layers[0] = {
      ...doc.layers[0],
      source: { angle: 45, kind: 'linear', stops: [{ color: '#000', offset: 0 }], type: 'gradient' },
    } as CanvasLayerContract;
    const { dispatch, engine } = setup(doc);
    expect(engine.rasterizeLayer('shape1')).toBe(true);
    expect(convertCalls(dispatch)).toHaveLength(1);
    engine.dispose();
  });

  it('is a no-op for a locked layer, a missing layer, and a non-parametric (paint) layer', () => {
    const { dispatch, engine } = setup(shapeLayerDoc({ isLocked: true }));
    expect(engine.rasterizeLayer('shape1')).toBe(false);
    expect(engine.rasterizeLayer('nope')).toBe(false);
    expect(convertCalls(dispatch)).toHaveLength(0);
    engine.dispose();

    const paintDoc = shapeLayerDoc();
    paintDoc.layers[0] = { ...paintDoc.layers[0], source: { bitmap: null, type: 'paint' } } as CanvasLayerContract;
    const paint = setup(paintDoc);
    expect(paint.engine.rasterizeLayer('shape1')).toBe(false);
    paint.engine.dispose();
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
        dispatch: (action) => store.dispatch(action),
        encodeSurface,
        getLayerSource: (layerId) => {
          const layer = engine.getDocument()?.layers.find((candidate) => candidate.id === layerId);
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
      engine.attach(screen.element, overlay.element);
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

      expect(engine.rasterizeLayer('shape1')).toBe(true);
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      engine.undo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      return harness;
    };

    /** Every dispatched `updateCanvasLayerSource` whose source is `paint`. */
    const paintSourceDispatches = (dispatch: Mock): Extract<WorkbenchAction, { type: 'updateCanvasLayerSource' }>[] =>
      dispatch.mock.calls
        .map((call) => call[0] as WorkbenchAction)
        .filter(
          (action): action is Extract<WorkbenchAction, { type: 'updateCanvasLayerSource' }> =>
            action.type === 'updateCanvasLayerSource' && action.source.type === 'paint'
        );

    it('await flushPendingUploads(): the layer stays the parametric shape and no paint dispatch fires', async () => {
      const { dispatch, engine, uploadImage } = rasterizeThenUndo();

      await engine.flushPendingUploads();

      // The guard drops the flush before it ever encodes/uploads.
      expect(uploadImage).not.toHaveBeenCalled();
      expect(paintSourceDispatches(dispatch)).toHaveLength(0);
      const layer = engine.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layer.source.type).toBe('shape');

      engine.dispose();
    });

    it('advancing the debounce timer: the layer stays the parametric shape and no paint dispatch fires', async () => {
      const { dispatch, engine, uploadImage } = rasterizeThenUndo();

      await vi.advanceTimersByTimeAsync(1500);

      expect(uploadImage).not.toHaveBeenCalled();
      expect(paintSourceDispatches(dispatch)).toHaveLength(0);
      const layer = engine.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layer.source.type).toBe('shape');

      engine.dispose();
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
      expect(engine.rasterizeLayer('shape1')).toBe(true);
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await vi.advanceTimersByTimeAsync(1500);
      await engine.flushPendingUploads();
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
      engine.undo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      // Redo → paint bake again; the fresh conversion lands on `bitmap: null`
      // (only the debounced flush fills in the persisted ref).
      engine.redo();
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();
      const layerAfterRedo = engine.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layerAfterRedo.source).toEqual({ bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' });

      // Flush again: the re-baked pixels are identical, so the content hash
      // dedupes back to img-x with NO new upload — but the ref must still be
      // re-dispatched into the contract, since the document currently reads
      // `bitmap: null`, not img-x.
      await vi.advanceTimersByTimeAsync(1500);
      await engine.flushPendingUploads();

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
      const layerFinal = engine.getDocument()!.layers[0] as CanvasRasterLayerContractV2;
      expect(layerFinal.source.type).toBe('paint');
      if (layerFinal.source.type === 'paint') {
        expect(layerFinal.source.bitmap).not.toBeNull();
        expect(layerFinal.source.bitmap?.imageName).toBe('img-x');
      }

      engine.dispose();
    });

    it('sanity check: a normal (non-reverted) rasterize DOES flush to paint (the guard only blocks the reverted case)', async () => {
      vi.useFakeTimers();
      let doc = shapeLayerDoc();
      const { dispatch, engine, raf, setDocument, uploadImage } = setupWithRealBitmapStore(doc);

      expect(engine.rasterizeLayer('shape1')).toBe(true);
      doc = applyLastConvert(doc, dispatch);
      setDocument(doc);
      raf.flush();

      await vi.advanceTimersByTimeAsync(1500);
      await engine.flushPendingUploads();

      expect(uploadImage).toHaveBeenCalledTimes(1);
      expect(paintSourceDispatches(dispatch)).toHaveLength(1);

      engine.dispose();
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
    engine.nudgeSelectedLayer(1, 0);
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('no-ops on a locked selected layer', () => {
    const { dispatch, engine } = setup(selectedImageDoc({ isLocked: true }));
    engine.nudgeSelectedLayer(0, 1);
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('no-ops on a hidden selected layer', () => {
    const { dispatch, engine } = setup(selectedImageDoc({ isEnabled: false }));
    engine.nudgeSelectedLayer(0, 1);
    expect(dispatch).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('dispatches a transform update and records an undoable entry', () => {
    const { dispatch, engine } = setup(selectedImageDoc());
    engine.nudgeSelectedLayer(3, -2);
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenNthCalledWith(1, {
      id: 'a',
      patch: { transform: { x: 3, y: -2 } },
      type: 'updateCanvasLayer',
    });
    expect(engine.stores.canUndo.get()).toBe(true);

    // Undo dispatches the inverse (back to the original position).
    engine.undo();
    expect(dispatch).toHaveBeenNthCalledWith(2, {
      id: 'a',
      patch: { transform: { x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });
    engine.dispose();
  });

  it('coalesces a rapid same-layer burst into one history entry', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_000);
    const { engine } = setup(selectedImageDoc());
    engine.nudgeSelectedLayer(1, 0);
    engine.nudgeSelectedLayer(1, 0);
    engine.nudgeSelectedLayer(1, 0);
    // A single undo empties the stack: the burst collapsed to one entry.
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.undo();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
    vi.restoreAllMocks();
  });

  it('starts a fresh entry once the coalescing window elapses', () => {
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(1_000);
    const { engine } = setup(selectedImageDoc());
    engine.nudgeSelectedLayer(1, 0);
    now.mockReturnValue(2_000); // > 500ms later
    engine.nudgeSelectedLayer(1, 0);
    // Two distinct entries: one undo still leaves something to undo.
    engine.undo();
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.undo();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('move');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 30));
    overlay.fire('pointerup', pointerAt(50, 30, { buttons: 0 }));

    const transformUpdates = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .filter((action) => action.type === 'updateCanvasLayer');
    expect(transformUpdates).toHaveLength(1);
    expect(transformUpdates[0]).toMatchObject({ id: 'paint1', type: 'updateCanvasLayer' });
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('move');

    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointerup', pointerAt(20, 20, { buttons: 0 }));

    const actions = dispatch.mock.calls.map((call) => call[0] as WorkbenchAction);
    expect(actions.some((action) => action.type === 'updateCanvasLayer')).toBe(false);
    expect(actions.some((action) => action.type === 'setCanvasSelectedLayer')).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.dispose();
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
    engine.onStrokeCommitted((event) => strokes.push(event));

    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

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
      .map((call) => call[0] as WorkbenchAction)
      .filter((action) => action.type === 'addCanvasLayer');
    expect(addCalls).toHaveLength(1);

    const putBefore = () => putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.beforeImageData);
    const beforeUndoPutBefore = putBefore();

    // Undo: removes the auto-created layer, and does NOT restore pre-stroke pixels
    // (the layer's cache is gone).
    engine.undo();
    const removeAfterUndo = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .filter((action) => action.type === 'removeCanvasLayers');
    expect(removeAfterUndo).toHaveLength(1);
    expect(removeAfterUndo[0]).toEqual({ ids: [newLayerId], type: 'removeCanvasLayers' });
    // No new before-pixel putImageData was introduced by the undo.
    expect(putBefore()).toBe(beforeUndoPutBefore);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);

    // Redo: re-adds the layer and re-applies the stroke's after pixels.
    engine.redo();
    const addAfterRedo = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .filter((action) => action.type === 'addCanvasLayer');
    expect(addAfterRedo).toHaveLength(2); // original auto-create + redo re-add
    expect(putImageDataCalls(surfaces).some((call) => call.image === strokes[0]!.afterImageData)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush(); // initial (no layers, no preview)
    screen.surface.callLog.length = 0;

    // Two rapid selections; the second supersedes the first.
    engine.setStagedPreview({ dataUrl: dataUrl('AAAA'), height: 10, width: 10 }); // decode #0
    engine.setStagedPreview({ dataUrl: dataUrl('BBBB'), height: 20, width: 20 }); // decode #1

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

    engine.dispose();
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
    engine.attach(screen.element, createFakeCanvas().element);

    engine.setStagedPreview({ dataUrl: dataUrl('AAAA'), height: 10, width: 10 });
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();
    expect(stagedDraws(screen.surface).length).toBeGreaterThan(0);

    screen.surface.callLog.length = 0;
    engine.setStagedPreview(null);
    raf.flush();
    expect(stagedDraws(screen.surface)).toHaveLength(0);

    engine.dispose();
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
    engine.attach(screen.element, createFakeCanvas().element);

    engine.setStagedPreview({ dataUrl: dataUrl('AAAA'), height: 10, width: 10 });
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();
    expect(stagedDraws(screen.surface).at(-1)!.slice(1)).toEqual([0, 0, 10, 10]);

    // Move the bbox (an ordinary edit, same document revision — not a replacement).
    screen.surface.callLog.length = 0;
    setDocument(emptyDoc({ height: 10, width: 10, x: 30, y: 40 }));
    raf.flush();
    expect(stagedDraws(screen.surface).at(-1)!.slice(1)).toEqual([30, 40, 10, 10]);

    engine.dispose();
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
    engine.attach(screen.element, createFakeCanvas().element);

    engine.setStagedPreview({ imageName: 'staged-candidate' });
    await flushMicrotasks();
    raf.flush();

    expect(resolver).toHaveBeenCalledWith('staged-candidate');
    // The decoded (stub, 0-sized) surface is drawn at the current bbox origin.
    expect(stagedDraws(screen.surface).at(-1)!.slice(1, 3)).toEqual([5, 7]);

    engine.dispose();
  });
});

describe('setFilterPreview', () => {
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

  it('decodes a per-layer filter preview candidate through the resolver and schedules a render', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(emptyDoc());
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);
    raf.flush();
    resolver.mockClear();

    engine.setFilterPreview('layer-x', { imageName: 'filtered-preview' });
    await flushMicrotasks();

    expect(resolver).toHaveBeenCalledWith('filtered-preview');
    // The resolved decode invalidates the layer, so a frame is scheduled.
    expect(raf.pendingCount()).toBeGreaterThan(0);

    engine.dispose();
  });

  it('clears a filter preview on null without decoding again', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const { store } = createReactiveStore(emptyDoc());
    const resolver = vi.fn(() => Promise.resolve(new Blob()));
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: resolver,
      projectId: 'p1',
      store,
    });
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);
    raf.flush();

    engine.setFilterPreview('layer-x', { imageName: 'p' });
    await flushMicrotasks();
    raf.flush();
    resolver.mockClear();

    // Clearing does not initiate a new decode.
    engine.setFilterPreview('layer-x', null);
    await flushMicrotasks();
    expect(resolver).not.toHaveBeenCalled();

    engine.dispose();
  });

  it('drops a stale decode superseded by a newer set for the same layer (version guard)', async () => {
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
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);
    raf.flush();

    // Two rapid previews for the SAME layer; the second supersedes the first.
    engine.setFilterPreview('L', { imageName: 'A' }); // decode #0 (stale)
    engine.setFilterPreview('L', { imageName: 'B' }); // decode #1 (newest)
    await flushMicrotasks();

    // The newest decode resolves and schedules a frame.
    bitmaps.resolveBitmap(1);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    // The stale (older) decode resolves afterwards: the version guard drops it, so
    // it must NOT schedule another frame.
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBe(0);

    engine.dispose();
  });

  // ---- Review fix (Task 38, finding 1): pruning on layer removal / doc replace ---
  //
  // A filter preview is per-layer session state. If the layer it belongs to leaves
  // the mirrored document (deleted, or a wholesale document swap), the preview must
  // be dropped AND its decode token bumped — never reset to a value a still-in-flight
  // decode could later match — or a late-resolving (or undo-restored) decode can
  // resurrect a preview for a layer nobody is looking at anymore.

  it("bumps the layer's preview token when its layer leaves the document, so a late-resolving decode never resurrects a preview after an undo restores the same id", async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const bitmaps = createDeferredBitmapBackend();
    const withLayer = { ...emptyDoc(), layers: [previewableLayer('L')] };
    const { setDocument, store } = createReactiveStore(withLayer);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);
    raf.flush();

    // Start a filter-preview decode for layer L (decode #0).
    engine.setFilterPreview('L', { imageName: 'A' });
    await flushMicrotasks();

    // Layer L leaves the document via an ordinary layer-array edit (same dims,
    // same revision) — onLayersChanged, not onDocumentReplaced. This must bump
    // L's token so decode #0's captured token can never match again.
    setDocument({ ...withLayer, layers: [] });
    raf.flush();

    // An undo restores the layer (same id) and the user starts a fresh preview
    // (decode #1) before decode #0 ever resolves.
    setDocument(withLayer);
    raf.flush();
    // The restored layer's OWN (unrelated) rasterize dispatch/completion cycle
    // must be fully settled before the assertions below, or its completion's
    // unconditional `scheduler.invalidate` would land during a later
    // `flushMicrotasks()` and be mistaken for filter-preview activity.
    await flushMicrotasks();
    raf.flush();
    engine.setFilterPreview('L', { imageName: 'C' });
    await flushMicrotasks();

    // Decode #0 (stale, pre-deletion) resolves late: the bumped token rejects
    // it, so it must NOT resurrect a preview for the restored layer.
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBe(0);

    // Decode #1 (the fresh, post-restore request) resolves normally.
    bitmaps.resolveBitmap(1);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBeGreaterThan(0);

    engine.dispose();
  });

  it('clears every filter preview on a wholesale document replace, rejecting a decode that was in flight before the swap', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);

    const bitmaps = createDeferredBitmapBackend();
    const withLayer = { ...emptyDoc(), layers: [previewableLayer('L')] };
    const { setDocument, store } = createReactiveStore(withLayer);
    const engine = createCanvasEngine({
      backend: bitmaps.backend,
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: 'p1',
      store,
    });
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);
    raf.flush();

    // A live, already-decoded preview for L.
    engine.setFilterPreview('L', { imageName: 'A' }); // decode #0
    await flushMicrotasks();
    bitmaps.resolveBitmap(0);
    await flushMicrotasks();
    raf.flush();

    // A second, slower decode for the SAME id starts (decode #1) and is still
    // in flight when a wholesale document swap arrives (dims change — a project
    // switch or snapshot restore that changes dims).
    engine.setFilterPreview('L', { imageName: 'A2' }); // decode #1, in flight
    await flushMicrotasks();
    setDocument({ ...withLayer, height: 200, width: 200 }, 1);
    raf.flush();
    // Settle the swapped-in layer's own (unrelated) rasterize dispatch/completion
    // cycle before asserting — see the analogous comment in the prior test.
    await flushMicrotasks();
    raf.flush();

    // Decode #1 resolves AFTER the swap: even though it targets the SAME layer
    // id, it describes a document that no longer exists, so it must be rejected.
    bitmaps.resolveBitmap(1);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBe(0);

    // A fresh request post-swap still decodes normally.
    engine.setFilterPreview('L', { imageName: 'B' }); // decode #2
    await flushMicrotasks();
    bitmaps.resolveBitmap(2);
    await flushMicrotasks();
    expect(raf.pendingCount()).toBeGreaterThan(0);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, { buttons: 0 }));
    return engine;
  };

  it('keeps an unflushed paint layer’s pixels on a transform/opacity-only change (no re-rasterize)', async () => {
    const { engine, paintCache, raf, resolver, setDocument } = await paintOneStroke();

    // Baseline: the stroke composited into the cache (a drawImage). Exactly one
    // full clear so far — the initial rasterize; the stroke never clears.
    expect(paintCache.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
    const clearsBefore = fullClearCount(paintCache);

    // A prop-only edit that PRESERVES the source reference (exactly as the reducer
    // does — it spreads `...layer`): opacity + a transform nudge.
    const doc = engine.getDocument()!;
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
    expect(engine.getDocument()!.layers[0]!.opacity).toBe(0.5);

    engine.dispose();
  });

  it('exports an unflushed paint layer from its ready live cache when the contract bitmap is still null', async () => {
    const { engine, paintCache } = await paintOneStroke();

    const result = await engine.exportLayerPixels('paint1');

    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.surface).toBe(paintCache);
      expect(result.rect.width).toBeGreaterThan(0);
      expect(result.rect.height).toBeGreaterThan(0);
    }
    engine.dispose();
  });

  it('preserves live pixels through contract copy and conversion history', async () => {
    const engine = await paintOneStrokeWithReducer();
    const raster = engine.getDocument()!.layers[0]!;
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

    expect(engine.commitLayerCopy('Copy layer', raster.id, copy, 0)).toBe(true);
    expect((await engine.exportLayerPixels(copy.id)).status).toBe('ok');
    engine.undo();
    expect(engine.getDocument()!.layers.some((layer) => layer.id === copy.id)).toBe(false);
    engine.redo();
    expect((await engine.exportLayerPixels(copy.id)).status).toBe('ok');

    expect(engine.commitLayerConversion('Convert layer', raster, control)).toBe(true);
    expect(engine.getDocument()!.layers.find((layer) => layer.id === raster.id)?.type).toBe('control');
    expect((await engine.exportLayerPixels(raster.id)).status).toBe('ok');
    engine.undo();
    expect(engine.getDocument()!.layers.find((layer) => layer.id === raster.id)?.type).toBe('raster');
    expect((await engine.exportLayerPixels(raster.id)).status).toBe('ok');
    engine.dispose();
  });

  it('commitLayerConversion requires the caller to hold the immutable live layer object', async () => {
    const { dispatch, projectId, store } = createReducerBackedStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exportLayerPixels('a')).status).toBe('ok');
    const live = engine.getDocument()!.layers[0]!;
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

    expect(engine.commitLayerConversion('Convert', structuredClone(live), converted)).toBe(false);
    expect(dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'convertCanvasLayer' }));
    expect(engine.getDocument()!.layers[0]!.type).toBe('raster');
    engine.dispose();
  });

  it('commitLayerConversion refuses conversion when the live layer is locked', async () => {
    const { dispatch, projectId, store } = createReducerBackedStore(makeDoc());
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId,
      store,
    });
    expect((await engine.exportLayerPixels('a')).status).toBe('ok');
    const expectedUnlocked = engine.getDocument()!.layers[0]!;
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

    expect(engine.commitLayerConversion('Convert', expectedUnlocked, converted)).toBe(false);
    expect(dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'convertCanvasLayer' }));
    expect(engine.getDocument()!.layers[0]).toMatchObject({ isLocked: true, type: 'raster' });
    engine.dispose();
  });

  it('flushing after a prop-only change persists the painted (non-blank) surface', async () => {
    const { bitmapStore, engine, paintCache, raf, setDocument } = await paintOneStroke();
    const clearsBefore = fullClearCount(paintCache);

    const doc = engine.getDocument()!;
    const layer = doc.layers[0]!;
    setDocument({ ...doc, layers: [{ ...layer, opacity: 0.25 }] });
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    await engine.flushPendingUploads();

    // The flush barrier ran, and it operated on a cache that was never wiped: the
    // surface still carries the stroke (drawImage) with no re-rasterize clear, so
    // the bitmap store (which encodes this exact surface) persists real pixels.
    expect(bitmapStore.flushPendingUploads).toHaveBeenCalled();
    expect(fullClearCount(paintCache)).toBe(clearsBefore);
    expect(paintCache.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);

    engine.dispose();
  });

  it('re-rasterizes when the paint layer’s source genuinely changes (swap to a persisted bitmap)', async () => {
    const { engine, raf, resolver, setDocument } = await paintOneStroke();
    expect(resolver).not.toHaveBeenCalled();

    // A genuine source swap (undo/import → a NEW paint source object with a
    // persisted bitmap). isSelfEcho is false in the spy store, so this must
    // invalidate and re-rasterize — which decodes the persisted image.
    const doc = engine.getDocument()!;
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
    expect(resolver).toHaveBeenCalledWith('persisted');

    engine.dispose();
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
    engine.attach(createFakeCanvas().element, createFakeCanvas().element);

    // Initial rasterize of image 'a'.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver).toHaveBeenNthCalledWith(1, 'a');

    // Prop-only edit (opacity), source reference preserved: no re-rasterize.
    const doc = engine.getDocument()!;
    setDocument({ ...doc, layers: [{ ...doc.layers[0]!, opacity: 0.3 }] });
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(1);

    // Source swap (new image name → new source object): re-rasterizes the new source.
    const doc2 = engine.getDocument()!;
    const imgLayer = doc2.layers[0] as CanvasRasterLayerContractV2;
    setDocument({
      ...doc2,
      layers: [{ ...imgLayer, source: { image: { height: 10, imageName: 'a-v2', width: 10 }, type: 'image' } }],
    });
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    expect(resolver).toHaveBeenCalledTimes(2);
    expect(resolver).toHaveBeenNthCalledWith(2, 'a-v2');

    engine.dispose();
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
    engine.attach(createInputCanvas().element, overlay.element);

    // Settle the initial empty paint/mask rasterization before drawing. A stroke
    // then grows that current cache without updating the persisted contract.
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    engine.setTool('brush');
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
      expect(engine.hasExportableLayerContent(id), id).toBe(true);
    }
    engine.dispose();
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

    expect(engine.hasExportableLayerContent('empty-paint')).toBe(false);
    expect(engine.hasExportableLayerContent('polygon')).toBe(false);
    expect(engine.hasExportableLayerContent('missing')).toBe(false);
    engine.dispose();
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

    expect(engine.hasExportableLayerContent('persisted-mask')).toBe(true);
    expect(engine.hasExportableLayerContent('empty-mask')).toBe(false);
    engine.dispose();
  });

  it('returns true for current live unpersisted paint pixels', async () => {
    const { engine } = await createLiveUnpersistedLayer(sourceLayer('live-paint', { bitmap: null, type: 'paint' }));

    expect(engine.hasExportableLayerContent('live-paint')).toBe(true);
    engine.dispose();
  });

  it('returns false for a stale non-empty live cache with no persisted pixels', async () => {
    const { engine, setDocument } = await createLiveUnpersistedLayer(
      sourceLayer('stale-paint', { bitmap: null, type: 'paint' })
    );
    expect(engine.hasExportableLayerContent('stale-paint')).toBe(true);

    const doc = engine.getDocument()!;
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
    expect(engine.drawLayerThumbnail('stale-paint', target, 96)).toBe(true);
    expect(engine.hasExportableLayerContent('stale-paint')).toBe(false);
    engine.dispose();
  });

  it('returns true for current live unpersisted mask pixels', async () => {
    const { engine } = await createLiveUnpersistedLayer(maskLayer('live-mask'));

    expect(engine.hasExportableLayerContent('live-mask')).toBe(true);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');
    // Open a stroke (pointer down + move, NO up) so the gesture stays active.
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(30, 30));
    return { dispatch, engine, overlay };
  };

  it('no-ops nudgeSelectedLayer while a stroke gesture is open', () => {
    const { dispatch, engine, overlay } = startOpenStroke();
    const before = dispatch.mock.calls.length;

    engine.nudgeSelectedLayer(1, 0);
    // No structural transform dispatch, no history entry.
    expect(dispatch.mock.calls.filter((call) => call[0].type === 'updateCanvasLayer')).toHaveLength(0);
    expect(engine.stores.canUndo.get()).toBe(false);

    // After the gesture ends, the nudge lands.
    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    engine.nudgeSelectedLayer(1, 0);
    expect(dispatch.mock.calls.length).toBeGreaterThan(before);
    expect(dispatch.mock.calls.some((call) => call[0].type === 'updateCanvasLayer')).toBe(true);

    engine.dispose();
  });

  it('no-ops commitStructural while a stroke gesture is open, then commits after it ends', () => {
    const { dispatch, engine, overlay } = startOpenStroke();
    const forward: WorkbenchAction = { id: 'x', type: 'setCanvasSelectedLayer' };
    const inverse: WorkbenchAction = { id: null, type: 'setCanvasSelectedLayer' };

    engine.commitStructural('Select', forward, inverse);
    // Nothing dispatched, nothing recorded on history mid-gesture.
    expect(dispatch.mock.calls.some((call) => call[0] === forward)).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);

    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    engine.commitStructural('Select', forward, inverse);
    expect(dispatch.mock.calls.some((call) => call[0] === forward)).toBe(true);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    // Build both layer caches so a merge could otherwise succeed; await the async
    // decode so both are READY (merge refuses stale/in-flight caches — finding 20).
    raf.flush();
    await flushMicrotasks();
    raf.flush();
    engine.setTool('brush');

    // Open a stroke into the selected 'upper' paint layer (no pointer-up).
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(30, 30));

    // Mid-gesture merge is refused (matches commitStructural/nudge): not undoable.
    expect(engine.mergeLayerDown('upper')).toBe(false);
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
      false
    );

    // After the gesture ends, the merge lands.
    overlay.fire('pointerup', pointerAt(30, 30, { buttons: 0 }));
    expect(engine.mergeLayerDown('upper')).toBe(true);
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'mergeCanvasLayersDown')).toBe(
      true
    );

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

    // A bare hover move (no button) sets the cursor ring at a known position.
    overlay.fire('pointermove', pointerAt(30, 30, { buttons: 0 }));
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    // The `[`/`]` path: a size step with NO pointer event must schedule a frame
    // (the ring redraws at its last position with the new radius).
    engine.stepBrushSize(1);
    expect(raf.pendingCount()).toBeGreaterThan(0);
    raf.flush();

    // The options-bar slider path (a direct store write) likewise invalidates.
    engine.stores.brushOptions.set({ ...engine.stores.brushOptions.get(), size: 123 });
    expect(raf.pendingCount()).toBeGreaterThan(0);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    // No hover move happened, so there is no ring to resize: no frame scheduled.
    engine.stepBrushSize(1);
    expect(raf.pendingCount()).toBe(0);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('bbox');

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
    expect(dispatch.mock.calls.some((call) => (call[0] as WorkbenchAction).type === 'setCanvasBbox')).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('move');

    // Drag the doc-sized paint layer.
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 40));

    const updatesBefore = dispatch.mock.calls.filter(
      (call) => (call[0] as WorkbenchAction).type === 'updateCanvasLayer'
    ).length;

    setDocument({ ...paintDoc(), height: 200, width: 200 });

    // Pointer-up after the swap commits no transform update (the gesture was cancelled).
    overlay.fire('pointerup', pointerAt(50, 40, { buttons: 0 }));
    const updatesAfter = dispatch.mock.calls.filter(
      (call) => (call[0] as WorkbenchAction).type === 'updateCanvasLayer'
    ).length;
    expect(updatesAfter).toBe(updatesBefore);
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('brush');

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

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    // Zoom in to 20× → a view invalidation → recomposite with smoothing off.
    screen.surface.callLog.length = 0;
    engine.getViewport().zoomAtPoint(20, { x: 0, y: 0 });
    raf.flush();
    expect(findSet(screen.surface, 'imageSmoothingEnabled')).toContain(false);

    // Zoom out below 1× → recomposite with smoothing on (clean down-scale).
    screen.surface.callLog.length = 0;
    engine.getViewport().zoomAtPoint(0.5, { x: 0, y: 0 });
    raf.flush();
    expect(findSet(screen.surface, 'imageSmoothingEnabled')).toContain(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);

    // Force several composited frames (each recomposites and re-`createPattern`s).
    for (let i = 0; i < 5; i++) {
      engine.getViewport().zoomAtPoint(1 + i, { x: 0, y: 0 });
      raf.flush();
      await flushMicrotasks();
    }

    // The tile surface was built exactly once and reused (no per-frame rebuild).
    expect(tileSurfaceSizes).toEqual(['16x16']);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
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

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    expect(tiles).toHaveLength(1);

    // Re-feeding the SAME colors is a no-op (the store's equality check drops it),
    // so no invalidation and no tile rebuild.
    engine.stores.checkerColors.set({ ...DEFAULT_CHECKER_COLORS });
    raf.flush();
    await flushMicrotasks();
    expect(tiles).toHaveLength(1);

    engine.dispose();
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
    engine.resize(400, 400, 1);

    engine.fitToView();

    // avail = 400 - 48*2 = 304; fitting the 100px bbox → 3.04. Fitting the 1000px
    // doc rect would have been ~0.304 — the doc rect is no longer the fit target.
    expect(engine.getViewport().getZoom()).toBeCloseTo(3.04, 2);
    engine.dispose();
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
    engine.resize(400, 400, 1);

    engine.fitToView();

    // Union spans (0,0)..(1000,1000) = 1000px; avail 304 → 0.304.
    expect(engine.getViewport().getZoom()).toBeCloseTo(0.304, 2);
    engine.dispose();
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

    engine.setTool('transform');
    engine.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 });
    dispatch.mockClear();
    engine.applyTransform();

    const layerDispatches = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
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
    engine.undo();
    const undoDispatch = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .find((action) => action.type === 'updateCanvasLayer');
    expect(undoDispatch).toEqual({
      id: 'a',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });

    engine.dispose();
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

    engine.setTool('transform');
    // A session opened on the (now hit-testable) text layer.
    expect(engine.stores.transformSession.get()).not.toBeNull();
    engine.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 });
    dispatch.mockClear();
    engine.applyTransform();

    const actions = dispatch.mock.calls.map((call) => call[0] as WorkbenchAction);
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
    engine.undo();
    const undo = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .find((a) => a.type === 'updateCanvasLayer');
    expect(undo).toEqual({
      id: 't',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });

    engine.dispose();
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

    engine.setTool('transform');
    engine.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 6 });
    expect(engine.stores.transformSession.get()).not.toBeNull();

    engine.handleEscapePriority({ gestureWasActive: false });
    expect(engine.stores.transformSession.get()).toBeNull();

    engine.dispose();
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

    engine.setTool('transform');
    dispatch.mockClear();
    engine.applyTransform();

    expect(dispatch.mock.calls.filter((c) => (c[0] as WorkbenchAction).type === 'updateCanvasLayer')).toHaveLength(0);
    expect(engine.stores.transformSession.get()).toBeNull();
    expect(engine.stores.canUndo.get()).toBe(false);

    engine.dispose();
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

    engine.setTool('transform');
    engine.updateTransformSession({ rotation: 0, scaleX: 3, scaleY: 3, x: 0, y: 0 });
    dispatch.mockClear();
    engine.cancelTransform();

    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.transformSession.get()).toBeNull();

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush(); // build the paint layer cache

    const surfacesBeforeApply = surfaces.length;
    const dirtyBefore = bitmapStore.markLayerDirty.mock.calls.length;

    engine.setTool('transform');
    engine.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 20 });
    dispatch.mockClear();
    engine.applyTransform();

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
      .map((call) => call[0] as WorkbenchAction)
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
    engine.undo();
    const undoDispatch = dispatch.mock.calls
      .map((call) => call[0] as WorkbenchAction)
      .find((action) => action.type === 'updateCanvasLayer');
    expect(undoDispatch).toEqual({
      id: 'paint1',
      patch: { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 } },
      type: 'updateCanvasLayer',
    });
    expect(engine.stores.canRedo.get()).toBe(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();

    engine.setTool('transform');
    expect(engine.stores.transformSession.get()).not.toBeNull();

    // A dims change is a wholesale document replacement.
    setDocument({ ...selectedImageDoc(), height: 200, width: 200 }, 1);

    expect(engine.stores.transformSession.get()).toBeNull();

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();

    engine.setTool('transform');
    engine.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 2, x: 5, y: 5 });
    expect(engine.stores.transformSession.get()).not.toBeNull();

    // Delete the session's layer via an ordinary layer-array edit (same dims,
    // same revision) — NOT a wholesale replace, so this exercises the
    // `onLayersChanged` teardown rather than `onDocumentReplaced`'s.
    setDocument({ ...doc, layers: [] });

    expect(engine.stores.transformSession.get()).toBeNull();

    engine.dispose();
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
      engine.attach(screen.element, overlay.element);
      raf.flush();

      engine.setTool('transform');
      // A numeric edit, as the options bar would drive.
      engine.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 1, x: 5, y: 0 });
      const edited = { rotation: 0, scaleX: 2, scaleY: 1, x: 5, y: 0 };
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      const setTool = engine.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void;

      setTool('view', { temporary: true }); // space down
      expect(engine.stores.activeTool.get()).toBe('view');
      // The session — and the numeric edit — survive the hold.
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      setTool('transform', { temporary: true }); // space up: resume
      expect(engine.stores.activeTool.get()).toBe('transform');
      // Resuming does not reopen the session from the layer's committed
      // transform, discarding the edit.
      expect(engine.stores.transformSession.get()?.transform).toEqual(edited);

      engine.dispose();
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
      engine.attach(screen.element, overlay.element);
      raf.flush();

      engine.setTool('transform');
      expect(engine.stores.transformSession.get()).not.toBeNull();

      engine.setTool('view'); // a real switch — no `{ temporary: true }`

      expect(engine.stores.transformSession.get()).toBeNull();

      engine.dispose();
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
      engine.attach(screen.element, overlay.element);
      raf.flush();

      engine.setTool('transform');
      expect(engine.stores.transformSession.get()).not.toBeNull();

      const setTool = engine.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void;
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

      engine.dispose();
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
    engine.selectAll();
    expect(engine.stores.hasSelection.get()).toBe(true);
    engine.deselect();
    expect(engine.stores.hasSelection.get()).toBe(false);
    engine.dispose();
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
    engine.invertSelection();
    expect(engine.stores.hasSelection.get()).toBe(true);
    engine.dispose();
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

  it('fillSelection on the selected paint layer records one undoable edit + persists', () => {
    const { bitmapStore, engine } = makeEngine(paintDoc());
    engine.selectAll();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(bitmapStore.markLayerDirty).toHaveBeenCalledWith('paint1');
    // Undo restores (canRedo becomes available).
    engine.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.dispose();
  });

  it('eraseSelection records one undoable edit (on existing content)', () => {
    const { engine } = makeEngine(paintDoc());
    engine.selectAll();
    // Content-sized: erase only affects EXISTING pixels, so give the layer content
    // first (a fill grows the empty paint cache to the selection). Then erase records
    // its own edit within that extent.
    engine.fillSelection();
    engine.eraseSelection();
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.dispose();
  });

  it('eraseSelection is a no-op on an EMPTY paint layer (no pixels to erase)', () => {
    const { engine } = makeEngine(paintDoc());
    engine.selectAll();
    engine.eraseSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('fillSelection is a no-op with no selection', () => {
    const { engine } = makeEngine(paintDoc());
    engine.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('fillSelection is a no-op on an image-source layer (deferred to rasterize task)', () => {
    const { engine } = makeEngine(imageSelectedDoc());
    engine.selectAll();
    engine.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('fillSelection is a no-op on a transparency-locked EMPTY layer (source-atop clamps to existing pixels)', () => {
    const doc = paintDoc();
    const target = doc.layers[0];
    if (target?.type === 'raster') {
      target.isTransparencyLocked = true;
    }
    const { engine } = makeEngine(doc);
    engine.selectAll();
    engine.fillSelection();
    // The layer has no existing pixels, so a transparency-locked (source-atop) fill
    // lands nothing — no undoable edit.
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('fillSelection is a no-op on a locked paint layer', () => {
    const { engine } = makeEngine(lockedPaintDoc());
    engine.selectAll();
    engine.fillSelection();
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.dispose();
  });

  it('clearCaches flushes pending bitmap uploads before invalidating (unflushed strokes survive)', async () => {
    const { bitmapStore, engine } = makeEngine(paintDoc());
    // The debug "Clear caches" action must persist any in-flight (debounced) paint
    // upload before it drops the layer caches — otherwise an unflushed stroke is lost.
    await engine.clearCaches();
    expect(bitmapStore.flushPendingUploads).toHaveBeenCalledTimes(1);
    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();
    await flushMicrotasks();
    raf.flush();

    screen.surface.callLog.length = 0;
    overlay.surface.callLog.length = 0;

    engine.selectAll();
    raf.flush();

    // Overlay redrawn (ants stroked); the screen composite never ran.
    const compositeOps = screen.surface.callLog.filter(
      (e) => e.op === 'drawImage' || e.op === 'clearRect' || e.op === 'fillRect'
    );
    expect(compositeOps).toHaveLength(0);
    expect(overlay.surface.callLog.some((e) => e.op === 'stroke')).toBe(true);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    raf.flush();

    // With a selection the loop keeps rescheduling itself frame after frame.
    engine.selectAll();
    raf.flush();
    raf.flush();
    expect(raf.pendingCount()).toBeGreaterThan(0);

    // Deselect stops it: after draining, nothing reschedules.
    engine.deselect();
    raf.flush();
    expect(raf.pendingCount()).toBe(0);

    // Re-select, then detach: the loop must not leak a pending frame.
    engine.selectAll();
    raf.flush();
    expect(raf.pendingCount()).toBeGreaterThan(0);
    engine.detach();
    expect(raf.pendingCount()).toBe(0);

    engine.dispose();
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
    engine.attach(screen.element, overlay.element);
    engine.setTool('lasso');
    raf.flush();
    dispatch.mockClear();

    overlay.fire('pointerdown', pointerAt(10, 10));
    overlay.fire('pointermove', pointerAt(40, 10));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(10, 40, { buttons: 0 }));

    expect(engine.stores.hasSelection.get()).toBe(true);
    // Selection is transient: no reducer traffic from the lasso gesture.
    expect(dispatch).not.toHaveBeenCalled();

    engine.dispose();
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
    engine.selectAll();
    expect(engine.stores.hasSelection.get()).toBe(true);
    // A new document revision replaces the mirror.
    setDocument(paintDoc(), 1);
    expect(engine.stores.hasSelection.get()).toBe(false);
    engine.dispose();
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
    const layerActions = () => dispatch.mock.calls.map((call) => call[0] as WorkbenchAction);
    return { dispatch, engine, layerActions, setDocument };
  };

  it('create-mode commit dispatches ONE addCanvasLayer with the typed content; undo removes it', () => {
    const { dispatch, engine, layerActions } = makeEngine(paintDoc());
    engine.setTool('text');
    engine.openTextCreate({ x: 10, y: 20 });
    expect(engine.stores.textEditSession.get()?.mode).toBe('create');

    dispatch.mockClear();
    engine.commitTextEdit('Typed here');

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
    engine.undo();
    const removes = layerActions().filter((a) => a.type === 'removeCanvasLayers');
    expect(removes).toHaveLength(1);
    engine.dispose();
  });

  it('an empty create-mode commit dispatches nothing (cancel semantics)', () => {
    const { dispatch, engine } = makeEngine(paintDoc());
    engine.setTool('text');
    engine.openTextCreate({ x: 0, y: 0 });
    dispatch.mockClear();
    engine.commitTextEdit('   ');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('edit-mode commit dispatches ONE updateCanvasLayerSource with the exact inverse', () => {
    const { dispatch, engine, layerActions } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()?.mode).toBe('edit');

    dispatch.mockClear();
    engine.commitTextEdit('changed');

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
    engine.undo();
    const inverse = layerActions().find((a) => a.type === 'updateCanvasLayerSource');
    if (inverse?.type === 'updateCanvasLayerSource' && inverse.source.type === 'text') {
      expect(inverse.source.content).toBe('hello');
    } else {
      throw new Error('expected the inverse to restore the original content');
    }
    engine.dispose();
  });

  it('folds a live style change into the single edit commit', () => {
    const { dispatch, engine, layerActions } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    engine.updateTextEditStyle({ color: '#ff0000', fontSize: 40 });

    dispatch.mockClear();
    engine.commitTextEdit('hello');

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
    engine.dispose();
  });

  it('an unchanged edit-mode commit dispatches nothing', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    dispatch.mockClear();
    engine.commitTextEdit('hello');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('cancel drops the session with no dispatch', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    dispatch.mockClear();
    engine.cancelTextEdit();
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
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
    engine.setTool('text');
    engine.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('a temporary tool switch (space-hold) preserves the open session', () => {
    const { engine } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).not.toBeNull();

    // Space-hold switches to the view tool temporarily; the session must survive.
    (engine.setTool as (id: ToolId, opts?: { temporary?: boolean }) => void)('view', { temporary: true });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    engine.dispose();
  });

  it('a real tool switch away from the text tool tears the session down', () => {
    const { engine } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).not.toBeNull();

    engine.setTool('brush');
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('drops the session on a wholesale document replace', () => {
    vi.stubGlobal('Path2D', class FakePath2D {});
    const { engine, setDocument } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    setDocument(paintDoc(), 1);
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('drops the session on dispose', () => {
    const { engine } = makeEngine(paintDoc());
    engine.setTool('text');
    engine.openTextCreate({ x: 0, y: 0 });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    engine.dispose();
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
    engine.setTool('text');
    engine.openTextCreate({ x: 10, y: 20 });
    engine.setTextEditContentReader(() => 'live typed text');

    expect(engine.commitOpenTextSession()).toBe(true);

    const adds = layerActions().filter((a) => a.type === 'addCanvasLayer');
    expect(adds).toHaveLength(1);
    const add = adds[0];
    if (add?.type === 'addCanvasLayer' && add.layer.type === 'raster' && add.layer.source.type === 'text') {
      expect(add.layer.source.content).toBe('live typed text');
    } else {
      throw new Error('expected an addCanvasLayer with the live reader content');
    }
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('commitOpenTextSession returns false when no session is open', () => {
    const { dispatch, engine } = makeEngine(paintDoc());
    expect(engine.commitOpenTextSession()).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('commitOpenTextSession falls back to the session content when no reader is registered', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1'); // session content = 'hello'
    dispatch.mockClear();

    expect(engine.commitOpenTextSession()).toBe(true);
    // No reader → uses the session's own content ('hello') → unchanged edit → no dispatch.
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.stores.textEditSession.get()).toBeNull();
    engine.dispose();
  });

  it('a second commit after the session closed is a no-op (one commit per close)', () => {
    const { dispatch, engine, layerActions } = makeEngine(paintDoc());
    engine.setTool('text');
    engine.openTextCreate({ x: 0, y: 0 });
    engine.setTextEditContentReader(() => 'once');
    expect(engine.commitOpenTextSession()).toBe(true);
    dispatch.mockClear();
    // The portal's onBlur would re-fire commitTextEdit after the pointerdown commit;
    // the session is already null, so it dispatches nothing.
    engine.commitTextEdit('once');
    expect(engine.commitOpenTextSession()).toBe(false);
    expect(layerActions().filter((a) => a.type === 'addCanvasLayer')).toHaveLength(0);
    engine.dispose();
  });

  it('handleEscapePriority cancels a defocused-but-open text session (no dispatch)', () => {
    const { dispatch, engine } = makeEngine(textDoc());
    engine.setTool('text');
    engine.openTextEdit('txt1');
    dispatch.mockClear();

    engine.handleEscapePriority({ gestureWasActive: false });
    expect(engine.stores.textEditSession.get()).toBeNull();
    expect(dispatch).not.toHaveBeenCalled();
    engine.dispose();
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
    expect(engine.contextMenuLayerIdAt({ x: 5, y: 5 })).toBe('control');
    engine.dispose();
  });

  it('returns null on empty space', () => {
    const engine = makeEngine(stackedDoc());
    expect(engine.contextMenuLayerIdAt({ x: 60, y: 60 })).toBeNull();
    engine.dispose();
  });

  it('returns null while a text-edit session is open (never opens over an in-progress edit)', () => {
    const engine = makeEngine(stackedDoc());
    engine.setTool('text');
    engine.openTextCreate({ x: 5, y: 5 });
    expect(engine.stores.textEditSession.get()).not.toBeNull();
    expect(engine.contextMenuLayerIdAt({ x: 5, y: 5 })).toBeNull();
    engine.dispose();
  });
});
