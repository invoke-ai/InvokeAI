/**
 * Inpaint mask engine behaviour: painting into a mask's alpha cache, persisting
 * the mask bitmap through the bitmap store, and the in-place mask invert.
 *
 * Isolated in its own file so it can `vi.mock` the canvas-image upload seam
 * (the engine's INTERNAL bitmap store uses it) without touching the rest of the
 * engine test suite.
 */

import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { CanvasDocumentContractV2, CanvasStateContractV2, WorkbenchState } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EngineStore } from './engine';
import type { StrokeCommittedEvent } from './tools/tool';

vi.mock('@workbench/canvas-operations/backend/canvasImages', () => ({
  CanvasImageUploadError: class extends Error {},
  uploadCanvasImage: vi.fn(() => Promise.resolve({ height: 64, imageName: 'mask-img', width: 64 })),
}));

const makeCanvas = (document: CanvasDocumentContractV2, documentRevision = 0): CanvasStateContractV2 =>
  ({
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
  }) as CanvasStateContractV2;

const maskDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [
    {
      blendMode: 'normal',
      id: 'mask1',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'Inpaint Mask 1',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    },
  ],
  selectedLayerId: 'mask1',
  version: 2,
  width: 100,
});

const createReactiveStore = (document: CanvasDocumentContractV2) => {
  let state = {
    activeProjectId: 'p1',
    projects: [{ canvas: makeCanvas(document), id: 'p1' }],
  } as unknown as WorkbenchState;
  const listeners = new Set<() => void>();
  const dispatch = vi.fn<(action: WorkbenchAction) => void>();
  return {
    dispatch,
    setDocument: (next: CanvasDocumentContractV2, revision = 0) => {
      state = {
        activeProjectId: 'p1',
        projects: [{ canvas: makeCanvas(next, revision), id: 'p1' }],
      } as unknown as WorkbenchState;
      for (const listener of listeners) {
        listener();
      }
    },
    store: {
      dispatch,
      getState: () => state,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    } as unknown as EngineStore,
  };
};

const createReducerBackedStore = (document: CanvasDocumentContractV2) => {
  let state = createInitialWorkbenchState();
  const projectId = state.activeProjectId;
  state = workbenchReducer(state, { document, type: 'replaceCanvasDocument' });
  const listeners = new Set<() => void>();
  const dispatch = vi.fn((action: WorkbenchAction) => {
    state = workbenchReducer(state, action);
    for (const listener of listeners) {
      listener();
    }
  });
  return {
    projectId,
    store: {
      dispatch,
      getState: () => state,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    } as EngineStore,
  };
};

const createControllableRaf = () => {
  let nextHandle = 1;
  const callbacks = new Map<number, FrameRequestCallback>();
  return {
    cancelFrame: (handle: number) => callbacks.delete(handle),
    flush: () => {
      const queued = [...callbacks.values()];
      callbacks.clear();
      for (const cb of queued) {
        cb(0);
      }
    },
    requestFrame: (cb: FrameRequestCallback) => {
      const handle = nextHandle++;
      callbacks.set(handle, cb);
      return handle;
    },
  };
};

const createInputCanvas = (width = 100, height = 100) => {
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
  const fire = (type: string, event: Partial<PointerEvent>) => {
    for (const handler of listeners.get(type) ?? []) {
      handler({ preventDefault: () => {}, ...event } as unknown as Event);
    }
  };
  return { element, fire };
};

const pointerAt = (x: number, y: number, buttons = 1): Partial<PointerEvent> =>
  ({
    altKey: false,
    button: 0,
    buttons,
    clientX: x,
    clientY: y,
    ctrlKey: false,
    metaKey: false,
    pointerId: 1,
    pointerType: 'mouse',
    pressure: 0.5,
    shiftKey: false,
    timeStamp: 0,
  }) as Partial<PointerEvent>;

const setupEngine = (doc: CanvasDocumentContractV2) => {
  const raf = createControllableRaf();
  vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
  vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
  vi.stubGlobal('Path2D', class FakePath2D {});

  const reactive = createReactiveStore(doc);
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    projectId: 'p1',
    store: reactive.store,
  });
  const strokes: StrokeCommittedEvent[] = [];
  engine.tools.onStrokeCommitted((event) => strokes.push(event));

  const screen = createInputCanvas();
  const overlay = createInputCanvas();
  engine.surface.attach(screen.element, overlay.element);
  raf.flush();

  return { dispatch: reactive.dispatch, engine, overlay, raf, setDocument: reactive.setDocument, strokes };
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('inpaint mask painting', () => {
  it('routes a brush stroke into the selected mask cache (no auto-created paint layer)', () => {
    const { dispatch, engine, overlay, strokes } = setupEngine(maskDoc());
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, 0));

    expect(strokes).toHaveLength(1);
    expect(strokes[0]!.layerId).toBe('mask1');
    expect(strokes[0]!.tool).toBe('brush');
    // A mask target must NOT spawn a fresh paint layer (the auto-create path).
    const added = dispatch.mock.calls.map((c) => c[0]).filter((a) => a.type === 'addCanvasLayer');
    expect(added).toHaveLength(0);
    engine.lifecycle.dispose();
  });

  it('erases from the mask cache with the eraser (destination-out) as a committed stroke', () => {
    const { engine, overlay, strokes } = setupEngine(maskDoc());
    // Paint some coverage first.
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(60, 60));
    overlay.fire('pointerup', pointerAt(60, 60, 0));
    // Then erase.
    engine.tools.setTool('eraser');
    overlay.fire('pointerdown', pointerAt(30, 30));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, 0));

    expect(strokes).toHaveLength(2);
    expect(strokes[1]!.tool).toBe('eraser');
    expect(strokes[1]!.layerId).toBe('mask1');
    engine.lifecycle.dispose();
  });

  it('does not paint into a locked mask', () => {
    const doc = maskDoc();
    doc.layers[0]!.isLocked = true;
    const { dispatch, engine, overlay, strokes } = setupEngine(doc);
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(40, 40));
    overlay.fire('pointerup', pointerAt(40, 40, 0));

    expect(strokes).toHaveLength(0);
    // Also never spawns a paint layer over the locked mask.
    expect(dispatch.mock.calls.map((c) => c[0]).some((a) => a.type === 'addCanvasLayer')).toBe(false);
    engine.lifecycle.dispose();
  });

  it('persists the mask via updateCanvasLayerConfig (bitmap + offset) after a stroke flush', async () => {
    const { dispatch, engine, overlay } = setupEngine(maskDoc());
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointerup', pointerAt(50, 50, 0));

    await engine.lifecycle.flushPendingUploads();

    const configDispatch = dispatch.mock.calls
      .map((c) => c[0])
      .find(
        (a): a is Extract<WorkbenchAction, { type: 'updateCanvasLayerConfig' }> =>
          a.type === 'updateCanvasLayerConfig' && a.id === 'mask1'
      );
    expect(configDispatch).toBeDefined();
    const config = configDispatch!.config;
    expect(config.layerType).toBe('inpaint_mask');
    // The mask bitmap ref + its content offset are dispatched (not an image source).
    expect(config).toHaveProperty('mask');
    const mask = (config as { mask: { bitmap: unknown; offset: unknown } }).mask;
    expect(mask.bitmap).toMatchObject({ imageName: 'mask-img' });
    expect(mask.offset).toBeDefined();
    // The mask persistence must NEVER dispatch a paint source (that would convert
    // the mask into a raster paint layer).
    expect(dispatch.mock.calls.map((c) => c[0]).some((a) => a.type === 'updateCanvasLayerSource')).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('mask invert', () => {
  it('inverts a mask as an undoable op and returns true', () => {
    const { engine } = setupEngine(maskDoc());
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.layers.invertMask('mask1')).toBe(true);
    // One undoable image patch was recorded.
    expect(engine.stores.canUndo.get()).toBe(true);
    // Round trips: undo then redo restore without throwing.
    engine.history.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.history.redo();
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('returns false for a missing layer, a non-mask layer, or a locked mask', () => {
    const lockedDoc = maskDoc();
    lockedDoc.layers[0]!.isLocked = true;
    const { engine } = setupEngine(lockedDoc);
    expect(engine.layers.invertMask('nope')).toBe(false);
    expect(engine.layers.invertMask('mask1')).toBe(false); // locked
    engine.lifecycle.dispose();
  });

  // Regression: the invert domain used to come ONLY from `getSourceContentRect`,
  // which reads the persisted `mask.bitmap` — stale/null until the debounced
  // bitmap-store flush runs. A stroke painted moments earlier (already reflected
  // in the live `layerCache`, not yet flushed to the contract) that extends past
  // the document bbox was silently excluded from the invert, so its pixels never
  // got flipped. The domain must union in the live cache rect too.
  it('unions the live (unflushed) cache rect into the invert domain, covering an out-of-bbox stroke', () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});

    const reactive = createReactiveStore(maskDoc());
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
      store: reactive.store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();

    engine.tools.setTool('brush');
    // Paint well outside the document bbox (0,0,100,100 per `maskDoc`). The live
    // cache grows to cover the stroke immediately; deliberately never call
    // `flushPendingUploads()`, so the contract's `mask.bitmap` stays null and
    // `getSourceContentRect` alone would report an empty content rect.
    overlay.fire('pointerdown', pointerAt(150, 150));
    overlay.fire('pointermove', pointerAt(180, 180));
    overlay.fire('pointerup', pointerAt(180, 180, 0));

    expect(engine.layers.invertMask('mask1')).toBe(true);

    // Growing the cache (both while painting and inside `invertMask` itself)
    // reallocates a fresh backing surface each time its extent changes, so
    // `surfaces` holds one entry per intermediate size, not one per layer. The
    // invert's own read/write always lands on the LAST surface that gets a
    // `putImageData` — the final, invert-sized surface — so pick that one
    // rather than the first surface that happens to have a `getImageData`
    // (which would be a stale, already-abandoned paint-time surface).
    const putSurfaces = surfaces.filter((surface) => surface.callLog.some((entry) => entry.op === 'putImageData'));
    const maskCache = putSurfaces[putSurfaces.length - 1];
    expect(maskCache).toBeDefined();
    const getCalls = maskCache!.callLog.filter((entry) => entry.op === 'getImageData');
    expect(getCalls.length).toBeGreaterThan(0);
    const [sx, sy, sw, sh] = getCalls[getCalls.length - 1]!.args as [number, number, number, number];
    // A domain of just the document bbox (0,0,100,100) — what `getSourceContentRect`
    // alone would report, since `mask.bitmap` is still null pre-flush — would read
    // a region that ends at x/y 100. Unioning in the live cache rect must grow the
    // domain to also cover the stroke out at document (150,150)-(180,180), well
    // past that 100 bound on both axes.
    expect(sx + sw).toBeGreaterThan(110);
    expect(sy + sh).toBeGreaterThan(110);

    engine.lifecycle.dispose();
  });

  it('extracts through a live unflushed mask whose contract bitmap is still null', async () => {
    const raf = createControllableRaf();
    vi.stubGlobal('requestAnimationFrame', raf.requestFrame);
    vi.stubGlobal('cancelAnimationFrame', raf.cancelFrame);
    vi.stubGlobal('Path2D', class FakePath2D {});
    const doc = maskDoc();
    doc.layers.push({
      blendMode: 'normal',
      id: 'raster1',
      isEnabled: true,
      isLocked: false,
      name: 'Raster 1',
      opacity: 1,
      source: { image: { height: 100, imageName: 'raster-image', width: 100 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    });
    const reducer = createReducerBackedStore(doc);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const overlay = createInputCanvas();
    const screen = createInputCanvas();
    engine.surface.attach(screen.element, overlay.element);
    raf.flush();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    raf.flush();

    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointerup', pointerAt(50, 50, 0));

    expect(await engine.exports.extractMaskedArea('mask1')).toMatchObject({ status: 'extracted' });
    engine.lifecycle.dispose();
  });
});

describe('mask clear', () => {
  it('clears a live unflushed inpaint mask and restores it through undo', () => {
    const { engine, overlay } = setupEngine(maskDoc());
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointerup', pointerAt(50, 50, 0));

    expect(engine.layers.clearMask('mask1')).toBe(true);
    expect(engine.layers.clearMask('mask1')).toBe(false);
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.history.undo();
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.history.redo();
    expect(engine.layers.clearMask('mask1')).toBe(false);
    engine.history.undo();
    expect(engine.layers.clearMask('mask1')).toBe(true);
    engine.lifecycle.dispose();
  });

  it('clears a regional-guidance mask', () => {
    const doc = maskDoc();
    doc.layers[0] = {
      autoNegative: false,
      blendMode: 'normal',
      id: 'mask1',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'Region 1',
      negativePrompt: null,
      opacity: 1,
      positivePrompt: null,
      referenceImages: [],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'regional_guidance',
    };
    const { engine, overlay } = setupEngine(doc);
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointerup', pointerAt(50, 50, 0));

    expect(engine.layers.clearMask('mask1')).toBe(true);
    engine.lifecycle.dispose();
  });

  it('clears a cold hidden persisted mask and restores its bitmap reference on undo', () => {
    const doc = maskDoc();
    const mask = doc.layers[0]!;
    if (mask.type !== 'inpaint_mask') {
      throw new Error('Expected inpaint mask fixture');
    }
    mask.isEnabled = false;
    mask.mask = {
      ...mask.mask,
      bitmap: { height: 40, imageName: 'persisted-mask', width: 50 },
      offset: { x: 7, y: 9 },
    };
    const { dispatch, engine } = setupEngine(doc);

    expect(engine.layers.clearMask('mask1')).toBe(true);
    expect(dispatch.mock.calls.at(-1)?.[0]).toMatchObject({
      config: { mask: { bitmap: null } },
      id: 'mask1',
      type: 'updateCanvasLayerConfig',
    });

    engine.history.undo();
    expect(dispatch.mock.calls.at(-1)?.[0]).toMatchObject({
      config: {
        mask: { bitmap: { imageName: 'persisted-mask' }, offset: { x: 7, y: 9 } },
      },
      id: 'mask1',
      type: 'updateCanvasLayerConfig',
    });
    engine.lifecycle.dispose();
  });

  it('refuses missing, non-mask, locked, and empty layers', () => {
    const locked = maskDoc();
    locked.layers[0]!.isLocked = true;
    const { engine } = setupEngine(locked);

    expect(engine.layers.clearMask('missing')).toBe(false);
    expect(engine.layers.clearMask('mask1')).toBe(false);
    engine.lifecycle.dispose();

    const raster = maskDoc();
    raster.layers[0] = {
      blendMode: 'normal',
      id: 'mask1',
      isEnabled: true,
      isLocked: false,
      name: 'Raster 1',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const { engine: rasterEngine } = setupEngine(raster);
    expect(rasterEngine.layers.clearMask('mask1')).toBe(false);
    rasterEngine.lifecycle.dispose();

    const { engine: emptyEngine } = setupEngine(maskDoc());
    expect(emptyEngine.layers.clearMask('mask1')).toBe(false);
    emptyEngine.lifecycle.dispose();
  });
});
