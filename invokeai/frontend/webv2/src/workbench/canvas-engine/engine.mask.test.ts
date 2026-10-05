/**
 * Mask painting, persistence and inversion tests. Isolated to mock the engine's internal image-upload seam without
 * affecting other engine tests.
 */

import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { WorkbenchAction } from '@workbench/workbenchState.testing';

import {
  applyCanvasProjectMutation,
  isCanvasProjectMutation,
  type CanvasProjectMutation,
} from '@workbench/canvasProjectMutations';

type EngineTestAction = WorkbenchAction | CanvasProjectMutation;

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import {
  createCanvasEngine as createApplicationCanvasEngine,
  type CanvasEngineOptions,
} from '@workbench/canvas-operations/createCanvasEngine';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { StrokeCommittedEvent } from './tools/tool';

interface EngineStore {
  dispatch(action: EngineTestAction): void;
  getState(): WorkbenchState;
  subscribe(listener: () => void): () => void;
}

const createMutationPort = (store: EngineStore, projectId: string): CanvasProjectMutationPort => ({
  commitEdit: () => undefined,
  dispatch: (mutation) => {
    const before = store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null;
    if (!before) {
      return false;
    }
    store.dispatch(mutation);
    return store.getState().projects.find((project) => project.id === projectId)?.canvas !== before;
  },
  getCanvasState: () => store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null,
  subscribe: store.subscribe,
});

const createCanvasEngine = ({
  projectId,
  store,
  ...options
}: Omit<CanvasEngineOptions, 'ensureProjectOnServer' | 'mutationPort' | 'reportError'> & { store: EngineStore }) =>
  createApplicationCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    ...options,
    mutationPort: createMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });

vi.mock('@workbench/canvas-operations/backend/canvasImages', () => ({
  CanvasImageUploadError: class extends Error {},
  uploadCanvasImage: vi.fn(() => Promise.resolve({ height: 64, imageName: 'mask-img', width: 64 })),
}));

const makeCanvas = (document: CanvasDocumentContractV3, documentRevision = 0): CanvasStateContractV3 =>
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
    version: 3,
  }) as CanvasStateContractV3;

const maskDoc = (): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom([
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
  ]),
  selectedLayerId: 'mask1',
  version: 3,
  width: 100,
});

const createReactiveStore = (document: CanvasDocumentContractV3) => {
  let state = {
    activeProjectId: 'p1',
    projects: [{ canvas: makeCanvas(document), id: 'p1' }],
  } as unknown as WorkbenchState;
  const listeners = new Set<() => void>();
  const dispatch = vi.fn((action: EngineTestAction) => {
    if (isCanvasProjectMutation(action)) {
      const project = state.projects[0] as unknown as Project;
      state = {
        ...state,
        projects: [applyCanvasProjectMutation(project, action)],
      } as WorkbenchState;
    }
    for (const listener of listeners) {
      listener();
    }
  });
  return {
    dispatch,
    setDocument: (next: CanvasDocumentContractV3, revision = 0) => {
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
    } as EngineStore,
  };
};

const createReducerBackedStore = (document: CanvasDocumentContractV3) => {
  let state = createInitialWorkbenchState();
  const projectId = state.activeProjectId;
  state = workbenchReducer(state, {
    mutation: { document, type: 'replaceCanvasDocument' },
    projectId,
    type: 'applyCanvasProjectMutation',
  });
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

const setupEngine = (doc: CanvasDocumentContractV3, options: { readbackAlpha?: number } = {}) => {
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

  const reactive = createReactiveStore(doc);
  const stubBackend = createTestStubRasterBackend(options);
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

  /** The image data most recently written into any surface with `putImageData`. */
  const lastWrittenPixels = (): unknown =>
    surfaces.flatMap((surface) => surface.callLog.filter((entry) => entry.op === 'putImageData')).at(-1)?.args[0];
  return {
    dispatch: reactive.dispatch,
    engine,
    lastWrittenPixels,
    overlay,
    raf,
    setDocument: reactive.setDocument,
    strokes,
  };
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
    getDocumentLeaves(doc)[0]!.isLocked = true;
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
    const { dispatch, engine, overlay } = setupEngine(maskDoc(), { readbackAlpha: 255 });
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointerup', pointerAt(50, 50, 0));

    await engine.lifecycle.flushPendingUploads();

    const configDispatch = dispatch.mock.calls
      .map((c) => c[0])
      .find(
        (a): a is Extract<EngineTestAction, { type: 'updateCanvasLayerConfig' }> =>
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

describe('inpaint mask shapes', () => {
  it('draws a shape into the selected mask as one undoable stroke that persists as mask alpha', async () => {
    const { dispatch, engine, lastWrittenPixels, overlay, strokes } = setupEngine(maskDoc(), { readbackAlpha: 255 });
    engine.tools.setTool('shape');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(60, 50));
    overlay.fire('pointerup', pointerAt(60, 50, 0));

    expect(strokes).toHaveLength(1);
    expect(strokes[0]).toMatchObject({ layerId: 'mask1', tool: 'shape' });
    const actions = () => dispatch.mock.calls.map((c) => c[0]);
    expect(actions().some((a) => a.type === 'addCanvasLayer')).toBe(false);
    expect(engine.history.getEntries().past).toEqual(['Draw shape']);

    // Undo puts the mask's pre-shape pixels back over the shape's rect; redo puts the shape's pixels back.
    expect(await engine.history.undo()).toBe('applied');
    expect(lastWrittenPixels()).toBe(strokes[0]!.beforeImageData);
    expect(await engine.history.redo()).toBe('applied');
    expect(lastWrittenPixels()).toBe(strokes[0]!.afterImageData);

    await engine.lifecycle.flushPendingUploads();
    const persisted = actions().filter(
      (a): a is Extract<EngineTestAction, { type: 'updateCanvasLayerConfig' }> =>
        a.type === 'updateCanvasLayerConfig' && a.id === 'mask1'
    );
    expect(persisted.at(-1)?.config).toMatchObject({
      layerType: 'inpaint_mask',
      mask: { bitmap: { imageName: 'mask-img' } },
    });
    expect(actions().some((a) => a.type === 'updateCanvasLayerSource')).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('accepting a result while a mask is selected', () => {
  it('keeps the mask selected through accept, undo and redo so the next stroke refines it', async () => {
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
    const reducer = createReducerBackedStore(maskDoc());
    const candidate = {
      height: 40,
      imageName: 'left-eye.png',
      imageUrl: '/left-eye.png',
      placement: { height: 40, opacity: 1, width: 40, x: 10, y: 10 },
      queuedAt: '2026-07-16T00:00:00.000Z',
      sourceQueueItemId: 'queue-eye',
      thumbnailUrl: '/left-eye-thumb.png',
      width: 40,
    };
    reducer.store.dispatch({ candidate, projectId: reducer.projectId, type: 'appendCanvasStagingCandidate' });
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      projectId: reducer.projectId,
      store: reducer.store,
    });
    const strokes: StrokeCommittedEvent[] = [];
    engine.tools.onStrokeCommitted((event) => strokes.push(event));
    const overlay = createInputCanvas();
    engine.surface.attach(createInputCanvas().element, overlay.element);
    raf.flush();
    const document = () => reducer.store.getState().projects[0]!.canvas.document;

    const accepted = engine.layers.commitStagedImage({ candidate, selectedImageIndex: 0 });
    expect(accepted.status).toBe('committed');
    const layerId = accepted.status === 'committed' ? accepted.layerId : '';
    expect(document().stacks.raster[0]).toMatchObject({ id: layerId, isEnabled: true });
    expect(document().selectedLayerId).toBe('mask1');

    // The other eye, straight away: the brush lands on the mask, not on the accepted image.
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(60, 20));
    overlay.fire('pointermove', pointerAt(80, 30));
    overlay.fire('pointerup', pointerAt(80, 30, 0));
    expect(strokes.map((stroke) => stroke.layerId)).toEqual(['mask1']);

    expect(await engine.history.undo()).toBe('applied');
    expect(await engine.history.undo()).toBe('applied');
    expect(document().stacks.raster).toEqual([]);
    expect(document().selectedLayerId).toBe('mask1');
    expect(await engine.history.redo()).toBe('applied');
    expect(document().stacks.raster[0]?.id).toBe(layerId);
    expect(document().selectedLayerId).toBe('mask1');
    engine.lifecycle.dispose();
  });
});

describe('mask invert', () => {
  it('inverts a mask as one undoable step', async () => {
    const { engine } = setupEngine(maskDoc());
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.layers.invertMask('mask1')).toEqual({ status: 'committed' });
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(await engine.history.undo()).toBe('applied');
    expect(engine.stores.canRedo.get()).toBe(true);
    expect(await engine.history.redo()).toBe('applied');
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('reports a missing layer and a locked mask distinctly without recording history', () => {
    const lockedDoc = maskDoc();
    getDocumentLeaves(lockedDoc)[0]!.isLocked = true;
    const { engine } = setupEngine(lockedDoc);
    expect(engine.layers.invertMask('nope')).toEqual({ status: 'missing' });
    expect(engine.layers.invertMask('mask1')).toEqual({ status: 'locked' });
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  // Invert must include live cache bounds: unflushed strokes outside the bbox are absent from persisted
  // `mask.bitmap` bounds.
  it('unions the live (unflushed) cache rect into the invert domain, covering an out-of-bbox stroke', () => {
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
    // Paint beyond the 100x100 bbox without flushing, leaving a grown live cache but null persisted bitmap.
    overlay.fire('pointerdown', pointerAt(150, 150));
    overlay.fire('pointermove', pointerAt(180, 180));
    overlay.fire('pointerup', pointerAt(180, 180, 0));

    expect(engine.layers.invertMask('mask1')).toEqual({ status: 'committed' });

    // Growth replaces backing surfaces. Inspect the last surface receiving `putImageData`, which owns the final
    // invert extent.
    const putSurfaces = surfaces.filter((surface) => surface.callLog.some((entry) => entry.op === 'putImageData'));
    const maskCache = putSurfaces[putSurfaces.length - 1];
    expect(maskCache).toBeDefined();
    const getCalls = maskCache!.callLog.filter((entry) => entry.op === 'getImageData');
    expect(getCalls.length).toBeGreaterThan(0);
    const [sx, sy, sw, sh] = getCalls[getCalls.length - 1]!.args as [number, number, number, number];
    // The invert domain must cover the unflushed stroke at (150,150)-(180,180), beyond persisted/bbox bounds
    // ending at 100.
    expect(sx + sw).toBeGreaterThan(110);
    expect(sy + sh).toBeGreaterThan(110);

    engine.lifecycle.dispose();
  });

  it('extracts through a live unflushed mask whose contract bitmap is still null', async () => {
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
    const doc = maskDoc();
    doc.stacks.raster.push({
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
  it('clears a live unflushed inpaint mask and restores it through undo', async () => {
    const { engine, overlay } = setupEngine(maskDoc());
    engine.tools.setTool('brush');
    overlay.fire('pointerdown', pointerAt(20, 20));
    overlay.fire('pointermove', pointerAt(50, 50));
    overlay.fire('pointerup', pointerAt(50, 50, 0));

    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'committed' });
    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'nothing' });
    expect(engine.stores.canUndo.get()).toBe(true);

    expect(await engine.history.undo()).toBe('applied');
    expect(engine.stores.canRedo.get()).toBe(true);
    expect(await engine.history.redo()).toBe('applied');
    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'nothing' });
    expect(await engine.history.undo()).toBe('applied');
    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'committed' });
    engine.lifecycle.dispose();
  });

  it('clears a regional-guidance mask', () => {
    const doc = maskDoc();
    doc.stacks.inpaint_mask[0] = {
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

    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'committed' });
    engine.lifecycle.dispose();
  });

  it('clears a cold hidden persisted mask and restores its bitmap reference on undo', async () => {
    const doc = maskDoc();
    const mask = getDocumentLeaves(doc)[0]!;
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

    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'committed' });
    expect(dispatch.mock.calls.at(-1)?.[0]).toMatchObject({
      config: { mask: { bitmap: null } },
      id: 'mask1',
      type: 'updateCanvasLayerConfig',
    });

    expect(await engine.history.undo()).toBe('applied');
    expect(dispatch.mock.calls.at(-1)?.[0]).toMatchObject({
      config: {
        mask: { bitmap: { imageName: 'persisted-mask' }, offset: { x: 7, y: 9 } },
      },
      id: 'mask1',
      type: 'updateCanvasLayerConfig',
    });
    engine.lifecycle.dispose();
  });

  it('reports missing, non-mask, locked, and empty layers distinctly', () => {
    const locked = maskDoc();
    (locked.stacks.inpaint_mask[0] as CanvasLayerContract).isLocked = true;
    const { engine } = setupEngine(locked);

    expect(engine.layers.clearMask('missing')).toEqual({ status: 'missing' });
    expect(engine.layers.clearMask('mask1')).toEqual({ status: 'locked' });
    engine.lifecycle.dispose();

    const raster = maskDoc();
    raster.stacks.inpaint_mask.length = 0;
    raster.stacks.raster[0] = {
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
    expect(rasterEngine.layers.clearMask('mask1')).toEqual({ status: 'unsupported' });
    rasterEngine.lifecycle.dispose();

    const { engine: emptyEngine } = setupEngine(maskDoc());
    expect(emptyEngine.layers.clearMask('mask1')).toEqual({ status: 'nothing' });
    emptyEngine.lifecycle.dispose();
  });
});
