import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project } from '@workbench/projectContracts';

import { createEmptyCanvasState } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { CanvasControlLayerContract, CanvasDocumentContractV3, CanvasPixelSourceContract } from './contracts';
import type * as RasterControllerModule from './controllers/rasterController';
import type { RasterController } from './controllers/rasterController';
import type { BitmapStore } from './document/bitmapStore';
import type { RasterCallLogEntry, StubRasterSurface } from './render/raster.testStub';

import { documentFrom, layerContract } from './document-model/documentFixtures.testStub';
import { createCanvasEngine } from './engine';
import { createTestStubRasterBackend } from './render/raster.testStub';

const owners = vi.hoisted(() => ({ raster: null as RasterController | null }));

// Captures the engine's real raster owner to read its reservations; its behavior is unchanged.
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

const DEPTH = { settings: {}, type: 'depth_anything_depth_estimation' };
// Sizes no other surface in these tests shares, so the original's surface is recognizable.
const PHOTO = { height: 32, imageName: 'photo', width: 48 };
const MAP = { height: 10, imageName: 'map', width: 10 };
const PHOTO_BYTES = PHOTO.width * PHOTO.height * 4;

const controlLayer = (overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  controlType: 'depth',
  detection: DEPTH,
  filter: DEPTH,
  id: 'control',
  isEnabled: true,
  isLocked: false,
  name: 'Control Layer 1',
  opacity: 1,
  original: { image: PHOTO, type: 'image' },
  source: { bitmap: MAP, offset: { x: 0, y: 0 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: false,
  ...overrides,
});

const bitmapStore: BitmapStore = {
  discardLayer: () => undefined,
  dispose: () => undefined,
  flushPendingUploads: () => Promise.resolve(),
  hasPendingClear: () => false,
  hasPendingWork: () => false,
  isSelfEcho: () => false,
  markLayerDirty: () => undefined,
  reset: () => undefined,
  suspendLayer: () => () => undefined,
};

const createHarness = (layer: CanvasControlLayerContract = controlLayer()) => {
  const document: CanvasDocumentContractV3 = documentFrom([layer, layerContract('raster')], layer.id);
  const initial = createInitialWorkbenchState().projects[0]!;
  let project: Project = { ...initial, canvas: { ...createEmptyCanvasState(), document } };
  let projectOpen = true;
  const listeners = new Set<() => void>();
  const notify = (): void => {
    for (const listener of listeners) {
      listener();
    }
  };
  const apply = (mutation: Parameters<CanvasProjectMutationPort['dispatch']>[0]): boolean => {
    const next = applyCanvasProjectMutation(project, mutation);
    if (next.canvas === project.canvas) {
      return false;
    }
    project = next;
    notify();
    return true;
  };
  // Image fetches wait until the test settles them, so a request can be superseded or fail mid-decode.
  const pendingImages: { resolve: () => void; reject: () => void }[] = [];
  const resolveImage = vi.fn(
    (_name: string) =>
      new Promise<Blob>((resolve, reject) => {
        pendingImages.push({ reject: () => reject(new Error('unreachable')), resolve: () => resolve(new Blob()) });
      })
  );
  const surfaces: StubRasterSurface[] = [];
  const stub = createTestStubRasterBackend();
  const { engine } = createCanvasEngine({
    backend: {
      ...stub,
      createSurface: (width, height) => {
        const surface = stub.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    },
    bitmapStore,
    fonts: null,
    imageResolver: resolveImage,
    mutationPort: {
      commitEdit: () => undefined,
      dispatch: apply,
      getCanvasState: () => (projectOpen ? project.canvas : null),
      subscribe: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    projectId: project.id,
    reportError: () => undefined,
    uploadImage: () => Promise.resolve({ height: 1, imageName: 'image', width: 1 }),
    uploadIntermediateImage: () => Promise.resolve({ height: 1, imageName: 'intermediate', width: 1 }),
  });
  const raster = owners.raster!;
  const settleImages = async (outcome: 'resolve' | 'reject'): Promise<void> => {
    for (let round = 0; round < 10; round += 1) {
      pendingImages.splice(0).forEach((image) => image[outcome]());
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    }
  };
  /** Lets every waiting image fetch finish, then the decodes behind them. */
  const releaseImages = () => settleImages('resolve');
  const show = async (view: 'original' | 'overlay' | 'map', layerId = 'control') => {
    const outcome = engine.previews.setControlLayerView(layerId, view);
    await releaseImages();
    return outcome;
  };
  const live = () =>
    project.canvas.document.stacks.control.find((node) => node.id === 'control') as CanvasControlLayerContract;
  return {
    apply,
    canvasDocument: () => project.canvas.document,
    engine,
    failImages: () => settleImages('reject'),
    /** What the view holds: decoding reserves, a shown original is retained as detached pixels. */
    held: () => {
      const { detachedBytes, reservedBytes } = raster.memory.snapshot();
      return { detached: detachedBytes, reserved: reservedBytes };
    },
    leaveProject: () => {
      projectOpen = false;
      notify();
    },
    live,
    mapCanvas: () => raster.layers.peek('control')?.surface.canvas,
    originalSurfaces: () => surfaces.filter((s) => s.width === PHOTO.width && s.height === PHOTO.height),
    releaseImages,
    resolveImage,
    show,
    view: () => engine.interaction.get('controlLayerView'),
  };
};

type Harness = ReturnType<typeof createHarness>;

const NOTHING_HELD = { detached: 0, reserved: 0 };
const ORIGINAL_HELD = { detached: PHOTO_BYTES, reserved: 0 };

/** A canvas element whose recorded pointer listeners `fire` invokes. */
const fakeCanvas = () => {
  const surface = createTestStubRasterBackend().createSurface(100, 100);
  const listeners = new Map<string, Set<(event: Event) => void>>();
  const element = {
    addEventListener: (type: string, handler: (event: Event) => void) => {
      listeners.set(type, (listeners.get(type) ?? new Set()).add(handler));
    },
    getBoundingClientRect: () => ({ bottom: 100, height: 100, left: 0, right: 100, top: 0, width: 100, x: 0, y: 0 }),
    getContext: () => surface.ctx,
    height: 100,
    releasePointerCapture: () => undefined,
    removeEventListener: (type: string, handler: (event: Event) => void) => {
      listeners.get(type)?.delete(handler);
    },
    setPointerCapture: () => undefined,
    width: 100,
  } as unknown as HTMLCanvasElement;
  const fire = (type: string, x: number, y: number): void => {
    const event = {
      altKey: false,
      button: 0,
      buttons: 1,
      clientX: x,
      clientY: y,
      ctrlKey: false,
      metaKey: false,
      pointerId: 1,
      pointerType: 'mouse',
      preventDefault: () => undefined,
      pressure: 0.5,
      shiftKey: false,
      timeStamp: 0,
    } as unknown as Event;
    for (const handler of listeners.get(type) ?? []) {
      handler(event);
    }
  };
  return { element, fire, surface };
};

const draws = (log: RasterCallLogEntry[]) => log.filter((entry) => entry.op === 'drawImage').map((entry) => entry.args);

/** Attaches recording canvases; `frame()` renders the queued frame and returns its `drawImage` arguments. */
const attachScreen = async (h: Harness) => {
  const frames: FrameRequestCallback[] = [];
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => frames.push(callback));
  vi.stubGlobal('cancelAnimationFrame', () => undefined);
  const screen = fakeCanvas();
  const overlay = fakeCanvas();
  h.engine.surface.attach(screen.element, overlay.element);
  const frame = () => {
    screen.surface.callLog.length = 0;
    frames.splice(0).forEach((callback) => callback(0));
    return draws(screen.surface.callLog);
  };
  const sources = () => frame().map((args) => args[0]);
  frame();
  await h.releaseImages();
  frame();
  return { frame, overlay, sources };
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('setControlLayerView', () => {
  it('shows an image original, switches to overlay without decoding again, and ends on map', async () => {
    const h = createHarness();

    const shown = h.engine.previews.setControlLayerView('control', 'original');
    // Decoding holds a reservation for the surface it is about to allocate.
    expect(h.held()).toEqual({ detached: 0, reserved: PHOTO_BYTES });
    await h.releaseImages();
    await expect(shown).resolves.toBe('shown');
    expect(h.view()).toEqual({ layerId: 'control', view: 'original' });
    expect(h.held()).toEqual(ORIGINAL_HELD);
    expect(h.originalSurfaces()).toHaveLength(1);

    const fetches = h.resolveImage.mock.calls.length;
    await expect(h.show('overlay')).resolves.toBe('shown');
    expect(h.view()).toEqual({ layerId: 'control', view: 'overlay' });
    expect(h.resolveImage).toHaveBeenCalledTimes(fetches);
    expect(h.held()).toEqual(ORIGINAL_HELD);

    await expect(h.show('map')).resolves.toBe('shown');
    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
  });

  it('places a painted original at its offset', async () => {
    const original: CanvasPixelSourceContract = { bitmap: PHOTO, offset: { x: -4, y: 2 }, type: 'paint' };
    const h = createHarness(controlLayer({ original }));
    const { frame } = await attachScreen(h);

    await expect(h.show('original')).resolves.toBe('shown');

    const [surface] = h.originalSurfaces();
    expect(
      frame()
        .find((args) => args[0] === surface!.canvas)
        ?.slice(1)
    ).toEqual([-4, 2]);
  });

  it.each([
    ['a control layer without an original', 'control'],
    ['a raster layer', 'raster'],
    ['an unknown layer', 'gone'],
  ])('reports %s as missing', async (_case, layerId) => {
    const untyped = controlLayer();
    delete untyped.original;
    const h = createHarness(untyped);

    await expect(h.show('original', layerId)).resolves.toBe('missing');
    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
  });

  it('reports an original that cannot be decoded as failed', async () => {
    const h = createHarness();

    const shown = h.engine.previews.setControlLayerView('control', 'original');
    await h.failImages();

    await expect(shown).resolves.toBe('failed');
    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
  });

  it.each<[string, (h: Harness) => void]>([
    ['an operation holds the document', (h) => h.engine.stores.documentEditingLocked.set(true)],
    ['a generation locks the canvas', (h) => h.engine.tools.setInteractionLocked(true)],
  ])('refuses a view while %s', async (_case, lock) => {
    const h = createHarness();
    lock(h);

    await expect(h.show('original')).resolves.toBe('locked');
    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
    expect(h.resolveImage).not.toHaveBeenCalledWith('photo', expect.anything());
  });

  it('drops a decode a newer request superseded', async () => {
    const h = createHarness();

    const first = h.engine.previews.setControlLayerView('control', 'original');
    const second = h.engine.previews.setControlLayerView('control', 'overlay');
    await h.releaseImages();

    await expect(first).resolves.toBe('stale');
    await expect(second).resolves.toBe('shown');
    expect(h.view()).toEqual({ layerId: 'control', view: 'overlay' });
    expect(h.held()).toEqual(ORIGINAL_HELD);
  });

  it.each<[string, (h: Harness) => Promise<unknown> | unknown]>([
    [
      'an edit of the layer',
      (h) =>
        h.apply({
          config: { adapter: { weight: 0.5 }, layerType: 'control' },
          id: 'control',
          type: 'updateCanvasLayerConfig',
        }),
    ],
    ['a map request', (h) => h.engine.previews.setControlLayerView('control', 'map')],
    ['selecting another layer', (h) => h.apply({ id: 'raster', type: 'setCanvasSelectedLayer' })],
    ['cooldown', (h) => h.engine.lifecycle.beginCooldown()],
    ['disposal', (h) => h.engine.lifecycle.dispose()],
  ])('drops a decode in flight on %s', async (_case, act) => {
    const h = createHarness();

    const pending = h.engine.previews.setControlLayerView('control', 'original');
    await act(h);
    await h.releaseImages();

    await expect(pending).resolves.toBe('stale');
    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
  });

  it('refuses an original the raster budget cannot hold, before decoding it', async () => {
    const huge = { height: 12_000, imageName: 'huge', width: 12_000 };
    const h = createHarness(controlLayer({ original: { image: huge, type: 'image' } }));

    await expect(h.show('original')).resolves.toBe('over-budget');
    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
    expect(h.resolveImage).not.toHaveBeenCalledWith('huge', expect.anything());
  });

  it('never records history or touches the document', async () => {
    const h = createHarness();
    const entries = h.engine.history.getEntries();
    const document = h.canvasDocument();

    await h.show('original');
    await h.show('overlay');
    await h.show('map');

    expect(h.engine.history.getEntries()).toEqual(entries);
    expect(h.canvasDocument()).toBe(document);
  });

  it.each<[string, (h: Harness) => Promise<unknown> | unknown]>([
    [
      'an edit of the layer',
      (h) =>
        h.engine.controlDerivation.commitControlLayerPatch({
          expected: h.live(),
          label: 'Weight',
          next: { ...h.live(), adapter: { ...h.live().adapter, weight: 0.5 } },
        }),
    ],
    [
      'an opacity change of the layer',
      (h) => h.apply({ id: 'control', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' }),
    ],
    ['undo', (h) => h.engine.history.undo()],
    [
      'redo',
      async (h) => {
        await h.engine.history.undo();
        await h.show('original');
        expect(h.view()).not.toBeNull();
        await h.engine.history.redo();
      },
    ],
    [
      'a detection map commit',
      async (h) => {
        const guard = h.engine.controlDerivation.captureControlDetectionGuard('control')!;
        const committed = h.engine.controlDerivation.commitControlMap({
          detection: DEPTH,
          guard,
          image: { height: 10, imageName: 'map-2', width: 10 },
          makeDurable: () => Promise.resolve(),
          rect: { height: 10, width: 10, x: 0, y: 0 },
        });
        await h.releaseImages();
        expect(await committed).toEqual({ status: 'committed' });
      },
    ],
    ['selecting another layer', (h) => h.apply({ id: 'raster', type: 'setCanvasSelectedLayer' })],
    ['removing the layer', (h) => h.apply({ ids: ['control'], type: 'removeCanvasLayers' })],
    [
      'replacing the document',
      (h) => h.apply({ document: documentFrom([controlLayer()], 'control'), type: 'replaceCanvasDocument' }),
    ],
    ['leaving the project', (h) => h.leaveProject()],
    ['cooldown', (h) => h.engine.lifecycle.beginCooldown()],
    ['a cache clear', (h) => h.engine.diagnostics.clearCaches()],
    ['an operation taking the document', (h) => h.engine.stores.documentEditingLocked.set(true)],
    ['a generation locking the canvas', (h) => h.engine.tools.setInteractionLocked(true)],
    ['disposal', (h) => h.engine.lifecycle.dispose()],
  ])('returns to the map on %s', async (_case, act) => {
    const h = createHarness();
    // Undo needs a step to revert; recording it before the view leaves the view on the current layer.
    h.engine.controlDerivation.commitControlLayerPatch({
      expected: h.live(),
      label: 'Name',
      next: { ...h.live(), name: 'Depth' },
    });
    await h.show('original');
    expect(h.view()).not.toBeNull();

    await act(h);

    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
  });

  it('keeps the view while another layer is edited', async () => {
    const h = createHarness();
    await h.show('original');

    h.apply({ id: 'raster', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });

    expect(h.view()).toEqual({ layerId: 'control', view: 'original' });
  });

  it('draws the original on screen while export and thumbnails keep reading the map', async () => {
    const h = createHarness();
    const { sources } = await attachScreen(h);

    await h.show('original');
    const [original] = h.originalSurfaces();
    expect(sources()).toContain(original!.canvas);

    const exported = h.engine.exports.exportBakedLayerBlob('control');
    await h.releaseImages();
    const result = await exported;
    expect(result.status === 'ok' && (await result.blob.text())).toBe('stub-surface-10x10');

    const thumbnail = fakeCanvas();
    expect(h.engine.previews.drawLayerThumbnail('control', thumbnail.element, 64)).toBe(true);
    expect(draws(thumbnail.surface.callLog).map((args) => args[0])).not.toContain(original!.canvas);
  });

  it('returns to the map when an operation takes the document, and stays there after it ends', async () => {
    const h = createHarness();
    const { sources } = await attachScreen(h);
    await h.show('original');
    const [original] = h.originalSurfaces();
    expect(sources()).not.toContain(h.mapCanvas());

    // What the operation controller publishes when Filter or Select Object acquires the edit lease.
    h.engine.stores.documentEditingLocked.set(true);

    expect(h.view()).toBeNull();
    const locked = sources();
    expect(locked).toContain(h.mapCanvas());
    expect(locked).not.toContain(original!.canvas);

    h.engine.stores.documentEditingLocked.set(false);
    await h.releaseImages();

    expect(h.view()).toBeNull();
    expect(sources()).not.toContain(original!.canvas);
  });

  it('returns to the map as a stroke starts on the layer, so the stroke paints visibly', async () => {
    vi.stubGlobal(
      'Path2D',
      class {
        closePath() {}
        lineTo() {}
        moveTo() {}
        quadraticCurveTo() {}
      }
    );
    const h = createHarness();
    const { overlay, sources } = await attachScreen(h);
    await h.show('original');
    expect(sources()).not.toContain(h.mapCanvas());
    h.engine.tools.setTool('brush');

    overlay.fire('pointerdown', 5, 5);

    expect(h.view()).toBeNull();
    expect(h.held()).toEqual(NOTHING_HELD);
    expect(sources()).toContain(h.mapCanvas());
  });
});
