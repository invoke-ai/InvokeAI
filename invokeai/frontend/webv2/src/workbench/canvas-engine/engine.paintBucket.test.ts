/**
 * The paint bucket through the whole engine: pointer pipeline, flood fill worker service, stroke edits and history.
 * The stub backend reads back one opaque color, so each fill takes its whole domain (the 100 px frame).
 */

import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type { FloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createInlineFloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService.testing';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import {
  applyCanvasProjectMutation,
  isCanvasProjectMutation,
  type CanvasProjectMutation,
} from '@workbench/canvasProjectMutations';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { StrokeCommittedEvent } from './tools/tool';

const makeCanvas = (document: CanvasDocumentContractV3): CanvasStateContractV3 =>
  ({
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
  }) as CanvasStateContractV3;

const paintLayer: CanvasLayerContract = {
  blendMode: 'normal',
  id: 'paint1',
  isEnabled: true,
  isLocked: false,
  name: 'Paint',
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};

const documentWith = (selectedLayerId: string | null): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  selectedLayerId,
  stacks: stacksFrom([paintLayer]),
  version: 3,
  width: 100,
});

/** A store that applies canvas mutations, so created layers, removals and replays reach the document. */
const createStore = (document: CanvasDocumentContractV3) => {
  let state = {
    activeProjectId: 'p1',
    projects: [{ canvas: makeCanvas(document), id: 'p1' }],
  } as unknown as WorkbenchState;
  const listeners = new Set<() => void>();
  const canvas = () => state.projects[0]!.canvas!;
  const mutationPort: CanvasProjectMutationPort = {
    commitEdit: () => undefined,
    dispatch: (mutation: CanvasProjectMutation) => {
      const before = canvas();
      if (isCanvasProjectMutation(mutation)) {
        state = {
          ...state,
          projects: [applyCanvasProjectMutation(state.projects[0] as unknown as Project, mutation)],
        } as WorkbenchState;
      }
      listeners.forEach((listener) => listener());
      return canvas() !== before;
    },
    getCanvasState: () => canvas(),
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return { document: () => canvas().document, mutationPort };
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
  const fire = (type: string, x: number, y: number, buttons: number) => {
    const event = {
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
      preventDefault: () => {},
      shiftKey: false,
      timeStamp: 0,
    };
    for (const handler of listeners.get(type) ?? []) {
      handler(event as unknown as Event);
    }
  };
  return { element, fire };
};

/** An inline executor whose jobs wait until `open`, so a test can act while a released fill settles. */
const createGatedExecutor = () => {
  const inner = createInlineFloodFillExecutor();
  let open: () => void = () => undefined;
  const gate = new Promise<void>((resolve) => {
    open = resolve;
  });
  const executor: FloodFillExecutor = {
    dispose: () => inner.dispose(),
    release: (pressId) => inner.release(pressId),
    run: (job) => gate.then(() => inner.run(job)),
  };
  return { executor, open: () => open() };
};

const settle = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

const setup = (
  selectedLayerId: string | null = 'paint1',
  createFloodFillExecutor: () => FloodFillExecutor = createInlineFloodFillExecutor
) => {
  const raf = { callbacks: new Map<number, FrameRequestCallback>(), next: 1 };
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    raf.callbacks.set(raf.next, callback);
    return raf.next++;
  });
  vi.stubGlobal('cancelAnimationFrame', (handle: number) => raf.callbacks.delete(handle));
  const store = createStore(documentWith(selectedLayerId));
  const base = createTestStubRasterBackend();
  const surfaces: StubRasterSurface[] = [];
  const engine = createCanvasEngine({
    backend: {
      ...base,
      createSurface: (width: number, height: number) => {
        const surface = base.createSurface(width, height);
        surfaces.push(surface);
        return surface;
      },
    },
    createFloodFillExecutor,
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: store.mutationPort,
    projectId: 'p1',
    reportError: () => undefined,
  });
  const strokes: StrokeCommittedEvent[] = [];
  engine.tools.onStrokeCommitted((event) => strokes.push(event));
  const overlay = createInputCanvas();
  engine.surface.attach(createInputCanvas().element, overlay.element);
  engine.tools.setTool('paintBucket');
  /** Every image written back into a surface with `putImageData`. */
  const writtenImages = (): unknown[] =>
    surfaces.flatMap((surface) => surface.callLog.filter((entry) => entry.op === 'putImageData')).map((e) => e.args[0]);
  return {
    click: () => {
      overlay.fire('pointerdown', 20, 20, 1);
      overlay.fire('pointerup', 20, 20, 0);
    },
    document: store.document,
    engine,
    overlay,
    strokes,
    writtenImages,
  };
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('paint bucket in the engine', () => {
  it('records one Bucket fill that undo and redo replay', async () => {
    const { click, engine, strokes, writtenImages } = setup();
    click();
    await settle();

    expect(strokes.map((stroke) => [stroke.layerId, stroke.tool])).toEqual([['paint1', 'paintBucket']]);
    expect(engine.history.getEntries().past).toEqual(['Bucket fill']);

    expect(await engine.history.undo()).toBe('applied');
    expect(writtenImages()).toContain(strokes[0]!.beforeImageData);
    expect(engine.history.getEntries().future).toEqual(['Bucket fill']);
    expect(await engine.history.redo()).toBe('applied');
    expect(writtenImages()).toContain(strokes[0]!.afterImageData);
    engine.lifecycle.dispose();
  });

  it('undoes a fill that created its layer by removing layer and pixels together', async () => {
    const { click, document, engine, strokes } = setup(null);
    click();
    await settle();

    const created = strokes[0]!.layerId;
    expect(strokes[0]!.createdLayer?.layer.id).toBe(created);
    expect(getDocumentLeaves(document()).map((leaf) => leaf.id)).toContain(created);
    expect(engine.history.getEntries().past).toEqual(['Bucket fill']);

    await engine.history.undo();
    expect(getDocumentLeaves(document()).map((leaf) => leaf.id)).not.toContain(created);
    await engine.history.redo();
    expect(getDocumentLeaves(document()).map((leaf) => leaf.id)).toContain(created);
    engine.lifecycle.dispose();
  });

  it('holds undo while a released fill settles, then records it', async () => {
    const gated = createGatedExecutor();
    const { click, engine } = setup('paint1', () => gated.executor);
    click();

    expect(await engine.history.undo()).toBe('refused');
    gated.open();
    await settle();
    expect(engine.history.getEntries().past).toEqual(['Bucket fill']);
    engine.lifecycle.dispose();
  });

  it('puts the layer back and records nothing when cooldown meets a held press', async () => {
    const { engine, overlay, strokes } = setup();
    overlay.fire('pointerdown', 20, 20, 1);
    overlay.fire('pointermove', 60, 20, 1);
    await settle();
    expect(engine.stores.paintBucketScrub.get()?.tolerance).toBe(30);

    void engine.lifecycle.beginCooldown();
    // Detaching cancelled the press: its readout is gone and a late release lands nowhere.
    expect(engine.stores.paintBucketScrub.get()).toBeNull();
    overlay.fire('pointerup', 60, 20, 0);
    await settle();
    expect(strokes).toHaveLength(0);
    expect(engine.history.getEntries().past).toEqual([]);
    engine.lifecycle.dispose();
  });

  it('ends a settling fill on cooldown without recording it', async () => {
    const gated = createGatedExecutor();
    const { click, engine, strokes } = setup('paint1', () => gated.executor);
    click();

    void engine.lifecycle.beginCooldown();
    gated.open();
    await settle();
    expect(strokes).toHaveLength(0);
    expect(engine.history.getEntries().past).toEqual([]);
    engine.lifecycle.dispose();
  });

  it.each(['held', 'settling'] as const)(
    'takes a layer a %s press created back out of the document before cooldown flushes',
    async (phase) => {
      const gated = createGatedExecutor();
      const { document, engine, overlay, strokes } = setup(null, () => gated.executor);
      const leavesBefore = getDocumentLeaves(document()).map((leaf) => leaf.id);
      overlay.fire('pointerdown', 20, 20, 1);
      if (phase === 'settling') {
        overlay.fire('pointerup', 20, 20, 0);
      }
      expect(getDocumentLeaves(document())).toHaveLength(leavesBefore.length + 1);

      void engine.lifecycle.beginCooldown();
      // Synchronously, before the flush that cooldown starts can read anything.
      expect(getDocumentLeaves(document()).map((leaf) => leaf.id)).toEqual(leavesBefore);
      gated.open();
      await settle();
      expect(strokes).toHaveLength(0);
      expect(engine.history.getEntries().past).toEqual([]);
      engine.lifecycle.dispose();
    }
  );

  it.each(['cooldown', 'dispose'] as const)(
    'puts back a settling fill on %s while a temporary hold has made another tool active',
    async (ending) => {
      const gated = createGatedExecutor();
      const { click, document, engine, strokes } = setup(null, () => gated.executor);
      const leavesBefore = getDocumentLeaves(document()).map((leaf) => leaf.id);
      click();
      // Alt's picker hold while the released fill waits on the worker.
      engine.tools.setTool('colorPicker', { temporary: true });
      expect(engine.stores.activeTool.get()).toBe('colorPicker');
      expect(getDocumentLeaves(document())).toHaveLength(leavesBefore.length + 1);

      if (ending === 'cooldown') {
        void engine.lifecycle.beginCooldown();
      } else {
        engine.lifecycle.dispose();
      }
      // Synchronously: the created layer and its preview are gone before any flush or teardown reads them.
      expect(getDocumentLeaves(document()).map((leaf) => leaf.id)).toEqual(leavesBefore);
      gated.open();
      await settle();
      expect(strokes).toHaveLength(0);
      if (ending === 'cooldown') {
        expect(engine.history.getEntries().past).toEqual([]);
        engine.lifecycle.dispose();
      }
    }
  );

  it('disposes cleanly with a held or settling press', async () => {
    const held = setup();
    held.overlay.fire('pointerdown', 20, 20, 1);
    await settle();
    expect(() => held.engine.lifecycle.dispose()).not.toThrow();
    expect(held.strokes).toHaveLength(0);

    const gated = createGatedExecutor();
    const settling = setup('paint1', () => gated.executor);
    settling.click();
    expect(() => settling.engine.lifecycle.dispose()).not.toThrow();
    gated.open();
    await settle();
    expect(settling.strokes).toHaveLength(0);
  });
});
