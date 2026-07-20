import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';

import { createBitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import {
  createCompositeDedupeCache,
  executeControlComposite,
  executeRegionalMaskComposite,
} from '@workbench/canvas-operations/compositeForGeneration';
import {
  planControlComposites,
  planRegionalMaskComposites,
} from '@workbench/canvas-operations/generationCompositePlan';
import { createEmptyCanvasDocumentV2 } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import type { EngineDeps, RegistryTimers } from './engineRegistry';

import { createEngineRegistry } from './engineRegistry';

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const drainUntil = async (predicate: () => boolean, maxTicks = 50): Promise<void> => {
  for (let tick = 0; tick < maxTicks && !predicate(); tick += 1) {
    await Promise.resolve();
  }
};

const createFakeDeps = (): EngineDeps => {
  let project = createInitialWorkbenchState().projects[0]!;
  const listeners = new Set<() => void>();
  return {
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: {
      dispatch: (mutation) => {
        const before = project.canvas;
        project = applyCanvasProjectMutation(project, mutation);
        for (const listener of listeners) {
          listener();
        }
        return project.canvas !== before;
      },
      getCanvasState: () => project.canvas,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    reportError: vi.fn(),
  };
};

const controlLayer = (id: string): CanvasControlLayerContract => ({
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: null,
    weight: 0.75,
  },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 64, imageName: `${id}.png`, width: 64 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const regionalLayer = (id: string): CanvasRegionalGuidanceLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: {
    bitmap: { height: 64, imageName: `${id}-mask.png`, width: 64 },
    fill: { color: '#ff0000', style: 'solid' },
  },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: 'regional prompt',
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

const invocationDocument = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 64, width: 64, x: 0, y: 0 },
  height: 64,
  layers: [controlLayer('control-a'), regionalLayer('regional-a')],
  selectedLayerId: 'control-a',
  version: 2,
  width: 64,
});

const createFakeTimers = (): { timers: RegistryTimers; flush: () => void; pending: () => number } => {
  let nextHandle = 0;
  const scheduled = new Map<number, () => void>();
  return {
    flush: () => {
      const callbacks = [...scheduled.values()];
      scheduled.clear();
      for (const callback of callbacks) {
        callback();
      }
    },
    pending: () => scheduled.size,
    timers: {
      clearTimeout: (handle) => {
        scheduled.delete(handle);
      },
      setTimeout: (handler) => {
        nextHandle += 1;
        scheduled.set(nextHandle, handler);
        return nextHandle;
      },
    },
  };
};

describe('createEngineRegistry', () => {
  it('returns the same instance per project id and distinct instances across ids', () => {
    const registry = createEngineRegistry();
    const deps = createFakeDeps();

    const first = registry.getOrCreateEngine('p1', deps);
    const again = registry.getOrCreateEngine('p1', deps);
    const other = registry.getOrCreateEngine('p2', deps);

    expect(again).toBe(first);
    expect(other).not.toBe(first);
    expect(registry.getEngine('p1')).toBe(first);

    first.lifecycle.dispose();
    other.lifecycle.dispose();
  });

  it('disposes after the grace period once the last reference is released', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ gracePeriodMs: 30_000, timers: fakeTimers.timers });
    const engine = registry.getOrCreateEngine('p1', createFakeDeps());
    const disposeSpy = vi.spyOn(engine.lifecycle, 'dispose');
    const cooldownSpy = vi.spyOn(engine.lifecycle, 'beginCooldown');

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(1);
    expect(cooldownSpy).toHaveBeenCalledOnce();
    expect(disposeSpy).not.toHaveBeenCalled();

    fakeTimers.flush();
    await Promise.resolve();
    await Promise.resolve();
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(registry.getEngine('p1')).toBeUndefined();
  });

  it('retains a dirty engine and retries cooldown before disposing it', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ gracePeriodMs: 30_000, timers: fakeTimers.timers });
    const engine = registry.getOrCreateEngine('p1', createFakeDeps());
    const disposeSpy = vi.spyOn(engine.lifecycle, 'dispose');
    const cooldownSpy = vi
      .spyOn(engine.lifecycle, 'beginCooldown')
      .mockResolvedValueOnce('dirty')
      .mockResolvedValueOnce('cooled');

    registry.releaseEngine('p1');
    fakeTimers.flush();
    await Promise.resolve();
    await Promise.resolve();

    expect(registry.getEngine('p1')).toBe(engine);
    expect(disposeSpy).not.toHaveBeenCalled();
    expect(cooldownSpy).toHaveBeenCalledTimes(2);
    expect(fakeTimers.pending()).toBe(1);

    fakeTimers.flush();
    await Promise.resolve();
    await Promise.resolve();

    expect(registry.getEngine('p1')).toBeUndefined();
    expect(disposeSpy).toHaveBeenCalledOnce();
  });

  it('cancels the pending disposal when the engine is re-acquired', () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ timers: fakeTimers.timers });
    const engine = registry.getOrCreateEngine('p1', createFakeDeps());
    const disposeSpy = vi.spyOn(engine.lifecycle, 'dispose');
    const activateSpy = vi.spyOn(engine.lifecycle, 'activate');

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(1);

    const reacquired = registry.getOrCreateEngine('p1', createFakeDeps());
    expect(reacquired).toBe(engine);
    expect(fakeTimers.pending()).toBe(0);
    expect(activateSpy).toHaveBeenCalledOnce();

    fakeTimers.flush();
    expect(disposeSpy).not.toHaveBeenCalled();

    engine.lifecycle.dispose();
  });

  it('continues control and regional composites from a detached snapshot after a project switch', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ timers: fakeTimers.timers });
    const projectADeps = createFakeDeps();
    projectADeps.mutationPort.dispatch({ document: invocationDocument(), type: 'replaceCanvasDocument' });
    const projectAEngine = registry.getOrCreateEngine('project-a', projectADeps);
    const documentSnapshot = projectAEngine.document.captureSnapshot();
    expect(documentSnapshot).not.toBeNull();

    const capture = await projectAEngine.exports.captureRasterSnapshot(documentSnapshot!, ['control-a', 'regional-a']);
    expect(capture.status).toBe('ok');
    if (capture.status !== 'ok') {
      throw new Error('Expected a detached raster snapshot');
    }

    registry.releaseEngine('project-a');
    const projectBEngine = registry.getOrCreateEngine('project-b', createFakeDeps());
    await drainUntil(() => projectAEngine.lifecycle.getLifecycleState() === 'cool');

    expect(projectAEngine.lifecycle.getLifecycleState()).toBe('cool');
    expect(registry.getEngine('project-a')).toBe(projectAEngine);
    expect(registry.getEngine('project-b')).toBe(projectBEngine);
    expect(capture.snapshot.layerSurfaces.size).toBe(2);

    const frozenDocument = capture.snapshot.canvas.document;
    const controlPlan = planControlComposites(frozenDocument, frozenDocument.bbox)[0];
    const regionalPlan = planRegionalMaskComposites(frozenDocument, frozenDocument.bbox)[0];
    expect(controlPlan).toBeDefined();
    expect(regionalPlan).toBeDefined();

    const surfaceRequests: string[] = [];
    const uploadImage = vi.fn((blob: Blob) => {
      void blob;
      return Promise.resolve({ height: 64, imageName: `snapshot-${surfaceRequests.length}.png`, width: 64 });
    });
    const executorDeps = {
      ...projectAEngine.exports.getCompositeExecutorDeps(),
      dedupe: createCompositeDedupeCache(),
      getLayerSurface: (layerId: string) => {
        surfaceRequests.push(layerId);
        const detached = capture.snapshot.layerSurfaces.get(layerId);
        if (!detached) {
          throw new Error(`Detached snapshot is missing ${layerId}`);
        }
        return Promise.resolve(detached);
      },
      hashBlob: (blob: Blob) => blob.text(),
      uploadImage,
    };

    await executeControlComposite(controlPlan!.entry, executorDeps);
    await executeRegionalMaskComposite(regionalPlan!.entry, executorDeps);

    expect(surfaceRequests).toEqual(['control-a', 'regional-a']);
    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(capture.snapshot.layerSurfaces.size).toBe(2);

    capture.snapshot.release();
    registry.releaseEngine('project-b');
    fakeTimers.flush();
    await drainUntil(
      () => registry.getEngine('project-a') === undefined && registry.getEngine('project-b') === undefined
    );
  });

  it('keeps a pending bitmap upload alive when the engine reaches zero references and is reacquired', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ timers: fakeTimers.timers });
    let project = createInitialWorkbenchState().projects[0]!;
    project = applyCanvasProjectMutation(project, {
      document: createEmptyCanvasDocumentV2(),
      type: 'replaceCanvasDocument',
    });
    project = applyCanvasProjectMutation(project, {
      layer: {
        blendMode: 'normal',
        id: 'pending-layer',
        isEnabled: true,
        isLocked: false,
        name: 'Pending layer',
        opacity: 1,
        source: { bitmap: null, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'raster',
      },
      type: 'addCanvasLayer',
    });
    const listeners = new Set<() => void>();
    const mutationPort = {
      dispatch: (mutation: Parameters<typeof applyCanvasProjectMutation>[1]) => {
        const before = project.canvas;
        project = applyCanvasProjectMutation(project, mutation);
        for (const listener of listeners) {
          listener();
        }
        return project.canvas !== before;
      },
      getCanvasState: () => project.canvas,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };
    const uploaded = createDeferred<{ height: number; imageName: string; width: number }>();
    const uploadImage = vi.fn(() => uploaded.promise);
    const surface = createTestStubRasterBackend().createSurface(8, 8);
    const bitmapStore = createBitmapStore({
      debounceMs: 60_000,
      dispatch: mutationPort.dispatch,
      encodeSurface: () => Promise.resolve(new Blob(['pending-pixels'], { type: 'image/png' })),
      getLayerSource: () => {
        const layer = project.canvas.document.layers.find((candidate) => candidate.id === 'pending-layer');
        return layer?.type === 'raster' && layer.source.type === 'paint' ? layer.source : null;
      },
      getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
      hashBlob: () => Promise.resolve('pending-pixels'),
      uploadImage,
    });
    const deps: EngineDeps = {
      backend: createTestStubRasterBackend(),
      bitmapStore,
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort,
      reportError: vi.fn(),
    };
    const engine = registry.getOrCreateEngine(project.id, deps);
    bitmapStore.markLayerDirty('pending-layer');

    registry.releaseEngine(project.id);
    await drainUntil(() => uploadImage.mock.calls.length === 1);
    expect(fakeTimers.pending()).toBe(1);

    const reacquired = registry.getOrCreateEngine(project.id, deps);
    expect(reacquired).toBe(engine);
    expect(fakeTimers.pending()).toBe(0);

    uploaded.resolve({ height: 8, imageName: 'persisted-after-reacquire.png', width: 8 });
    await bitmapStore.flushPendingUploads();

    const layer = project.canvas.document.layers.find((candidate) => candidate.id === 'pending-layer');
    expect(layer?.type === 'raster' && layer.source.type === 'paint' ? layer.source.bitmap?.imageName : null).toBe(
      'persisted-after-reacquire.png'
    );
    engine.lifecycle.dispose();
  });

  it('only schedules disposal when the last reference is released', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ timers: fakeTimers.timers });
    const deps = createFakeDeps();

    registry.getOrCreateEngine('p1', deps);
    registry.getOrCreateEngine('p1', deps);

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(0);

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(1);

    fakeTimers.flush();
    await Promise.resolve();
    await Promise.resolve();
    expect(registry.getEngine('p1')).toBeUndefined();
  });
});
