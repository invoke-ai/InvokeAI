import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type { LayerFilterControllerDeps } from './layerFilterController';

import { createLayerFilterController } from './layerFilterController';

const layer: CanvasRasterLayerContractV2 = {
  blendMode: 'normal',
  id: 'L',
  isEnabled: true,
  isLocked: false,
  name: 'Layer',
  opacity: 1,
  source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};
const guard: LayerExportGuard = {
  cacheVersion: 1,
  documentGeneration: 1,
  layer,
  layerId: layer.id,
  projectId: 'p1',
};
const surface = createTestStubRasterBackend().createSurface(10, 10);
const exported = {
  guard,
  rect: { height: 10, width: 10, x: 3, y: 4 },
  status: 'ok' as const,
  surface,
};

const createDeferred = <T>(): { promise: Promise<T>; resolve(value: T): void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const result = (imageName: string) => ({ height: 10, imageName, origin: { x: 3, y: 4 }, width: 10 });

const createDeps = (overrides: Partial<LayerFilterControllerDeps> = {}) => {
  const deps: LayerFilterControllerDeps = {
    clearPreview: vi.fn(),
    commit: vi.fn(() => Promise.resolve({ layerId: layer.id, status: 'committed' as const })),
    exportPixels: vi.fn(() => Promise.resolve(exported)),
    makeDurable: vi.fn(() => Promise.resolve()),
    runFilter: vi.fn(() => Promise.resolve(result('filtered'))),
    showPreview: vi.fn(() => Promise.resolve('shown' as const)),
    ...overrides,
  };
  return deps;
};

describe('createLayerFilterController', () => {
  it('lets only the newest preview request publish', async () => {
    const first = createDeferred<ReturnType<typeof result>>();
    const second = createDeferred<ReturnType<typeof result>>();
    const runs = [first, second];
    const deps = createDeps({
      runFilter: vi.fn(() => runs.shift()!.promise),
    });
    const controller = createLayerFilterController(deps);

    const older = controller.preview('canny_edge_detection', {});
    await vi.waitFor(() => expect(deps.runFilter).toHaveBeenCalledTimes(1));
    const newer = controller.preview('canny_edge_detection', {});
    await vi.waitFor(() => expect(deps.runFilter).toHaveBeenCalledTimes(2));
    second.resolve(result('newer'));
    await newer;
    first.resolve(result('older'));
    await older;

    expect(deps.showPreview).toHaveBeenCalledOnce();
    expect(deps.showPreview).toHaveBeenCalledWith('newer', guard);
    expect(controller.getSnapshot().preview?.imageName).toBe('newer');
  });

  it('cancel aborts an in-flight preview and clears transient state', async () => {
    const filtered = createDeferred<ReturnType<typeof result>>();
    let signal: AbortSignal | undefined;
    const deps = createDeps({
      runFilter: vi.fn((options) => {
        signal = options.signal;
        return filtered.promise;
      }),
    });
    const controller = createLayerFilterController(deps);
    const pending = controller.preview('canny_edge_detection', {});
    await vi.waitFor(() => expect(signal).toBeDefined());

    controller.cancel();
    filtered.resolve(result('cancelled'));
    await pending;

    expect(signal?.aborted).toBe(true);
    expect(deps.showPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({ error: null, isRunning: false, preview: null });
    expect(deps.clearPreview).toHaveBeenCalled();
  });

  it('dispose aborts work, clears the preview, and stops notifications', async () => {
    const filtered = createDeferred<ReturnType<typeof result>>();
    let signal: AbortSignal | undefined;
    const deps = createDeps({
      runFilter: vi.fn((options) => {
        signal = options.signal;
        return filtered.promise;
      }),
    });
    const controller = createLayerFilterController(deps);
    const listener = vi.fn();
    controller.subscribe(listener);
    const pending = controller.preview('canny_edge_detection', {});
    await vi.waitFor(() => expect(signal).toBeDefined());
    listener.mockClear();

    controller.dispose();
    filtered.resolve(result('disposed'));
    await pending;

    expect(signal?.aborted).toBe(true);
    expect(deps.clearPreview).toHaveBeenCalled();
    expect(listener).not.toHaveBeenCalled();
  });

  it('keeps the preview and reports an actionable durability failure', async () => {
    const durabilityError = new Error('patch failed');
    const deps = createDeps({ makeDurable: vi.fn(() => Promise.reject(durabilityError)) });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    vi.mocked(deps.clearPreview).mockClear();

    await controller.commit('replace');

    expect(deps.commit).not.toHaveBeenCalled();
    expect(deps.clearPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({
      error: { key: 'durabilityFailure', message: 'patch failed' },
      isRunning: false,
      preview: { imageName: 'filtered' },
    });
  });

  it('reports a stale guarded preview without retaining a commit candidate', async () => {
    const deps = createDeps({ showPreview: vi.fn(() => Promise.resolve('stale' as const)) });
    const controller = createLayerFilterController(deps);

    await controller.preview('canny_edge_detection', {});

    expect(controller.getSnapshot()).toMatchObject({ error: { key: 'stale' }, isRunning: false, preview: null });
  });

  it('reports a graph failure without retaining a commit candidate', async () => {
    const deps = createDeps({ runFilter: vi.fn(() => Promise.reject(new Error('graph exploded'))) });
    const controller = createLayerFilterController(deps);

    await controller.preview('canny_edge_detection', {});

    expect(deps.showPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toEqual({
      error: { key: 'graphFailure', message: 'graph exploded' },
      isRunning: false,
      preview: null,
    });
  });

  it.each(['replace', 'copy'] as const)('makes durable, commits, and clears after %s success', async (mode) => {
    const calls: string[] = [];
    const deps = createDeps({
      clearPreview: vi.fn(() => calls.push('clear')),
      commit: vi.fn((options) => {
        calls.push(`commit:${options.mode}`);
        expect(options.signal.aborted).toBe(false);
        return Promise.resolve({ layerId: layer.id, status: 'committed' as const });
      }),
      makeDurable: vi.fn(() => {
        calls.push('durable');
        return Promise.resolve();
      }),
    });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    calls.length = 0;

    await controller.commit(mode);

    expect(calls).toEqual(['durable', `commit:${mode}`, 'clear']);
    expect(deps.commit).toHaveBeenCalledWith({
      guard,
      image: { height: 10, imageName: 'filtered', width: 10 },
      mode,
      rect: exported.rect,
      signal: expect.any(AbortSignal),
    });
    expect(controller.getSnapshot()).toMatchObject({ error: null, isRunning: false, preview: null });
  });

  it('routes an aborted commit without clearing the retained preview', async () => {
    const deps = createDeps({ commit: vi.fn(() => Promise.resolve({ status: 'aborted' as const })) });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    vi.mocked(deps.clearPreview).mockClear();

    await controller.commit('replace');

    expect(deps.clearPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({
      error: null,
      isRunning: false,
      preview: { imageName: 'filtered' },
    });
  });

  it('routes a locked commit to an actionable locked error while retaining the preview', async () => {
    const deps = createDeps({ commit: vi.fn(() => Promise.resolve({ status: 'locked' as const })) });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    vi.mocked(deps.clearPreview).mockClear();

    await controller.commit('replace');

    expect(deps.clearPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({
      error: { key: 'locked' },
      isRunning: false,
      preview: { imageName: 'filtered' },
    });
  });

  it('routes a failed commit message while retaining the preview', async () => {
    const deps = createDeps({
      commit: vi.fn(() => Promise.resolve({ message: 'cache install failed', status: 'failed' as const })),
    });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    vi.mocked(deps.clearPreview).mockClear();

    await controller.commit('copy');

    expect(deps.clearPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({
      error: { key: 'commitFailure', message: 'cache install failed' },
      isRunning: false,
      preview: { imageName: 'filtered' },
    });
  });

  it('routes a rejected commit message while retaining the preview', async () => {
    const deps = createDeps({ commit: vi.fn(() => Promise.reject(new Error('commit exploded'))) });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    vi.mocked(deps.clearPreview).mockClear();

    await controller.commit('replace');

    expect(deps.clearPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({
      error: { key: 'commitFailure', message: 'commit exploded' },
      isRunning: false,
      preview: { imageName: 'filtered' },
    });
  });

  it('cancel aborts a deferred commit and prevents its result from publishing', async () => {
    const committed = createDeferred<{ layerId: string; status: 'committed' }>();
    let signal: AbortSignal | undefined;
    const deps = createDeps({
      commit: vi.fn((options) => {
        signal = options.signal;
        return committed.promise;
      }),
    });
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});
    vi.mocked(deps.clearPreview).mockClear();
    const pending = controller.commit('replace');
    await vi.waitFor(() => expect(signal).toBeDefined());

    controller.cancel();
    committed.resolve({ layerId: layer.id, status: 'committed' });
    await pending;

    expect(signal?.aborted).toBe(true);
    expect(deps.clearPreview).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toEqual({ error: null, isRunning: false, preview: null });
  });

  it('remains reusable after callback-ref cleanup calls cancel', async () => {
    const deps = createDeps();
    const controller = createLayerFilterController(deps);
    await controller.preview('canny_edge_detection', {});

    controller.cancel();
    await controller.preview('content_shuffle', { scale_factor: 2 });

    expect(deps.runFilter).toHaveBeenCalledTimes(2);
    expect(deps.runFilter).toHaveBeenLastCalledWith({
      filterType: 'content_shuffle',
      input: { rect: exported.rect, surface: exported.surface },
      settings: { scale_factor: 2 },
      signal: expect.any(AbortSignal),
    });
    expect(controller.getSnapshot()).toMatchObject({
      error: null,
      isRunning: false,
      preview: { imageName: 'filtered' },
    });
  });
});
