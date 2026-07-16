import type { ExportLayerPixelsResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasOperationController } from '@workbench/canvas-operations/operationController';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { FilterOperationSessionDeps } from './filterOperationSession';

import { createFilterOperationSession, FILTER_AUTO_PROCESS_DEBOUNCE_MS } from './filterOperationSession';

const layer: CanvasRasterLayerContractV2 = {
  blendMode: 'normal',
  filter: { settings: { radius: 2 }, type: 'canny_edge_detection' },
  id: 'layer-1',
  isEnabled: true,
  isLocked: false,
  name: 'Layer',
  opacity: 1,
  source: { image: { height: 10, imageName: 'source', width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};
const guard: LayerExportGuard = {
  cacheVersion: 1,
  documentGeneration: 1,
  layer,
  layerId: layer.id,
  projectId: 'project-1',
};
const surface = createTestStubRasterBackend().createSurface(10, 10);
const exported: ExportLayerPixelsResult = {
  guard,
  release: vi.fn(),
  rect: { height: 10, width: 10, x: 3, y: 4 },
  status: 'ok',
  surface,
};

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const createDeps = (overrides: Partial<FilterOperationSessionDeps> = {}): FilterOperationSessionDeps => ({
  canCommit: vi.fn(() => true),
  clearPreview: vi.fn(),
  commit: vi.fn(() => Promise.resolve({ layerId: layer.id, status: 'committed' as const })),
  controller: createCanvasOperationController({ isGuardCurrent: () => true }),
  exportPixels: vi.fn(() => Promise.resolve(exported)),
  isDraftValid: vi.fn(() => true),
  isGuardCurrent: vi.fn(() => true),
  makeDurable: vi.fn(() => Promise.resolve()),
  publishPreview: vi.fn(() => Promise.resolve('shown' as const)),
  runFilter: vi.fn(() => Promise.resolve({ height: 10, imageName: 'filtered', origin: { x: 3, y: 4 }, width: 10 })),
  ...overrides,
});

describe('createFilterOperationSession', () => {
  it('starts with a cloned persisted settings snapshot and an independent local draft', () => {
    const deps = createDeps();
    const session = createFilterOperationSession({
      deps,
      guard,
      initialFilter: layer.filter!,
      layerName: 'Portrait',
      layerType: 'raster',
    });

    expect(session).not.toBeNull();
    expect(session!.getSnapshot()).toMatchObject({
      autoProcess: true,
      draft: { settings: { radius: 2 }, type: 'canny_edge_detection' },
      initialFilter: { settings: { radius: 2 }, type: 'canny_edge_detection' },
      layerId: layer.id,
      layerName: 'Portrait',
      layerType: 'raster',
      status: 'ready',
    });

    session!.updateDraft({ settings: { radius: 9 }, type: 'content_shuffle' });
    expect(session!.getSnapshot().draft).toEqual({ settings: { radius: 9 }, type: 'content_shuffle' });
    expect(session!.getSnapshot().initialFilter).toEqual(layer.filter);
    expect(session!.getSnapshot().layerName).toBe('Portrait');
    session!.dispose();
  });

  it('publishes only the newest guarded process result', async () => {
    const older = deferred<{ height: number; imageName: string; origin: { x: number; y: number }; width: number }>();
    const newer = deferred<{ height: number; imageName: string; origin: { x: number; y: number }; width: number }>();
    const runs = [older, newer];
    const deps = createDeps({ runFilter: vi.fn(() => runs.shift()!.promise) });
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    const first = session.process();
    await vi.waitFor(() => expect(deps.runFilter).toHaveBeenCalledTimes(1));
    const second = session.process();
    await vi.waitFor(() => expect(deps.runFilter).toHaveBeenCalledTimes(2));
    newer.resolve({ height: 10, imageName: 'newer', origin: { x: 3, y: 4 }, width: 10 });
    await second;
    older.resolve({ height: 10, imageName: 'older', origin: { x: 3, y: 4 }, width: 10 });
    await first;

    expect(deps.publishPreview).toHaveBeenCalledOnce();
    expect(deps.publishPreview).toHaveBeenCalledWith(
      'newer',
      { height: 10, width: 10, x: 3, y: 4 },
      guard,
      'canny_edge_detection'
    );
    expect(session.getSnapshot().preview?.imageName).toBe('newer');
  });

  it('publishes and commits the filter output rect instead of the exported source rect', async () => {
    const deps = createDeps({
      runFilter: vi.fn(() =>
        Promise.resolve({ height: 22, imageName: 'blurred', origin: { x: -3, y: -2 }, width: 24 })
      ),
    });
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    await session.process();
    expect(session.getSnapshot().preview?.rect).toEqual({ height: 22, width: 24, x: -3, y: -2 });
    await session.commit('apply');

    expect(deps.commit).toHaveBeenCalledWith(
      expect.objectContaining({
        image: { height: 22, imageName: 'blurred', width: 24 },
        rect: { height: 22, width: 24, x: -3, y: -2 },
      })
    );
  });

  it('reset selects current filter defaults, clears preview, and keeps the operation active', async () => {
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;
    await session.process();

    session.reset({ high_threshold: 200, low_threshold: 100 });

    expect(session.getSnapshot()).toMatchObject({
      draft: { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' },
      error: null,
      preview: null,
      status: 'ready',
    });
    expect(deps.clearPreview).toHaveBeenCalled();
    expect(deps.controller.getSnapshot()).toMatchObject({ identity: { kind: 'filter' }, status: 'active' });
  });

  it('keeps the preview and session retryable when durability fails', async () => {
    const deps = createDeps({ makeDurable: vi.fn(() => Promise.reject(new Error('promotion failed'))) });
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;
    await session.process();

    await session.commit('apply');

    expect(deps.commit).not.toHaveBeenCalled();
    expect(session.getSnapshot()).toMatchObject({
      error: 'promotion failed',
      preview: { imageName: 'filtered' },
      status: 'error',
    });
    expect(deps.controller.getSnapshot()).toMatchObject({ identity: { kind: 'filter' }, status: 'active' });
  });

  it.each(['apply', 'raster', 'control'] as const)(
    'promotes, commits to %s, and leaves the operation for its owner to end',
    async (target) => {
      const calls: string[] = [];
      const deps = createDeps({
        commit: vi.fn((options) => {
          calls.push(`commit:${options.target}`);
          return Promise.resolve({ layerId: layer.id, status: 'committed' as const });
        }),
        makeDurable: vi.fn(() => {
          calls.push('durable');
          return Promise.resolve();
        }),
      });
      const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;
      await session.process();

      await session.commit(target);

      expect(calls).toEqual(['durable', `commit:${target}`]);
      expect(deps.commit).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: layer.filter,
          guard,
          target,
        })
      );
      expect(session.getSnapshot()).toMatchObject({ error: null, preview: null, status: 'ready' });
      expect(deps.controller.getSnapshot()).toMatchObject({ identity: { kind: 'filter' }, status: 'active' });
      session.dispose();
      expect(deps.controller.getSnapshot()).toEqual({ status: 'idle' });
    }
  );

  it('retains the guarded preview after a failed commit so Apply can retry', async () => {
    const deps = createDeps({
      commit: vi.fn(() => Promise.resolve({ message: 'cache failed', status: 'failed' as const })),
    });
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;
    await session.process();

    await session.commit('apply');

    expect(session.getSnapshot()).toMatchObject({
      error: 'cache failed',
      preview: { imageName: 'filtered' },
      status: 'error',
    });
    expect(deps.controller.getSnapshot()).toMatchObject({ identity: { kind: 'filter' }, status: 'active' });
  });

  it('cancel aborts work, clears preview, and closes the operation independently of subscribers', async () => {
    const result = deferred<{ height: number; imageName: string; origin: { x: number; y: number }; width: number }>();
    let signal: AbortSignal | undefined;
    const deps = createDeps({
      runFilter: vi.fn((options) => {
        signal = options.signal;
        return result.promise;
      }),
    });
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;
    const unsubscribe = session.subscribe(vi.fn());
    unsubscribe();
    const pending = session.process();
    await vi.waitFor(() => expect(signal).toBeDefined());

    session.cancel();
    result.resolve({ height: 10, imageName: 'late', origin: { x: 3, y: 4 }, width: 10 });
    await pending;

    expect(signal?.aborted).toBe(true);
    expect(deps.publishPreview).not.toHaveBeenCalled();
    expect(deps.controller.getSnapshot()).toEqual({ status: 'idle' });
  });

  it.each(['export', 'graph'] as const)(
    'interrupts processing at the %s boundary while preserving the draft and active operation',
    async (boundary) => {
      const wait = deferred<unknown>();
      const deps = createDeps({
        exportPixels:
          boundary === 'export'
            ? vi.fn(() => wait.promise as Promise<ExportLayerPixelsResult>)
            : vi.fn(() => Promise.resolve(exported)),
        runFilter:
          boundary === 'graph'
            ? vi.fn(() => wait.promise as ReturnType<FilterOperationSessionDeps['runFilter']>)
            : vi.fn(() => Promise.resolve({ height: 10, imageName: 'filtered', origin: { x: 3, y: 4 }, width: 10 })),
      });
      const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;
      session.updateDraft({ settings: { radius: 17 }, type: 'img_blur' });
      const pending = session.process();
      await vi.waitFor(() => expect(boundary === 'export' ? deps.exportPixels : deps.runFilter).toHaveBeenCalledOnce());

      session.interruptProcessing();
      if (boundary === 'export') {
        wait.resolve(exported);
      } else {
        wait.resolve({ height: 10, imageName: 'late', origin: { x: 3, y: 4 }, width: 10 });
      }
      await pending;

      expect(deps.publishPreview).not.toHaveBeenCalled();
      expect(session.getSnapshot()).toMatchObject({
        draft: { settings: { radius: 17 }, type: 'img_blur' },
        preview: null,
        status: 'ready',
      });
      expect(deps.controller.getSnapshot()).toMatchObject({ phase: 'ready', status: 'active' });
    }
  );
});

describe('auto-process', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  const flushScheduledRun = async (deps: FilterOperationSessionDeps, times: number) => {
    vi.useRealTimers();
    await vi.waitFor(() => expect(deps.runFilter).toHaveBeenCalledTimes(times));
  };

  it('debounces draft updates into a single process run with the latest settings', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { radius: 4 }, type: 'img_blur' });
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS - 100);
    expect(deps.runFilter).not.toHaveBeenCalled();
    session.updateDraft({ settings: { radius: 6 }, type: 'img_blur' });
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS);
    await flushScheduledRun(deps, 1);
    expect(deps.runFilter).toHaveBeenCalledWith(
      expect.objectContaining({ filterType: 'img_blur', settings: { radius: 6 } })
    );
    await vi.waitFor(() => expect(session.getSnapshot().preview?.imageName).toBe('filtered'));
    session.dispose();
  });

  it('never schedules for an invalid draft', async () => {
    vi.useFakeTimers();
    const deps = createDeps({ isDraftValid: vi.fn(() => false) });
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { model: null }, type: 'spandrel_filter' });
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    expect(deps.runFilter).not.toHaveBeenCalled();
    session.dispose();
  });

  it('setAutoProcess(false) cancels the pending run and blocks future scheduling', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { radius: 4 }, type: 'img_blur' });
    session.setAutoProcess(false);
    expect(session.getSnapshot().autoProcess).toBe(false);
    session.updateDraft({ settings: { radius: 6 }, type: 'img_blur' });
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    expect(deps.runFilter).not.toHaveBeenCalled();
    session.dispose();
  });

  it('setAutoProcess(true) schedules a run when no preview exists yet', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.setAutoProcess(false);
    session.setAutoProcess(true);
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS);
    await flushScheduledRun(deps, 1);
    session.dispose();
  });

  it('cancel clears the pending auto-run', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { radius: 4 }, type: 'img_blur' });
    session.cancel();
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    expect(deps.runFilter).not.toHaveBeenCalled();
    session.dispose();
  });

  it('a manual process supersedes the pending debounced run', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { radius: 4 }, type: 'img_blur' });
    await session.process();
    expect(deps.runFilter).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    expect(deps.runFilter).toHaveBeenCalledTimes(1);
    session.dispose();
  });

  it('interruptProcessing clears the pending auto-run', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { radius: 4 }, type: 'img_blur' });
    expect(session.getSnapshot().status).toBe('ready');
    session.interruptProcessing();
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    expect(deps.runFilter).not.toHaveBeenCalled();
    session.dispose();
  });

  it('setAutoProcess(true) does not reschedule when a preview exists', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    session.updateDraft({ settings: { radius: 4 }, type: 'img_blur' });
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS);
    await flushScheduledRun(deps, 1);
    await vi.waitFor(() => expect(session.getSnapshot().preview?.imageName).toBe('filtered'));

    session.setAutoProcess(false);
    session.setAutoProcess(true);
    await new Promise((resolve) => {
      setTimeout(resolve, FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    });
    expect(deps.runFilter).toHaveBeenCalledTimes(1);
    session.dispose();
  });

  it('setAutoProcess(true) does not schedule while a run is in flight', async () => {
    vi.useFakeTimers();
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' })!;

    const inFlight = session.process();
    session.setAutoProcess(false);
    session.setAutoProcess(true);
    await vi.advanceTimersByTimeAsync(FILTER_AUTO_PROCESS_DEBOUNCE_MS * 3);
    vi.useRealTimers();
    await inFlight;
    expect(deps.runFilter).toHaveBeenCalledTimes(1);
    session.dispose();
  });
});
