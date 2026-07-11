import type { ExportLayerPixelsResult, LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { createCanvasOperationController } from '@workbench/canvas-engine/canvasOperationController';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type { FilterOperationSessionDeps } from './filterOperationSession';

import { createFilterOperationSession } from './filterOperationSession';

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
  clearPreview: vi.fn(),
  commit: vi.fn(() => Promise.resolve({ layerId: layer.id, status: 'committed' as const })),
  controller: createCanvasOperationController({ isGuardCurrent: () => true }),
  exportPixels: vi.fn(() => Promise.resolve(exported)),
  isGuardCurrent: vi.fn(() => true),
  makeDurable: vi.fn(() => Promise.resolve()),
  publishPreview: vi.fn(() => Promise.resolve('shown' as const)),
  runFilter: vi.fn(() => Promise.resolve({ height: 10, imageName: 'filtered', origin: { x: 3, y: 4 }, width: 10 })),
  ...overrides,
});

describe('createFilterOperationSession', () => {
  it('starts with a cloned persisted settings snapshot and an independent local draft', () => {
    const deps = createDeps();
    const session = createFilterOperationSession({ deps, guard, initialFilter: layer.filter!, layerType: 'raster' });

    expect(session).not.toBeNull();
    expect(session!.getSnapshot()).toMatchObject({
      draft: { settings: { radius: 2 }, type: 'canny_edge_detection' },
      initialFilter: { settings: { radius: 2 }, type: 'canny_edge_detection' },
      layerId: layer.id,
      layerType: 'raster',
      status: 'ready',
    });

    session!.updateDraft({ settings: { radius: 9 }, type: 'content_shuffle' });
    expect(session!.getSnapshot().draft).toEqual({ settings: { radius: 9 }, type: 'content_shuffle' });
    expect(session!.getSnapshot().initialFilter).toEqual(layer.filter);
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
    expect(deps.publishPreview).toHaveBeenCalledWith('newer', { height: 10, width: 10, x: 3, y: 4 }, guard);
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
    'promotes, commits to %s, and exits after success',
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
});
