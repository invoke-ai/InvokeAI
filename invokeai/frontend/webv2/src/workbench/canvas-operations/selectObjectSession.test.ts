import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { LayerExportGuard } from '@workbench/canvas-engine/engine';

import { createCanvasOperationController } from '@workbench/canvas-operations/operationController';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { SelectObjectSessionDeps } from './selectObjectSession';

import { createSelectObjectSession } from './selectObjectSession';

const layer: CanvasRasterLayerContractV2 = {
  blendMode: 'normal',
  id: 'source',
  isEnabled: true,
  isLocked: false,
  name: 'Source',
  opacity: 1,
  source: { fill: '#fff', height: 12, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 16 },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -5, y: 7 },
  type: 'raster',
};
const guard: LayerExportGuard = {
  cacheVersion: 3,
  documentGeneration: 4,
  layer,
  layerId: layer.id,
  projectId: 'p1',
};
const replacementGuard: LayerExportGuard = {
  ...guard,
  cacheVersion: 4,
};
const rect = { height: 12, width: 16, x: -5, y: 7 };
const blob = new Blob(['source'], { type: 'image/png' });

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const createHarness = (
  overrides: Partial<SelectObjectSessionDeps<string>> = {},
  { initialAutoProcess }: { initialAutoProcess?: boolean } = {}
) => {
  let currentGuard: LayerExportGuard | null = guard;
  let controllerSubscriber: (() => void) | null = null;
  const unsubscribe = vi.fn();
  const baseController = createCanvasOperationController({
    isGuardCurrent: (candidate) => candidate === guard,
  });
  const controller = {
    ...baseController,
    subscribe: vi.fn((listener: () => void) => {
      controllerSubscriber = listener;
      const detach = baseController.subscribe(listener);
      return () => {
        unsubscribe();
        detach();
      };
    }),
  };
  const deps: SelectObjectSessionDeps<string> = {
    captureGuard: vi.fn(() => currentGuard),
    controller,
    decodePreview: vi.fn(({ image }) => Promise.resolve(`decoded:${image.imageName}`)),
    exportSource: vi.fn(() => Promise.resolve({ blob, guard: currentGuard ?? guard, rect, status: 'ok' as const })),
    isGuardCurrent: (candidate) => candidate === currentGuard,
    publishPreview: vi.fn((): undefined => undefined),
    cleanupPreview: vi.fn(),
    runGraph: vi.fn(() =>
      Promise.resolve({ height: 12, imageName: 'result.png', origin: 'webv2:util:test', width: 16 })
    ),
    uploadIntermediate: vi.fn(() => Promise.resolve({ height: 12, imageName: 'input.png', width: 16 })),
    ...overrides,
  };
  const operation = controller.start({
    cleanupPreview: deps.cleanupPreview,
    guard,
    identity: { kind: 'select-object', layerId: layer.id, projectId: 'p1' },
  });
  if (!operation) {
    throw new Error('expected Select Object operation');
  }
  const session = createSelectObjectSession({ deps, initialAutoProcess: initialAutoProcess ?? false, operation });
  session.update({ input: { prompt: 'cat', type: 'prompt' } });
  return {
    controller,
    deps,
    fireControllerSubscriber() {
      controllerSubscriber?.();
    },
    session,
    setCurrentGuard(next: LayerExportGuard | null) {
      currentGuard = next;
    },
    unsubscribe,
  };
};

afterEach(() => {
  vi.useRealTimers();
});

describe('createSelectObjectSession', () => {
  it('reports phases from the source export, upload, SAM, render, and ready boundaries', async () => {
    const harness = createHarness();
    const phases: string[] = [];
    harness.session.subscribe(() => phases.push(harness.session.getSnapshot().status));

    await expect(harness.session.process()).resolves.toBe('published');

    expect(phases).toEqual(['preparing-source', 'uploading', 'processing-sam', 'rendering-preview', 'ready']);
  });

  it('does not publish a stale request phase after a replacement retry starts', async () => {
    const oldUpload = deferred<{ height: number; imageName: string; width: number }>();
    const harness = createHarness({
      uploadIntermediate: vi
        .fn()
        .mockImplementationOnce(() => oldUpload.promise)
        .mockResolvedValueOnce({ height: 12, imageName: 'new-source.png', width: 16 }),
    });
    const phases: string[] = [];
    harness.session.subscribe(() => phases.push(harness.session.getSnapshot().status));
    const oldPending = harness.session.process();
    await vi.waitFor(() => expect(harness.session.getSnapshot().status).toBe('uploading'));

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportSource).mockResolvedValue({ blob, guard: replacementGuard, rect, status: 'ok' });
    const newPending = harness.session.process();
    await expect(newPending).resolves.toBe('published');
    const readyIndex = phases.lastIndexOf('ready');
    oldUpload.resolve({ height: 12, imageName: 'old-source.png', width: 16 });
    await expect(oldPending).resolves.toBe('stale');

    expect(phases.slice(readyIndex + 1)).toEqual([]);
    expect(harness.session.getSnapshot()).toMatchObject({ status: 'ready' });
  });

  it('reports routing failures without discarding the preview and Reset clears the error', async () => {
    const harness = createHarness();
    await harness.session.process();
    const preview = harness.session.getSnapshot().preview;

    harness.session.reportError('commit failed');

    expect(harness.session.getSnapshot()).toMatchObject({
      error: { code: 'unknown', detail: 'commit failed' },
      preview,
      status: 'error',
    });
    harness.session.reset();
    expect(harness.session.getSnapshot()).toMatchObject({ error: null, preview: null, status: 'ready' });
  });

  it('survives source invalidation of other layers but closes when its own layer is invalidated', async () => {
    const harness = createHarness({ runGraph: vi.fn(() => Promise.reject(new Error('graph failed'))) });
    await expect(harness.session.process()).resolves.toBe('error');

    harness.controller.invalidateSource('p1', 'other-layer');

    expect(harness.session.getSnapshot()).toMatchObject({ error: { code: 'queue' }, status: 'error' });
    expect(harness.controller.getSnapshot()).toMatchObject({ status: 'active' });

    harness.controller.invalidateSource('p1', layer.id);

    expect(harness.controller.getSnapshot()).toMatchObject({ status: 'idle' });
    expect(harness.session.getSnapshot()).toMatchObject({ preview: null, sourceGuard: null, status: 'ready' });
  });

  it('reuses an uploaded export only while its exact guard remains current', async () => {
    const harness = createHarness();

    await expect(harness.session.process()).resolves.toBe('published');
    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(harness.deps.exportSource).toHaveBeenCalledTimes(1);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(1);
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot().sourceGuard).toBe(guard);

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportSource).mockResolvedValue({
      blob,
      guard: replacementGuard,
      rect,
      status: 'ok',
    });
    harness.session.update({ input: { prompt: 'bird', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(harness.deps.exportSource).toHaveBeenCalledTimes(2);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot().sourceGuard).toBe(replacementGuard);
  });

  it('starts an immediate same-input retry after invalidation when old work ignores abort', async () => {
    const older = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const newer = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const harness = createHarness({
      runGraph: vi
        .fn()
        .mockImplementationOnce(() => older.promise)
        .mockImplementationOnce(() => newer.promise),
    });
    const oldPending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportSource).mockResolvedValue({ blob, guard: replacementGuard, rect, status: 'ok' });
    const newPending = harness.session.process();

    expect(newPending).not.toBe(oldPending);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledTimes(2));
    expect(harness.deps.exportSource).toHaveBeenCalledTimes(2);
    older.resolve({ height: 12, imageName: 'old.png', origin: 'old', width: 16 });
    await expect(oldPending).resolves.toBe('stale');
    expect(harness.session.process()).toBe(newPending);

    newer.resolve({ height: 12, imageName: 'new.png', origin: 'new', width: 16 });
    await expect(newPending).resolves.toBe('published');
    expect(harness.session.getSnapshot().preview?.image.imageName).toBe('new.png');
    await expect(harness.session.process()).resolves.toBe('deduped');
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
  });

  it('deduplicates an identical stable input hash after publication', async () => {
    const harness = createHarness();

    await expect(harness.session.process()).resolves.toBe('published');
    await expect(harness.session.process()).resolves.toBe('deduped');

    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });

  it('runs the same input again on a manual retry after a failed auto-run', async () => {
    vi.useFakeTimers();
    const harness = createHarness({}, { initialAutoProcess: true });
    harness.setCurrentGuard(null);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(harness.session.getSnapshot()).toMatchObject({ error: { code: 'not-ready' }, status: 'error' });

    harness.setCurrentGuard(guard);
    const retry = harness.session.process();
    expect(harness.session.getSnapshot()).toMatchObject({ error: null, retrying: true, status: 'preparing-source' });
    await expect(retry).resolves.toBe('published');
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
    expect(harness.session.getSnapshot()).toMatchObject({ retrying: false, status: 'ready' });

    vi.mocked(harness.deps.runGraph).mockRejectedValueOnce(new Error('graph failed'));
    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    await vi.advanceTimersByTimeAsync(1_000);
    expect(harness.session.getSnapshot()).toMatchObject({ error: { code: 'queue' }, status: 'error' });
    await expect(harness.session.process()).resolves.toBe('published');
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(3);
    expect(harness.session.getSnapshot()).toMatchObject({ error: null, retrying: false, status: 'ready' });
  });

  it('debounces auto-processing for one second and manual processing runs immediately', async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.session.update({ autoProcess: true });
    harness.session.update({ input: { prompt: 'first', type: 'prompt' } });
    await vi.advanceTimersByTimeAsync(500);
    harness.session.update({ input: { prompt: 'second', type: 'prompt' } });
    await vi.advanceTimersByTimeAsync(999);
    expect(harness.deps.exportSource).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.update({ input: { prompt: 'manual', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
  });

  it('publishes only the latest request when graph completions arrive out of order', async () => {
    const older = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const newer = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const harness = createHarness({
      runGraph: vi
        .fn()
        .mockImplementationOnce(() => older.promise)
        .mockImplementationOnce(() => newer.promise),
    });
    const oldPending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());
    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    const newPending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledTimes(2));

    newer.resolve({ height: 12, imageName: 'new.png', origin: 'new', width: 16 });
    await expect(newPending).resolves.toBe('published');
    older.resolve({ height: 12, imageName: 'old.png', origin: 'old', width: 16 });
    await expect(oldPending).resolves.toBe('stale');

    expect(harness.deps.publishPreview).toHaveBeenCalledOnce();
    expect(harness.session.getSnapshot().preview?.image.imageName).toBe('new.png');
  });

  it('suppresses a decoded preview when the source guard becomes stale', async () => {
    const decode = deferred<string>();
    const harness = createHarness({ decodePreview: vi.fn(() => decode.promise) });
    const pending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.decodePreview).toHaveBeenCalledOnce());

    harness.setCurrentGuard(null);
    decode.resolve('decoded');

    await expect(pending).resolves.toBe('stale');
    expect(harness.deps.publishPreview).not.toHaveBeenCalled();
    expect(harness.session.getSnapshot()).toMatchObject({ preview: null, sourceGuard: null, status: 'ready' });
  });

  it('reset cancels work and scheduling, clears preview/cache, and restores defaults except auto-process', async () => {
    vi.useFakeTimers();
    const graph = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const harness = createHarness({ runGraph: vi.fn(() => graph.promise) });
    harness.session.update({ autoProcess: true, invert: true, model: 'segment-anything-huge' });
    const pending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.reset();
    graph.resolve({ height: 12, imageName: 'late.png', origin: 'test', width: 16 });

    await expect(pending).resolves.toBe('stale');
    expect(harness.session.getSnapshot()).toEqual({
      applyPolygonRefinement: false,
      autoProcess: true,
      error: null,
      input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
      invert: false,
      isolatedPreview: true,
      model: 'segment-anything-2-large',
      preview: null,
      retrying: false,
      sourceGuard: null,
      status: 'ready',
    });
    await vi.runAllTimersAsync();
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });

  it('dispose clears scheduling and preview, unsubscribes exactly once, and makes methods inert', async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    await harness.session.process();
    harness.session.update({ autoProcess: true });
    const listener = vi.fn();
    harness.session.subscribe(listener);
    vi.mocked(harness.deps.cleanupPreview).mockClear();
    listener.mockClear();

    harness.session.dispose();
    const disposedState = harness.session.getSnapshot();

    expect(disposedState).toMatchObject({ preview: null, sourceGuard: null, status: 'ready' });
    expect(harness.deps.cleanupPreview).toHaveBeenCalledOnce();
    expect(harness.unsubscribe).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    listener.mockClear();

    harness.fireControllerSubscriber();
    harness.session.update({ model: 'segment-anything-huge' });
    harness.session.reset();
    harness.session.cancel();
    harness.session.dispose();
    const lateListener = vi.fn();
    harness.session.subscribe(lateListener);

    await expect(harness.session.process()).resolves.toBe('stale');
    expect(harness.session.getSnapshot()).toBe(disposedState);
    expect(listener).not.toHaveBeenCalled();
    expect(lateListener).not.toHaveBeenCalled();
    expect(harness.unsubscribe).toHaveBeenCalledOnce();
  });

  it('dispose aborts in-flight processing and suppresses its completion', async () => {
    const graph = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const signals: AbortSignal[] = [];
    const harness = createHarness({
      runGraph: vi.fn((options) => {
        if (options.signal) {
          signals.push(options.signal);
        }
        return graph.promise;
      }),
    });
    const pending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.dispose();

    expect(signals[0]?.aborted).toBe(true);
    graph.resolve({ height: 12, imageName: 'late.png', origin: 'late', width: 16 });
    await expect(pending).resolves.toBe('stale');
    expect(harness.deps.publishPreview).not.toHaveBeenCalled();
    expect(harness.unsubscribe).toHaveBeenCalledOnce();
  });
});
