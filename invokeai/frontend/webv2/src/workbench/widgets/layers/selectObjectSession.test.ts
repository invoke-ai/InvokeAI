import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { createCanvasOperationController } from '@workbench/canvas-engine/canvasOperationController';
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
const replacementGuard: LayerExportGuard = { ...guard, cacheVersion: 4 };
const rect = { height: 12, width: 16, x: -5, y: 7 };
const blob = new Blob(['source'], { type: 'image/png' });

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const createHarness = (overrides: Partial<SelectObjectSessionDeps<string>> = {}) => {
  let currentGuard: LayerExportGuard | null = guard;
  const controller = createCanvasOperationController({ isGuardCurrent: (candidate) => candidate === currentGuard });
  const deps: SelectObjectSessionDeps<string> = {
    controller,
    decodePreview: vi.fn(({ image }) => Promise.resolve(`decoded:${image.imageName}`)),
    exportLayer: vi.fn(() => Promise.resolve({ blob, guard: currentGuard ?? guard, rect, status: 'ok' as const })),
    isGuardCurrent: (candidate) => candidate === currentGuard,
    publishPreview: vi.fn((): undefined => undefined),
    cleanupPreview: vi.fn(),
    runGraph: vi.fn(() => Promise.resolve({ imageName: 'result.png', origin: 'webv2:util:test' })),
    uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'input.png' })),
    ...overrides,
  };
  const session = createSelectObjectSession({ deps, layerId: layer.id, projectId: 'p1' });
  session.update({ input: { prompt: 'cat', type: 'prompt' } });
  return {
    controller,
    deps,
    session,
    setCurrentGuard(next: LayerExportGuard | null) {
      currentGuard = next;
    },
  };
};

afterEach(() => {
  vi.useRealTimers();
});

describe('createSelectObjectSession', () => {
  it('exposes the complete default processing state', () => {
    const { session } = createHarness();

    expect(session.getSnapshot()).toMatchObject({
      applyPolygonRefinement: false,
      autoProcess: false,
      error: null,
      input: { prompt: 'cat', type: 'prompt' },
      invert: false,
      isolatedPreview: true,
      model: 'segment-anything-2-large',
      preview: null,
      sourceGuard: null,
      status: 'ready',
    });
  });

  it('reuses an uploaded export only while its exact guard remains current', async () => {
    const harness = createHarness();

    await expect(harness.session.process()).resolves.toBe('published');
    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(harness.deps.exportLayer).toHaveBeenCalledTimes(1);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(1);
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot().sourceGuard).toBe(guard);

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportLayer).mockResolvedValue({
      blob,
      guard: replacementGuard,
      rect,
      status: 'ok',
    });
    harness.session.update({ input: { prompt: 'bird', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(harness.deps.exportLayer).toHaveBeenCalledTimes(2);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot().sourceGuard).toBe(replacementGuard);
  });

  it.each(['source', 'layer', 'project', 'document'] as const)(
    'invalidates the cached export and preview on %s invalidation',
    async (kind) => {
      const harness = createHarness();
      await harness.session.process();
      vi.mocked(harness.deps.cleanupPreview).mockClear();

      if (kind === 'source') {
        harness.controller.invalidateSource('p1', layer.id);
      } else if (kind === 'layer') {
        harness.controller.invalidateLayer('p1', layer.id);
      } else if (kind === 'project') {
        harness.controller.invalidateProject('p1');
      } else {
        harness.controller.invalidateDocument('p1');
      }

      expect(harness.session.getSnapshot()).toMatchObject({ preview: null, sourceGuard: null, status: 'ready' });
      expect(harness.deps.cleanupPreview).toHaveBeenCalledOnce();
    }
  );

  it('deduplicates an identical stable input hash after publication', async () => {
    const harness = createHarness();

    await expect(harness.session.process()).resolves.toBe('published');
    await expect(harness.session.process()).resolves.toBe('deduped');

    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });

  it('does not deduplicate the same hash after its exact source guard is replaced', async () => {
    const harness = createHarness();
    await expect(harness.session.process()).resolves.toBe('published');

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportLayer).mockResolvedValue({ blob, guard: replacementGuard, rect, status: 'ok' });

    await expect(harness.session.process()).resolves.toBe('published');
    expect(harness.deps.exportLayer).toHaveBeenCalledTimes(2);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(2);
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot().sourceGuard).toBe(replacementGuard);
  });

  it('debounces auto-processing for one second and manual processing runs immediately', async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.session.update({ autoProcess: true });
    harness.session.update({ input: { prompt: 'first', type: 'prompt' } });
    await vi.advanceTimersByTimeAsync(500);
    harness.session.update({ input: { prompt: 'second', type: 'prompt' } });
    await vi.advanceTimersByTimeAsync(999);
    expect(harness.deps.exportLayer).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.update({ input: { prompt: 'manual', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
  });

  it('immediately stales a delayed auto-run when input changes before the replacement debounce', async () => {
    vi.useFakeTimers();
    const oldGraph = deferred<{ imageName: string; origin: string }>();
    const harness = createHarness({ runGraph: vi.fn(() => oldGraph.promise) });
    harness.session.update({ autoProcess: true });
    await vi.advanceTimersByTimeAsync(1_000);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    expect(harness.session.getSnapshot()).toMatchObject({ preview: null, status: 'scheduled' });
    oldGraph.resolve({ imageName: 'old.png', origin: 'old' });
    await vi.advanceTimersByTimeAsync(999);

    expect(harness.deps.publishPreview).not.toHaveBeenCalled();
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });

  it.each([
    ['input', { input: { prompt: 'dog', type: 'prompt' as const } }],
    ['model', { model: 'segment-anything-huge' as const }],
    ['invert', { invert: true }],
    ['refinement', { applyPolygonRefinement: true }],
    ['preview isolation', { isolatedPreview: false }],
  ] as const)('immediately aborts in-flight work and clears preview for a %s update', async (_label, update) => {
    const signals: AbortSignal[] = [];
    const graph = deferred<{ imageName: string; origin: string }>();
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

    harness.session.update(update);

    expect(signals[0]?.aborted).toBe(true);
    expect(harness.session.getSnapshot().preview).toBeNull();
    graph.resolve({ imageName: 'stale.png', origin: 'stale' });
    await expect(pending).resolves.toBe('stale');
    expect(harness.deps.publishPreview).not.toHaveBeenCalled();
  });

  it('clears an already-published preview as soon as processing input changes', async () => {
    const harness = createHarness();
    await harness.session.process();
    vi.mocked(harness.deps.cleanupPreview).mockClear();

    harness.session.update({ model: 'segment-anything-huge' });

    expect(harness.session.getSnapshot()).toMatchObject({ preview: null, status: 'ready' });
    expect(harness.deps.cleanupPreview).toHaveBeenCalledOnce();
  });

  it('rejects non-finite document input before hashing or scheduling work', async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    harness.session.update({ autoProcess: true });
    harness.session.update({
      input: {
        bbox: null,
        excludePoints: [],
        includePoints: [{ x: Number.NaN, y: 1 }],
        type: 'visual',
      },
    });

    await expect(harness.session.process()).resolves.toBe('invalid');
    await vi.runAllTimersAsync();
    expect(harness.deps.exportLayer).not.toHaveBeenCalled();
    expect(harness.deps.runGraph).not.toHaveBeenCalled();
  });

  it('publishes only the latest request when graph completions arrive out of order', async () => {
    const older = deferred<{ imageName: string; origin: string }>();
    const newer = deferred<{ imageName: string; origin: string }>();
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

    newer.resolve({ imageName: 'new.png', origin: 'new' });
    await expect(newPending).resolves.toBe('published');
    older.resolve({ imageName: 'old.png', origin: 'old' });
    await expect(oldPending).resolves.toBe('stale');

    expect(harness.deps.publishPreview).toHaveBeenCalledOnce();
    expect(harness.session.getSnapshot().preview?.image.imageName).toBe('new.png');
  });

  it.each(['export', 'upload', 'queue', 'decode'] as const)(
    'cancels safely at the %s boundary without starting later work',
    async (boundary) => {
      const wait = deferred<unknown>();
      const harness = createHarness({
        decodePreview:
          boundary === 'decode'
            ? vi.fn(() => wait.promise as Promise<string>)
            : vi.fn(({ image }) => Promise.resolve(`decoded:${image.imageName}`)),
        exportLayer:
          boundary === 'export'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['exportLayer']>)
            : vi.fn(() => Promise.resolve({ blob, guard, rect, status: 'ok' as const })),
        runGraph:
          boundary === 'queue'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['runGraph']>)
            : vi.fn(() => Promise.resolve({ imageName: 'result.png', origin: 'test' })),
        uploadIntermediate:
          boundary === 'upload'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['uploadIntermediate']>)
            : vi.fn(() => Promise.resolve({ imageName: 'input.png' })),
      });
      const pending = harness.session.process();
      await vi.waitFor(() => {
        const fn =
          boundary === 'export'
            ? harness.deps.exportLayer
            : boundary === 'upload'
              ? harness.deps.uploadIntermediate
              : boundary === 'queue'
                ? harness.deps.runGraph
                : harness.deps.decodePreview;
        expect(fn).toHaveBeenCalledOnce();
      });

      harness.session.cancel();
      if (boundary === 'export') {
        wait.resolve({ blob, guard, rect, status: 'ok' });
      } else if (boundary === 'upload') {
        wait.resolve({ imageName: 'input.png' });
      } else if (boundary === 'queue') {
        wait.resolve({ imageName: 'result.png', origin: 'test' });
      } else {
        wait.resolve('decoded');
      }

      await expect(pending).resolves.toBe('stale');
      expect(harness.deps.publishPreview).not.toHaveBeenCalled();
      expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(boundary === 'export' ? 0 : 1);
      expect(harness.deps.runGraph).toHaveBeenCalledTimes(boundary === 'export' || boundary === 'upload' ? 0 : 1);
      expect(harness.deps.decodePreview).toHaveBeenCalledTimes(
        boundary === 'export' || boundary === 'upload' || boundary === 'queue' ? 0 : 1
      );
    }
  );

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

  it('reset cancels work and scheduling, clears preview/cache, and restores defaults', async () => {
    vi.useFakeTimers();
    const graph = deferred<{ imageName: string; origin: string }>();
    const harness = createHarness({ runGraph: vi.fn(() => graph.promise) });
    harness.session.update({ autoProcess: true, invert: true, model: 'segment-anything-huge' });
    const pending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.reset();
    graph.resolve({ imageName: 'late.png', origin: 'test' });

    await expect(pending).resolves.toBe('stale');
    expect(harness.session.getSnapshot()).toEqual({
      applyPolygonRefinement: false,
      autoProcess: false,
      error: null,
      input: { prompt: '', type: 'prompt' },
      invert: false,
      isolatedPreview: true,
      model: 'segment-anything-2-large',
      preview: null,
      sourceGuard: null,
      status: 'ready',
    });
    await vi.runAllTimersAsync();
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });
});
