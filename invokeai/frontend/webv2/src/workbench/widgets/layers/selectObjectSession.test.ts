import type { CanvasCompositeExportGuard } from '@workbench/canvas-engine/engine';
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
const guard: CanvasCompositeExportGuard = {
  bbox: { height: 12, width: 16, x: -5, y: 7 },
  documentFingerprint: 'document:3',
  documentGeneration: 4,
  participants: [{ cacheVersion: 3, layer, layerId: layer.id }],
  projectId: 'p1',
};
const replacementGuard: CanvasCompositeExportGuard = {
  ...guard,
  documentFingerprint: 'document:4',
  participants: [{ cacheVersion: 4, layer, layerId: layer.id }],
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

const createHarness = (overrides: Partial<SelectObjectSessionDeps<string>> = {}) => {
  let currentGuard: CanvasCompositeExportGuard | null = guard;
  let controllerSubscriber: (() => void) | null = null;
  const unsubscribe = vi.fn();
  const baseController = createCanvasOperationController({ isGuardCurrent: (candidate) => candidate === currentGuard });
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
    exportComposite: vi.fn(() => Promise.resolve({ blob, guard: currentGuard ?? guard, rect, status: 'ok' as const })),
    isGuardCurrent: (candidate) => candidate === currentGuard,
    publishPreview: vi.fn((): undefined => undefined),
    cleanupPreview: vi.fn(),
    runGraph: vi.fn(() => Promise.resolve({ imageName: 'result.png', origin: 'webv2:util:test' })),
    uploadIntermediate: vi.fn(() => Promise.resolve({ imageName: 'input.png' })),
    ...overrides,
  };
  const session = createSelectObjectSession({ deps, projectId: 'p1' });
  session.update({ input: { prompt: 'cat', type: 'prompt' } });
  return {
    controller,
    deps,
    fireControllerSubscriber() {
      controllerSubscriber?.();
    },
    session,
    setCurrentGuard(next: CanvasCompositeExportGuard | null) {
      currentGuard = next;
    },
    unsubscribe,
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

  it('contains subscriber exceptions so healthy subscribers and processing continue', async () => {
    const harness = createHarness();
    const throwing = vi.fn(() => {
      throw new Error('subscriber failed');
    });
    const healthy = vi.fn();
    harness.session.subscribe(throwing);
    harness.session.subscribe(healthy);

    expect(() => harness.session.update({ model: 'segment-anything-huge' })).not.toThrow();
    await expect(harness.session.process()).resolves.toBe('published');

    expect(throwing).toHaveBeenCalled();
    expect(healthy).toHaveBeenCalled();
    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'ready' });
  });

  it('never publishes error status without an error and returns to ready when update clears it', async () => {
    const harness = createHarness();
    const snapshots: Array<{ error: string | null; status: string }> = [];
    harness.session.subscribe(() => {
      const { error, status } = harness.session.getSnapshot();
      snapshots.push({ error, status });
    });
    harness.session.update({ input: { prompt: '   ', type: 'prompt' } });

    await expect(harness.session.process()).resolves.toBe('invalid');
    expect(harness.session.getSnapshot()).toMatchObject({
      error: 'A Segment Anything input is required.',
      status: 'error',
    });

    harness.session.update({ autoProcess: false });

    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'ready' });
    expect(snapshots.every(({ error, status }) => (status === 'error') === (error !== null))).toBe(true);
  });

  it('keeps a non-null error whenever processing fails', async () => {
    const harness = createHarness({ runGraph: vi.fn(() => Promise.reject(new Error('graph failed'))) });

    await expect(harness.session.process()).resolves.toBe('error');

    expect(harness.session.getSnapshot()).toEqual(expect.objectContaining({ error: 'graph failed', status: 'error' }));
  });

  it('reports routing failures without discarding the preview and Reset clears the error', async () => {
    const harness = createHarness();
    await harness.session.process();
    const preview = harness.session.getSnapshot().preview;

    harness.session.reportError('commit failed');

    expect(harness.session.getSnapshot()).toMatchObject({ error: 'commit failed', preview, status: 'error' });
    harness.session.reset();
    expect(harness.session.getSnapshot()).toMatchObject({ error: null, preview: null, status: 'ready' });
  });

  it('clears a failed process error when controller source invalidation returns the session to ready', async () => {
    const harness = createHarness({ runGraph: vi.fn(() => Promise.reject(new Error('graph failed'))) });
    await expect(harness.session.process()).resolves.toBe('error');

    harness.controller.invalidateSource('p1', layer.id);

    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'ready' });
  });

  it('reuses an uploaded export only while its exact guard remains current', async () => {
    const harness = createHarness();

    await expect(harness.session.process()).resolves.toBe('published');
    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(1);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(1);
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot().sourceGuard).toBe(guard);

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportComposite).mockResolvedValue({
      blob,
      guard: replacementGuard,
      rect,
      status: 'ok',
    });
    harness.session.update({ input: { prompt: 'bird', type: 'prompt' } });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(2);
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

  it('starts an immediate same-input retry after invalidation when old work ignores abort', async () => {
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

    harness.controller.invalidateSource('p1', layer.id);
    const newPending = harness.session.process();

    expect(newPending).not.toBe(oldPending);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledTimes(2));
    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(2);
    older.resolve({ imageName: 'old.png', origin: 'old' });
    await expect(oldPending).resolves.toBe('stale');
    expect(harness.session.process()).toBe(newPending);

    newer.resolve({ imageName: 'new.png', origin: 'new' });
    await expect(newPending).resolves.toBe('published');
    expect(harness.session.getSnapshot().preview?.image.imageName).toBe('new.png');
    await expect(harness.session.process()).resolves.toBe('deduped');
    expect(harness.deps.runGraph).toHaveBeenCalledTimes(2);
  });

  it('starts an immediate same-input retry when invalidation aborts an export that ignores its signal', async () => {
    const oldExport = deferred<Awaited<ReturnType<SelectObjectSessionDeps<string>['exportComposite']>>>();
    const harness = createHarness({
      exportComposite: vi
        .fn()
        .mockImplementationOnce(() => oldExport.promise)
        .mockImplementationOnce(() => Promise.resolve({ blob, guard, rect, status: 'ok' as const })),
    });
    const oldPending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.exportComposite).toHaveBeenCalledOnce());

    harness.controller.invalidateSource('p1', layer.id);
    const newPending = harness.session.process();

    expect(newPending).not.toBe(oldPending);
    await expect(newPending).resolves.toBe('published');
    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(2);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledOnce();
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
    oldExport.resolve({ blob, guard, rect, status: 'ok' });
    await expect(oldPending).resolves.toBe('stale');
    expect(harness.session.getSnapshot().preview?.image.imageName).toBe('result.png');
    await expect(harness.session.process()).resolves.toBe('deduped');
  });

  it('starts an immediate same-input retry when invalidation aborts an upload that ignores its signal', async () => {
    const oldUpload = deferred<{ imageName: string }>();
    const harness = createHarness({
      uploadIntermediate: vi
        .fn()
        .mockImplementationOnce(() => oldUpload.promise)
        .mockImplementationOnce(() => Promise.resolve({ imageName: 'new-input.png' })),
    });
    const oldPending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.uploadIntermediate).toHaveBeenCalledOnce());

    harness.controller.invalidateSource('p1', layer.id);
    const newPending = harness.session.process();

    expect(newPending).not.toBe(oldPending);
    await expect(newPending).resolves.toBe('published');
    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(2);
    expect(harness.deps.uploadIntermediate).toHaveBeenCalledTimes(2);
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
    oldUpload.resolve({ imageName: 'old-input.png' });
    await expect(oldPending).resolves.toBe('stale');
    expect(harness.session.getSnapshot().preview?.sourceImageName).toBe('new-input.png');
    await expect(harness.session.process()).resolves.toBe('deduped');
  });

  it('deduplicates an identical stable input hash after publication', async () => {
    const harness = createHarness();

    await expect(harness.session.process()).resolves.toBe('published');
    await expect(harness.session.process()).resolves.toBe('deduped');

    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });

  it('transitions an identical scheduled auto-run back to ready and preserves its preview', async () => {
    vi.useFakeTimers();
    const harness = createHarness();
    await harness.session.process();
    const preview = harness.session.getSnapshot().preview;
    const listener = vi.fn();
    harness.session.subscribe(listener);

    harness.session.update({ autoProcess: true });
    expect(harness.session.getSnapshot().status).toBe('scheduled');
    listener.mockClear();
    await vi.advanceTimersByTimeAsync(1_000);

    expect(harness.session.getSnapshot()).toMatchObject({ error: null, preview, status: 'ready' });
    expect(listener).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
  });

  it('transitions an identical scheduled auto-run back to processing when work is already in flight', async () => {
    vi.useFakeTimers();
    const graph = deferred<{ imageName: string; origin: string }>();
    const harness = createHarness({ runGraph: vi.fn(() => graph.promise) });
    const pending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.update({ autoProcess: true });
    expect(harness.session.getSnapshot().status).toBe('scheduled');
    await vi.advanceTimersByTimeAsync(1_000);

    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'processing' });
    expect(vi.getTimerCount()).toBe(0);
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
    graph.resolve({ imageName: 'result.png', origin: 'test' });
    await expect(pending).resolves.toBe('published');
  });

  it('does not deduplicate the same hash after its exact source guard is replaced', async () => {
    const harness = createHarness();
    await expect(harness.session.process()).resolves.toBe('published');

    harness.setCurrentGuard(replacementGuard);
    vi.mocked(harness.deps.exportComposite).mockResolvedValue({ blob, guard: replacementGuard, rect, status: 'ok' });

    await expect(harness.session.process()).resolves.toBe('published');
    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(2);
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
    expect(harness.deps.exportComposite).not.toHaveBeenCalled();
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

  it('changes ready preview isolation without clearing or reprocessing it', async () => {
    const harness = createHarness();
    await harness.session.process();
    const preview = harness.session.getSnapshot().preview;
    vi.mocked(harness.deps.publishPreview).mockClear();
    vi.mocked(harness.deps.runGraph).mockClear();
    vi.mocked(harness.deps.cleanupPreview).mockClear();

    harness.session.update({ isolatedPreview: false });

    expect(harness.session.getSnapshot()).toMatchObject({ isolatedPreview: false });
    expect(harness.session.getSnapshot().preview).toMatchObject({ data: preview?.data, isolated: false });
    expect(harness.deps.publishPreview).toHaveBeenCalledOnce();
    expect(harness.deps.cleanupPreview).not.toHaveBeenCalled();
    expect(harness.deps.runGraph).not.toHaveBeenCalled();
  });

  it('changes isolation during processing without aborting or rerunning the request', async () => {
    const graph = deferred<{ imageName: string; origin: string }>();
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

    harness.session.update({ isolatedPreview: false });
    graph.resolve({ imageName: 'result.png', origin: 'test' });

    await expect(pending).resolves.toBe('published');
    expect(signals[0]?.aborted).toBe(false);
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
    expect(harness.session.getSnapshot().preview).toMatchObject({ isolated: false });
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
    expect(harness.deps.exportComposite).not.toHaveBeenCalled();
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
        exportComposite:
          boundary === 'export'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['exportComposite']>)
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
            ? harness.deps.exportComposite
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

  it.each(['export', 'upload', 'queue', 'decode'] as const)(
    'interrupts processing at the %s boundary while preserving inputs and the active operation',
    async (boundary) => {
      const wait = deferred<unknown>();
      const harness = createHarness({
        decodePreview:
          boundary === 'decode'
            ? vi.fn(() => wait.promise as Promise<string>)
            : vi.fn(({ image }) => Promise.resolve(`decoded:${image.imageName}`)),
        exportComposite:
          boundary === 'export'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['exportComposite']>)
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
      harness.session.update({ input: { prompt: 'keep me', type: 'prompt' }, invert: true });
      const pending = harness.session.process();
      await vi.waitFor(() => {
        const fn =
          boundary === 'export'
            ? harness.deps.exportComposite
            : boundary === 'upload'
              ? harness.deps.uploadIntermediate
              : boundary === 'queue'
                ? harness.deps.runGraph
                : harness.deps.decodePreview;
        expect(fn).toHaveBeenCalledOnce();
      });

      harness.session.interruptProcessing();
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
      expect(harness.session.getSnapshot()).toMatchObject({
        input: { prompt: 'keep me', type: 'prompt' },
        invert: true,
        preview: null,
        status: 'ready',
      });
      expect(harness.deps.controller.getSnapshot()).toMatchObject({ phase: 'ready', status: 'active' });
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
    const graph = deferred<{ imageName: string; origin: string }>();
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
    graph.resolve({ imageName: 'late.png', origin: 'late' });
    await expect(pending).resolves.toBe('stale');
    expect(harness.deps.publishPreview).not.toHaveBeenCalled();
    expect(harness.unsubscribe).toHaveBeenCalledOnce();
  });
});
