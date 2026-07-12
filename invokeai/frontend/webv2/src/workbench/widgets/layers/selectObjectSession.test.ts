import type { CanvasCompositeExportGuard } from '@workbench/canvas-engine/engine';
import type { SamSessionError } from '@workbench/canvas-engine/engineStores';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { UtilityQueueError } from '@workbench/canvas-engine/backend/utilityQueue';
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
  candidates: [{ cacheVersion: 3, layer, layerId: layer.id }],
  documentFingerprint: 'document:3',
  documentGeneration: 4,
  participants: [{ cacheVersion: 3, layer, layerId: layer.id }],
  projectId: 'p1',
};
const replacementGuard: CanvasCompositeExportGuard = {
  ...guard,
  candidates: [{ cacheVersion: 4, layer, layerId: layer.id }],
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
    runGraph: vi.fn(() =>
      Promise.resolve({ height: 12, imageName: 'result.png', origin: 'webv2:util:test', width: 16 })
    ),
    uploadIntermediate: vi.fn(() => Promise.resolve({ height: 12, imageName: 'input.png', width: 16 })),
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

  it('reports phases from the composite, upload, SAM, render, and ready boundaries', async () => {
    const harness = createHarness();
    const phases: string[] = [];
    harness.session.subscribe(() => phases.push(harness.session.getSnapshot().status));

    await expect(harness.session.process()).resolves.toBe('published');

    expect(phases).toEqual(['preparing-composite', 'uploading', 'processing-sam', 'rendering-preview', 'ready']);
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

    harness.controller.invalidateSource('p1', layer.id);
    const newPending = harness.session.process();
    await expect(newPending).resolves.toBe('published');
    const readyIndex = phases.lastIndexOf('ready');
    oldUpload.resolve({ height: 12, imageName: 'old-source.png', width: 16 });
    await expect(oldPending).resolves.toBe('stale');

    expect(phases.slice(readyIndex + 1)).toEqual([]);
    expect(harness.session.getSnapshot()).toMatchObject({ status: 'ready' });
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
    const snapshots: Array<{ error: SamSessionError | null; status: string }> = [];
    harness.session.subscribe(() => {
      const { error, status } = harness.session.getSnapshot();
      snapshots.push({ error, status });
    });
    harness.session.update({ input: { prompt: '   ', type: 'prompt' } });

    await expect(harness.session.process()).resolves.toBe('invalid');
    expect(harness.session.getSnapshot()).toMatchObject({
      error: { code: 'invalid' },
      status: 'error',
    });

    harness.session.update({ autoProcess: false });

    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'ready' });
    expect(snapshots.every(({ error, status }) => (status === 'error') === (error !== null))).toBe(true);
  });

  it('keeps a non-null error whenever processing fails', async () => {
    const harness = createHarness({ runGraph: vi.fn(() => Promise.reject(new Error('graph failed'))) });

    await expect(harness.session.process()).resolves.toBe('error');

    expect(harness.session.getSnapshot()).toEqual(
      expect.objectContaining({ error: { code: 'queue', detail: 'graph failed' }, status: 'error' })
    );
  });

  it('retries the same input after a dimension mismatch and publishes the corrected output', async () => {
    const runGraph = vi
      .fn()
      .mockResolvedValueOnce({ height: 12, imageName: 'bad.png', origin: 'test', width: 15 })
      .mockResolvedValueOnce({ height: 12, imageName: 'good.png', origin: 'test', width: 16 });
    const harness = createHarness({ runGraph });

    await expect(harness.session.process()).resolves.toBe('error');
    expect(harness.session.getSnapshot()).toMatchObject({ preview: null, status: 'error' });
    await expect(harness.session.process()).resolves.toBe('published');

    expect(runGraph).toHaveBeenCalledTimes(2);
    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'ready' });
    expect(harness.session.getSnapshot().preview?.image.imageName).toBe('good.png');
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

  it.each([
    ['no-output', new UtilityQueueError('no-output', 'No output image.')],
    ['reconcile', new UtilityQueueError('reconcile', 'Queue lookup failed.')],
    ['queue', new UtilityQueueError('failed', 'SAM backend failed.')],
  ] as const)('maps utility queue failures to %s session errors', async (code, cause) => {
    const harness = createHarness({ runGraph: vi.fn(() => Promise.reject(cause)) });

    await expect(harness.session.process()).resolves.toBe('error');

    expect(harness.session.getSnapshot()).toMatchObject({ error: { code, detail: cause.message }, status: 'error' });
  });

  it('maps upload failures to a typed upload error', async () => {
    const harness = createHarness({
      uploadIntermediate: vi.fn(() => Promise.reject(new Error('upload service unavailable'))),
    });

    await expect(harness.session.process()).resolves.toBe('error');

    expect(harness.session.getSnapshot()).toMatchObject({
      error: { code: 'upload', detail: 'upload service unavailable' },
      status: 'error',
    });
  });

  it('maps empty and not-ready sources to typed errors', async () => {
    const empty = createHarness({ exportComposite: vi.fn(() => Promise.resolve({ status: 'empty' as const })) });
    await expect(empty.session.process()).resolves.toBe('error');
    expect(empty.session.getSnapshot().error).toEqual({ code: 'empty' });

    const notReady = createHarness();
    notReady.setCurrentGuard(null);
    await expect(notReady.session.process()).resolves.toBe('error');
    expect(notReady.session.getSnapshot().error).toEqual({ code: 'not-ready' });
  });

  it('maps output dimensions and decode failures to typed errors', async () => {
    const uploadDimensions = createHarness({
      uploadIntermediate: vi.fn(() => Promise.resolve({ height: 12, imageName: 'bad-source.png', width: 15 })),
    });
    await expect(uploadDimensions.session.process()).resolves.toBe('error');
    expect(uploadDimensions.session.getSnapshot().error).toMatchObject({ code: 'output-dimension' });

    const dimensions = createHarness({
      runGraph: vi.fn(() => Promise.resolve({ height: 12, imageName: 'bad.png', origin: 'test', width: 15 })),
    });
    await expect(dimensions.session.process()).resolves.toBe('error');
    expect(dimensions.session.getSnapshot().error).toMatchObject({ code: 'output-dimension' });

    const decode = createHarness({ decodePreview: vi.fn(() => Promise.reject(new Error('corrupt PNG'))) });
    await expect(decode.session.process()).resolves.toBe('error');
    expect(decode.session.getSnapshot().error).toEqual({ code: 'decode', detail: 'corrupt PNG' });
  });

  it('accepts typed locked errors while preserving string compatibility as unknown diagnostics', () => {
    const harness = createHarness();

    harness.session.reportError({ code: 'locked' });
    expect(harness.session.getSnapshot().error).toEqual({ code: 'locked' });

    harness.session.reportError('legacy failure detail');
    expect(harness.session.getSnapshot().error).toEqual({ code: 'unknown', detail: 'legacy failure detail' });
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

    harness.controller.invalidateSource('p1', layer.id);
    const newPending = harness.session.process();

    expect(newPending).not.toBe(oldPending);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledTimes(2));
    expect(harness.deps.exportComposite).toHaveBeenCalledTimes(2);
    older.resolve({ height: 12, imageName: 'old.png', origin: 'old', width: 16 });
    await expect(oldPending).resolves.toBe('stale');
    expect(harness.session.process()).toBe(newPending);

    newer.resolve({ height: 12, imageName: 'new.png', origin: 'new', width: 16 });
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
    const oldUpload = deferred<{ height: number; imageName: string; width: number }>();
    const harness = createHarness({
      uploadIntermediate: vi
        .fn()
        .mockImplementationOnce(() => oldUpload.promise)
        .mockImplementationOnce(() => Promise.resolve({ height: 12, imageName: 'new-input.png', width: 16 })),
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
    oldUpload.resolve({ height: 12, imageName: 'old-input.png', width: 16 });
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
    const graph = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const harness = createHarness({ runGraph: vi.fn(() => graph.promise) });
    const pending = harness.session.process();
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.update({ autoProcess: true });
    expect(harness.session.getSnapshot().status).toBe('scheduled');
    await vi.advanceTimersByTimeAsync(1_000);

    expect(harness.session.getSnapshot()).toMatchObject({ error: null, status: 'processing-sam' });
    expect(vi.getTimerCount()).toBe(0);
    expect(harness.deps.runGraph).toHaveBeenCalledOnce();
    graph.resolve({ height: 12, imageName: 'result.png', origin: 'test', width: 16 });
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
    const oldGraph = deferred<{ height: number; imageName: string; origin: string; width: number }>();
    const harness = createHarness({ runGraph: vi.fn(() => oldGraph.promise) });
    harness.session.update({ autoProcess: true });
    await vi.advanceTimersByTimeAsync(1_000);
    await vi.waitFor(() => expect(harness.deps.runGraph).toHaveBeenCalledOnce());

    harness.session.update({ input: { prompt: 'dog', type: 'prompt' } });
    expect(harness.session.getSnapshot()).toMatchObject({ preview: null, status: 'scheduled' });
    oldGraph.resolve({ height: 12, imageName: 'old.png', origin: 'old', width: 16 });
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
    const graph = deferred<{ height: number; imageName: string; origin: string; width: number }>();
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
    graph.resolve({ height: 12, imageName: 'stale.png', origin: 'stale', width: 16 });
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

    harness.session.update({ isolatedPreview: false });
    graph.resolve({ height: 12, imageName: 'result.png', origin: 'test', width: 16 });

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
            : vi.fn(() => Promise.resolve({ height: 12, imageName: 'result.png', origin: 'test', width: 16 })),
        uploadIntermediate:
          boundary === 'upload'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['uploadIntermediate']>)
            : vi.fn(() => Promise.resolve({ height: 12, imageName: 'input.png', width: 16 })),
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
        wait.resolve({ height: 12, imageName: 'input.png', width: 16 });
      } else if (boundary === 'queue') {
        wait.resolve({ height: 12, imageName: 'result.png', origin: 'test', width: 16 });
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
            : vi.fn(() => Promise.resolve({ height: 12, imageName: 'result.png', origin: 'test', width: 16 })),
        uploadIntermediate:
          boundary === 'upload'
            ? vi.fn(() => wait.promise as ReturnType<SelectObjectSessionDeps<string>['uploadIntermediate']>)
            : vi.fn(() => Promise.resolve({ height: 12, imageName: 'input.png', width: 16 })),
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
        wait.resolve({ height: 12, imageName: 'input.png', width: 16 });
      } else if (boundary === 'queue') {
        wait.resolve({ height: 12, imageName: 'result.png', origin: 'test', width: 16 });
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
