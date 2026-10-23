import type { ControlMapCommitResult } from '@workbench/canvas-engine/api';
import type { CanvasControlLayerContract, CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { LayerFlushResult } from '@workbench/canvas-engine/controllers/persistenceController';
import type { BackendGraphContract } from '@workbench/graphContracts';

import { createAccountLifecycle } from '@platform/state/accountLifecycle';
import {
  controlDetectionInputKey,
  createControlLayer,
  isControlMapCurrent,
  withDetectedMap,
} from '@workbench/canvas-engine/api';
import { documentFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEditGate, type CanvasEditLease } from '@workbench/canvas-engine/editGate';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CONTROL_DETECTION_COMMIT_TIMEOUT_MS, createControlDetectionCoordinator } from './ControlDetectionCoordinator';

const DEBOUNCE_MS = 400;
const image = (imageName: string) => ({ height: 64, imageName, width: 64 });
const photoLayer = (id: string, patch: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  ...createControlLayer({ id, name: id }),
  source: { image: image(`${id}.png`), type: 'image' },
  ...patch,
});

interface PendingGraph {
  imageName: string;
  signal: AbortSignal;
  resolve(output: { height: number; imageName: string; width: number }): void;
  reject(error: unknown): void;
}

const createHarness = (initial: CanvasControlLayerContract[] = [photoLayer('a')]) => {
  const layers = new Map(initial.map((layer) => [layer.id, layer]));
  let document: CanvasDocumentContractV3 = documentFrom([...layers.values()]);
  let revision = 0;
  const publish = (): void => {
    document = documentFrom([...layers.values()]);
  };
  const setLayer = (layer: CanvasControlLayerContract): void => {
    layers.set(layer.id, layer);
    publish();
  };
  const removeLayer = (id: string): void => {
    layers.delete(id);
    publish();
  };

  const gate = createCanvasEditGate();
  let lease: CanvasEditLease | null = null;
  let mapCommitOverride: (() => ControlMapCommitResult) | null = null;
  const derivation = {
    captureControlDetectionGuard: vi.fn((layerId: string) => {
      const layer = layers.get(layerId);
      return layer
        ? { documentRevision: revision, inputKey: controlDetectionInputKey(layer), layerId, projectId: 'p1' }
        : null;
    }),
    commitControlLayerPatch: vi.fn(
      (options: { next: CanvasControlLayerContract }): { status: 'committed' } | { status: 'busy' } => {
        setLayer(options.next);
        return { status: 'committed' };
      }
    ),
    commitControlMap: vi.fn(
      (options: {
        guard: { layerId: string; documentRevision: number; inputKey: string };
        image: { height: number; imageName: string; width: number };
        rect: { height: number; width: number; x: number; y: number };
        detection: { type: string; settings: Record<string, unknown> };
      }): Promise<ControlMapCommitResult> => {
        if (mapCommitOverride) {
          return Promise.resolve(mapCommitOverride());
        }
        // The engine refuses while a modal operation holds the edit lease.
        if (lease?.isCurrent()) {
          return Promise.resolve({ status: 'busy' });
        }
        const layer = layers.get(options.guard.layerId);
        if (!layer) {
          return Promise.resolve({ status: 'missing' });
        }
        if (options.guard.documentRevision !== revision || options.guard.inputKey !== controlDetectionInputKey(layer)) {
          return Promise.resolve({ status: 'stale' });
        }
        setLayer(withDetectedMap(layer, { image: options.image, rect: options.rect }, options.detection)!);
        return Promise.resolve({ status: 'committed' });
      }
    ),
  };

  const graphs: PendingGraph[] = [];
  const runFilterGraph = vi.fn(
    (options: { graph: BackendGraphContract; signal?: AbortSignal }) =>
      new Promise<{ height: number; imageName: string; width: number }>((resolve, reject) => {
        const flatten = options.graph.nodes.control_filter_flatten as { image?: { image_name: string } } | undefined;
        graphs.push({
          imageName: flatten?.image?.image_name ?? '',
          reject,
          resolve,
          signal: options.signal!,
        });
      })
  );
  const lockListeners = new Set<() => void>();
  const accounts = createAccountLifecycle();
  const flushLayerUploads = vi.fn((_layerId: string, _options: { signal?: AbortSignal }) =>
    Promise.resolve<LayerFlushResult>('clean')
  );
  const makeDurable = vi.fn((_imageName: string) => Promise.resolve());
  const coordinator = createControlDetectionCoordinator({
    accounts: { capture: accounts.capture, register: accounts.register },
    derivation,
    flushLayerUploads,
    getDocument: () => document,
    makeDurable,
    projectId: 'p1',
    runFilterGraph,
    subscribeLockChanges: (listener) => {
      lockListeners.add(listener);
      return () => lockListeners.delete(listener);
    },
  });

  let mapCount = 0;
  /** Completes the oldest pending graph with a 64×64 map. */
  const finishGraph = async (index = 0): Promise<void> => {
    const [pending] = graphs.splice(index, 1);
    mapCount += 1;
    pending!.resolve({ height: 64, imageName: `map-${String(mapCount)}.png`, width: 64 });
    await vi.advanceTimersByTimeAsync(0);
  };

  return {
    accounts,
    acquireLease: () => {
      lease = gate.tryAcquire({ kind: 'filter', layerId: 'raster' });
      return lease;
    },
    bumpRevision: () => {
      revision += 1;
    },
    coordinator,
    derivation,
    finishGraph,
    flushLayerUploads,
    graphs,
    layer: (id = 'a') => layers.get(id)!,
    lockListeners,
    makeDurable,
    emitLockChange: () => {
      for (const listener of lockListeners) {
        listener();
      }
    },
    removeLayer,
    runFilterGraph,
    setLayer,
    setMapCommit: (override: (() => ControlMapCommitResult) | null) => {
      mapCommitOverride = override;
    },
    status: (id = 'a') => coordinator.getStatus(id),
  };
};

const settle = (ms = DEBOUNCE_MS) => vi.advanceTimersByTimeAsync(ms);

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('ControlDetectionCoordinator', () => {
  it('debounces intents, runs one graph on the input and commits a durable map', async () => {
    const h = createHarness();

    expect(h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' })).toBe('updated');
    expect(h.status()).toEqual({ status: 'queued' });
    await settle(DEBOUNCE_MS - 1);
    expect(h.coordinator.applyIntent('a', { kind: 'set-detector-settings', settings: { low_threshold: 50 } })).toBe(
      'updated'
    );
    await settle(DEBOUNCE_MS - 1);
    expect(h.runFilterGraph).not.toHaveBeenCalled();
    await settle(1);

    expect(h.runFilterGraph).toHaveBeenCalledOnce();
    expect(h.graphs[0]!.imageName).toBe('a.png');
    expect(h.status()).toEqual({ detector: 'canny_edge_detection', status: 'processing' });
    expect(h.derivation.commitControlLayerPatch).toHaveBeenLastCalledWith(
      expect.objectContaining({ coalesceKey: 'a:canny_edge_detection:settings' })
    );

    await h.finishGraph();

    expect(h.makeDurable).toHaveBeenCalledExactlyOnceWith('map-1.png');
    expect(h.layer().source).toMatchObject({ bitmap: { imageName: 'map-1.png' }, type: 'paint' });
    expect(isControlMapCurrent(h.layer())).toBe(true);
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('maps refused patches without scheduling, and settles layers that need no detection', async () => {
    const h = createHarness();
    h.derivation.commitControlLayerPatch.mockReturnValueOnce({ status: 'busy' });

    expect(h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' })).toBe('blocked');
    expect(h.coordinator.applyIntent('missing', { kind: 'draw' })).toBe('missing');
    expect(h.coordinator.applyIntent('a', { controlType: 'tile', kind: 'set-type' })).toBe('updated');
    expect(h.coordinator.applyIntent('a', { controlType: 'tile', kind: 'set-type' })).toBe('unchanged');
    await settle();

    expect(h.runFilterGraph).not.toHaveBeenCalled();
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('aborts a running job when a newer intent changes its input, and keeps it otherwise', async () => {
    const h = createHarness();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    const first = h.graphs.shift()!;

    expect(h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type', name: 'Edges 1' })).toBe('updated');
    await settle();
    expect(first.signal.aborted).toBe(false);
    expect(h.runFilterGraph).toHaveBeenCalledOnce();

    h.coordinator.applyIntent('a', { detector: 'mlsd_detection', kind: 'set-detector' });
    expect(first.signal.aborted).toBe(true);
    await settle();
    expect(h.runFilterGraph).toHaveBeenCalledTimes(2);

    first.resolve({ height: 64, imageName: 'late.png', width: 64 });
    await settle(0);
    expect(h.derivation.commitControlMap).not.toHaveBeenCalled();
    await h.finishGraph();
    expect(h.makeDurable).toHaveBeenCalledExactlyOnceWith('map-1.png');
    expect(h.layer().filter?.type).toBe('mlsd_detection');
  });

  it.each([
    ['layer removal', (h: ReturnType<typeof createHarness>) => h.removeLayer('a')],
    ['a document swap', (h: ReturnType<typeof createHarness>) => h.bumpRevision()],
    [
      'a different detector (undo)',
      (h: ReturnType<typeof createHarness>) =>
        h.setLayer({ ...h.layer(), detection: { settings: {}, type: 'mlsd_detection' } }),
    ],
    [
      'an undo of a replaced image',
      (h: ReturnType<typeof createHarness>) =>
        h.setLayer({ ...h.layer(), source: { image: image('before-replace.png'), type: 'image' } }),
    ],
  ])('drops a result made stale by %s before preserving it', async (_label, staleIt) => {
    const h = createHarness();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();

    staleIt(h);
    await h.finishGraph();
    await settle();

    expect(h.makeDurable).not.toHaveBeenCalled();
    expect(h.derivation.commitControlMap).not.toHaveBeenCalled();
    expect(h.runFilterGraph).toHaveBeenCalledOnce();
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('restarts after the layer’s own paint lands under a run, debounced and at most three times', async () => {
    const paint = (name: string) => ({ bitmap: image(name), offset: { x: 0, y: 0 }, type: 'paint' }) as const;
    const h = createHarness([photoLayer('a', { controlType: 'edges', detection: null, source: paint('strokes.png') })]);
    h.coordinator.applyIntent('a', { detector: 'canny_edge_detection', kind: 'set-detector' });
    await settle();

    for (let run = 1; run <= 4; run += 1) {
      h.setLayer({ ...h.layer(), source: paint(`strokes-v${String(run)}.png`) });
      await h.finishGraph();
      if (run < 4) {
        expect(h.status()).toEqual({ status: 'queued' });
        await settle(DEBOUNCE_MS - 1);
        expect(h.runFilterGraph).toHaveBeenCalledTimes(run);
        await settle(1);
        expect(h.graphs[0]!.imageName).toBe(`strokes-v${String(run)}.png`);
      }
    }
    await settle();

    expect(h.runFilterGraph).toHaveBeenCalledTimes(4);
    expect(h.derivation.commitControlMap).not.toHaveBeenCalled();
    expect(h.makeDurable).not.toHaveBeenCalled();
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('lets a modal operation take the edit lease during a run, then commits after it releases', async () => {
    const h = createHarness();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();

    const lease = h.acquireLease();
    expect(lease).not.toBeNull();
    await h.finishGraph();
    expect(h.derivation.commitControlMap).toHaveBeenCalledOnce();
    expect(h.status()).toEqual({ status: 'queued' });

    lease!.release();
    h.emitLockChange();
    await settle(0);

    expect(h.derivation.commitControlMap).toHaveBeenCalledTimes(2);
    expect(h.makeDurable).toHaveBeenCalledOnce();
    expect(isControlMapCurrent(h.layer())).toBe(true);
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('retries a busy commit with backoff and reports commit-busy after the timeout', async () => {
    const h = createHarness();
    h.setMapCommit(() => ({ status: 'busy' }));
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    await h.finishGraph();

    const attempts = (): number => h.derivation.commitControlMap.mock.calls.length;
    expect(attempts()).toBe(1);
    await settle(249);
    expect(attempts()).toBe(1);
    await settle(1);
    expect(attempts()).toBe(2);
    await settle(500);
    expect(attempts()).toBe(3);
    await settle(1_000);
    expect(attempts()).toBe(4);
    await settle(2_000);
    expect(attempts()).toBe(5);
    expect(h.status()).toEqual({ status: 'queued' });

    await settle(CONTROL_DETECTION_COMMIT_TIMEOUT_MS);
    expect(h.status()).toEqual({ error: { code: 'commit-busy' }, status: 'error' });
    const final = attempts();
    await settle(10_000);
    expect(attempts()).toBe(final);
  });

  it.each([
    [{ message: 'too big', status: 'over-budget' } as const, { code: 'over-budget', detail: 'too big' }],
    [{ status: 'unrepresentable' } as const, { code: 'unrepresentable' }],
    [{ message: 'decode failed', status: 'failed' } as const, { code: 'apply-failed', detail: 'decode failed' }],
  ])('reports a %o commit as a typed error', async (result, error) => {
    const h = createHarness();
    h.setMapCommit(() => result);
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    await h.finishGraph();

    expect(h.status()).toEqual({ error, status: 'error' });
  });

  it('reports graph failures and unknown detectors, and Retry runs again', async () => {
    const h = createHarness();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    h.graphs.shift()!.reject(new Error('queue offline'));
    await settle(0);
    expect(h.status()).toEqual({ error: { code: 'process-failed', detail: 'queue offline' }, status: 'error' });

    expect(h.coordinator.retry('a')).toBe('scheduled');
    await settle(0);
    expect(h.runFilterGraph).toHaveBeenCalledTimes(2);
    await h.finishGraph();
    expect(h.coordinator.retry('a')).toBe('not-needed');
    expect(h.coordinator.retry('missing')).toBe('missing');

    h.coordinator.applyIntent('a', { detector: 'retired_detector', kind: 'set-detector' });
    await settle();
    expect(h.status()).toEqual({ error: { code: 'unknown-detector', detail: 'retired_detector' }, status: 'error' });
    expect(h.runFilterGraph).toHaveBeenCalledTimes(2);
  });

  it('flushes the layer’s own paint before detecting it', async () => {
    const h = createHarness([
      photoLayer('a', { controlType: 'edges', detection: null, source: { bitmap: null, type: 'paint' } }),
    ]);
    h.flushLayerUploads.mockImplementation(() => {
      h.setLayer({ ...h.layer(), source: { bitmap: image('strokes.png'), offset: { x: 3, y: 4 }, type: 'paint' } });
      return Promise.resolve('flushed');
    });

    h.coordinator.applyIntent('a', { detector: 'canny_edge_detection', kind: 'set-detector' });
    await settle();

    expect(h.flushLayerUploads).toHaveBeenCalledExactlyOnceWith('a', { signal: expect.any(AbortSignal) });
    expect(h.graphs[0]!.imageName).toBe('strokes.png');
    await h.finishGraph();
    expect(h.layer().original).toMatchObject({ bitmap: { imageName: 'strokes.png' } });
  });

  it('never schedules detection from document changes', async () => {
    const h = createHarness();
    // An undo back to a typed layer whose map is stale.
    h.setLayer({ ...h.layer(), controlType: 'edges', detection: { settings: {}, type: 'canny_edge_detection' } });
    h.coordinator.reconcileWithDocument();
    await settle(10_000);

    expect(h.runFilterGraph).not.toHaveBeenCalled();
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('runs at most two graphs at once', async () => {
    const h = createHarness([photoLayer('a'), photoLayer('b'), photoLayer('c')]);
    for (const id of ['a', 'b', 'c']) {
      h.coordinator.applyIntent(id, { controlType: 'edges', kind: 'set-type' });
    }
    await settle();

    expect(h.runFilterGraph).toHaveBeenCalledTimes(2);
    expect(h.status('c')).toEqual({ status: 'queued' });
    await h.finishGraph();
    expect(h.runFilterGraph).toHaveBeenCalledTimes(3);
    expect(h.graphs.map((graph) => graph.imageName)).toEqual(['b.png', 'c.png']);
  });

  it('suspends on cooldown and resumes interrupted layers on activation', async () => {
    const h = createHarness([photoLayer('a'), photoLayer('b')]);
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    const interrupted = h.graphs[0]!;

    h.coordinator.suspend();
    expect(interrupted.signal.aborted).toBe(true);
    expect(h.status()).toEqual({ status: 'queued' });
    h.coordinator.applyIntent('b', { controlType: 'edges', kind: 'set-type' });
    await settle(10_000);
    expect(h.runFilterGraph).toHaveBeenCalledOnce();

    h.coordinator.resume();
    await settle(0);
    expect(h.runFilterGraph).toHaveBeenCalledTimes(3);
  });

  it('drops every job and status when the account changes', async () => {
    const h = createHarness();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    const running = h.graphs[0]!;

    h.accounts.invalidate();
    expect(running.signal.aborted).toBe(true);
    await h.finishGraph();

    expect(h.derivation.commitControlMap).not.toHaveBeenCalled();
    expect(h.status()).toEqual({ status: 'idle' });
  });

  it('drops jobs and statuses of removed layers', async () => {
    const h = createHarness();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    h.removeLayer('a');
    h.coordinator.reconcileWithDocument();
    await settle();

    expect(h.status()).toEqual({ status: 'idle' });
    expect(h.runFilterGraph).not.toHaveBeenCalled();
  });

  it('settles a leftover error once the document no longer needs detection, without scheduling', async () => {
    const h = createHarness();
    const untyped = h.layer();
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    h.graphs.shift()!.reject(new Error('queue offline'));
    await settle(0);
    expect(h.status()).toMatchObject({ status: 'error' });

    // Unrelated document changes keep the error while the map is still stale.
    h.coordinator.reconcileWithDocument();
    expect(h.status()).toMatchObject({ status: 'error' });

    // Undo of the type change.
    h.setLayer(untyped);
    h.coordinator.reconcileWithDocument();
    await settle(10_000);

    expect(h.status()).toEqual({ status: 'idle' });
    expect(h.runFilterGraph).toHaveBeenCalledOnce();
  });

  it('reports a failed paint flush as source-not-ready', async () => {
    const h = createHarness([
      photoLayer('a', { controlType: 'edges', detection: null, source: { bitmap: null, type: 'paint' } }),
    ]);
    h.flushLayerUploads.mockResolvedValue('failed');

    h.coordinator.applyIntent('a', { detector: 'canny_edge_detection', kind: 'set-detector' });
    await settle();

    expect(h.status()).toEqual({ error: { code: 'source-not-ready' }, status: 'error' });
    expect(h.runFilterGraph).not.toHaveBeenCalled();
  });

  it('reports a result that cannot be preserved as apply-failed, without committing it', async () => {
    const h = createHarness();
    h.makeDurable.mockRejectedValue(new Error('PATCH 500'));
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    await h.finishGraph();

    expect(h.status()).toEqual({ error: { code: 'apply-failed', detail: 'PATCH 500' }, status: 'error' });
    expect(h.derivation.commitControlMap).not.toHaveBeenCalled();
  });

  it('gives up a queued slot when its job is cancelled', async () => {
    const h = createHarness([photoLayer('a'), photoLayer('b'), photoLayer('c'), photoLayer('d')]);
    for (const id of ['a', 'b', 'c', 'd']) {
      h.coordinator.applyIntent(id, { controlType: 'edges', kind: 'set-type' });
    }
    await settle();
    expect(h.runFilterGraph).toHaveBeenCalledTimes(2);

    // `c` waits for a slot; None cancels it.
    h.coordinator.applyIntent('c', { detector: null, kind: 'set-detector' });
    expect(h.status('c')).toEqual({ status: 'idle' });
    await h.finishGraph();

    expect(h.runFilterGraph).toHaveBeenCalledTimes(3);
    expect(h.graphs.map((graph) => graph.imageName)).toEqual(['b.png', 'd.png']);
  });

  it('disposes idempotently, cancelling work and releasing subscriptions', async () => {
    const h = createHarness();
    const listener = vi.fn();
    h.coordinator.subscribeStatus('a', listener);
    h.coordinator.applyIntent('a', { controlType: 'edges', kind: 'set-type' });
    await settle();
    const running = h.graphs[0]!;

    h.coordinator.dispose();
    h.coordinator.dispose();

    expect(running.signal.aborted).toBe(true);
    expect(h.lockListeners.size).toBe(0);
    expect(h.status()).toEqual({ status: 'idle' });
    expect(h.coordinator.applyIntent('a', { kind: 'draw' })).toBe('blocked');
    expect(h.coordinator.retry('a')).toBe('missing');
    await h.finishGraph();
    expect(h.derivation.commitControlMap).not.toHaveBeenCalled();
  });
});
