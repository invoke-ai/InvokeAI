/**
 * Background detection for control layers. Commands start it (a Properties intent, Retry, resuming after cooldown);
 * document changes never do, so undo and redo cannot re-run detection. One job per layer, at most
 * {@link CONTROL_DETECTION_MAX_CONCURRENT} graphs per engine; results land through the engine's guarded map commit,
 * which refuses while a modal operation, gesture or layer session holds the document, so a finished result waits
 * and retries instead of blocking them. The commit promotes a result to durable pixels only once it is about to
 * land, so a dropped result stays an intermediate the server collects.
 */

import type { AccountOwnedResource, AccountScope } from '@platform/state/accountLifecycle';
import type { CanvasControlDerivationCapability, ControlDetectionGuard } from '@workbench/canvas-engine/api';
import type {
  CanvasControlDetectionContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
} from '@workbench/canvas-engine/contracts';
import type { LayerFlushResult } from '@workbench/canvas-engine/controllers/persistenceController';
import type {
  ControlDetectionErrorCode,
  ControlDetectionStatus,
  ControlLayerIntent,
  ControlLayerIntentResult,
  RetryControlDetectionResult,
} from '@workbench/canvas-operations/operationTypes';
import type { BackendGraphContract } from '@workbench/graphContracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { captureAccountScope, registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import {
  controlDetectionInputKey,
  getControlDetectionInput,
  isControlMapCurrent,
  lookupDocumentLeaf,
  needsControlDetection,
} from '@workbench/canvas-engine/api';
import {
  createControlDetectionStatusStore,
  IDLE_CONTROL_DETECTION_STATUS,
  type ControlDetectionStatusStore,
} from '@workbench/canvas-operations/controlDetectionStatusStore';
import { applyControlLayerIntent } from '@workbench/canvas-operations/controlLayerIntents';
import { buildFilterDefaults, getFilterDefinition } from '@workbench/canvas-operations/filterGraphs';
import { FILTER_AUTO_PROCESS_DEBOUNCE_MS } from '@workbench/canvas-operations/filterOperationSession';

import { ControlDetectionFailure, runControlDetection, type ControlDetectionMap } from './controlDetectionRunner';

export const CONTROL_DETECTION_MAX_CONCURRENT = 2;
export const CONTROL_DETECTION_COMMIT_RETRY_MS = { initial: 250, max: 2_000 } as const;
/** How long a finished result keeps waiting for the document before it reports `commit-busy`. */
export const CONTROL_DETECTION_COMMIT_TIMEOUT_MS = 30_000;
/** Restarts after the input pixels changed under a running job (painting, an upload landing). */
export const CONTROL_DETECTION_MAX_RESTARTS = 3;

/** The engine's control derivation capability. */
export type ControlDerivationPort = Pick<
  CanvasControlDerivationCapability,
  'captureControlDetectionGuard' | 'commitControlLayerPatch' | 'commitControlMap'
>;

export interface ControlDetectionCoordinatorDeps {
  readonly projectId: string;
  readonly derivation: ControlDerivationPort;
  getDocument(): CanvasDocumentContractV3 | null;
  /** Persists the layer's pending paint, waiting for an edit that holds it. */
  flushLayerUploads(layerId: string, options: { signal?: AbortSignal }): Promise<LayerFlushResult>;
  runFilterGraph(options: {
    graph: BackendGraphContract;
    outputNodeId: string;
    signal?: AbortSignal;
  }): Promise<{ height: number; imageName: string; width: number }>;
  /** Adopts a result as canvas-owned pixels; called by the map commit once nothing refuses it. */
  makeDurable(imageName: string): Promise<void>;
  /** Notifies when the edit lease or an editing session may have been released. */
  subscribeLockChanges(listener: () => void): () => void;
  readonly statuses?: ControlDetectionStatusStore;
  /** Account lifetime; the platform lifecycle by default. */
  readonly accounts?: {
    capture(): AccountScope;
    register(resource: AccountOwnedResource): () => void;
  };
}

export interface ControlDetectionCoordinator {
  applyIntent(layerId: string, intent: ControlLayerIntent): ControlLayerIntentResult;
  retry(layerId: string): RetryControlDetectionResult;
  getStatus(layerId: string): ControlDetectionStatus;
  subscribeStatus(layerId: string, listener: () => void): () => void;
  /** Cancels every job and remembers its layer, for {@link resume}. */
  suspend(): void;
  /** Restarts the layers {@link suspend} interrupted that still need a map. */
  resume(): void;
  /**
   * Follows a document change without scheduling work: drops jobs and statuses of removed layers, cancels a job whose
   * layer no longer needs it or whose input changed other than by its own paint landing (undo, redo), and settles a
   * leftover error or queued status once the layer needs no detection (an undo past the failed edit).
   */
  reconcileWithDocument(): void;
  dispose(): void;
}

const INTENT_LABELS: Readonly<Record<ControlLayerIntent['kind'], string>> = {
  draw: 'Draw control image',
  'replace-image': 'Replace control image',
  'set-detector': 'Change detector',
  'set-detector-settings': 'Change detector settings',
  'set-type': 'Change control type',
};

const PATCH_REFUSALS: Readonly<
  Record<
    Exclude<ReturnType<ControlDerivationPort['commitControlLayerPatch']>['status'], 'committed'>,
    ControlLayerIntentResult
  >
> = {
  busy: 'blocked',
  failed: 'failed',
  locked: 'locked',
  missing: 'missing',
  'not-ready': 'blocked',
  'over-budget': 'over-budget',
  stale: 'blocked',
};

const QUEUED: ControlDetectionStatus = Object.freeze({ status: 'queued' });

type JobOutcome =
  | { kind: 'committed' | 'drop' | 'restart' }
  | { kind: 'error'; code: ControlDetectionErrorCode; detail?: string };

interface Job {
  readonly layerId: string;
  readonly controller: AbortController;
  /** The input key it detects (updated once a flush settles the paint); a command for the same key keeps it. */
  inputKey: string;
  /** The layer `inputKey` was read from, to tell the layer's own paint landing apart from other input changes. */
  recipe: CanvasControlLayerContract;
  readonly restarts: number;
  timer: ReturnType<typeof setTimeout> | null;
  running: boolean;
  /**
   * While its map commit runs: the commit's own publish changes the document and reconciles synchronously, and the
   * engine's checks already settle what the job's result may still do.
   */
  committing: boolean;
  /** Ends a commit retry wait early (a lock was released). */
  wake: (() => void) | null;
}

const detectorDefaults = (detector: string): Record<string, unknown> => {
  const definition = getFilterDefinition(detector);
  return definition ? buildFilterDefaults(definition) : {};
};

/** Whether the detection input is the layer's own paint, whose newest strokes may not be uploaded yet. */
const isOwnPaintInput = (layer: CanvasControlLayerContract): boolean =>
  layer.original === undefined && !layer.filter && layer.source.type === 'paint';

/**
 * Whether a run could produce a map: the document says the map is stale, or the input is the layer's own paint
 * whose pixels may still be in flight (a flush settles which).
 */
const wantsDetection = (layer: CanvasControlLayerContract): boolean =>
  needsControlDetection(layer) ||
  (layer.controlType !== undefined && !!layer.detection && !isControlMapCurrent(layer) && isOwnPaintInput(layer));

/**
 * Whether `after` differs from `before` only by its own paint landing (an upload, or strokes on the input): same type
 * and detector, and both detect their own paint source. An image swap (undo or redo of a replaced image) does not
 * qualify.
 */
const onlyOwnPaintChanged = (before: CanvasControlLayerContract, after: CanvasControlLayerContract): boolean =>
  isOwnPaintInput(before) &&
  isOwnPaintInput(after) &&
  before.controlType === after.controlType &&
  areJsonValuesStructurallyEqual(before.detection ?? null, after.detection ?? null);

const failureDetail = (error: unknown): string | undefined => {
  const detail = error instanceof Error ? error.message : typeof error === 'string' ? error : undefined;
  return detail || undefined;
};

const errorOutcome = (code: ControlDetectionErrorCode, detail?: string): JobOutcome =>
  detail ? { code, detail, kind: 'error' } : { code, kind: 'error' };

let coordinatorCount = 0;

export const createControlDetectionCoordinator = (
  deps: ControlDetectionCoordinatorDeps
): ControlDetectionCoordinator => {
  const statuses = deps.statuses ?? createControlDetectionStatusStore();
  const accounts = deps.accounts ?? { capture: captureAccountScope, register: registerAccountOwnedResource };
  const jobs = new Map<string, Job>();
  const interrupted = new Set<string>();
  let suspended = false;
  let disposed = false;

  let runningGraphs = 0;
  const slotWaiters: (() => void)[] = [];
  const acquireSlot = (signal: AbortSignal): Promise<void> => {
    signal.throwIfAborted();
    if (runningGraphs < CONTROL_DETECTION_MAX_CONCURRENT) {
      runningGraphs += 1;
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const grant = (): void => {
        signal.removeEventListener('abort', onAbort);
        resolve();
      };
      const onAbort = (): void => {
        slotWaiters.splice(slotWaiters.indexOf(grant), 1);
        reject(signal.reason);
      };
      slotWaiters.push(grant);
      signal.addEventListener('abort', onAbort, { once: true });
    });
  };
  /** Hands the slot straight to the next waiter, so the count never dips below a queued job. */
  const releaseSlot = (): void => {
    const next = slotWaiters.shift();
    if (next) {
      next();
    } else {
      runningGraphs -= 1;
    }
  };

  const readLayer = (layerId: string): CanvasControlLayerContract | null => {
    const layer = lookupDocumentLeaf(deps.getDocument(), layerId)?.layer;
    return layer?.type === 'control' ? layer : null;
  };

  /** Writes only while `job` is its layer's job, so a cancelled job never overwrites its successor's status. */
  const setJobStatus = (job: Job, status: ControlDetectionStatus): void => {
    if (jobs.get(job.layerId) === job) {
      statuses.set(job.layerId, status);
    }
  };

  const cancelJob = (layerId: string): void => {
    const job = jobs.get(layerId);
    if (!job) {
      return;
    }
    jobs.delete(layerId);
    if (job.timer !== null) {
      clearTimeout(job.timer);
      job.timer = null;
    }
    job.controller.abort();
  };

  const waitForRetry = (job: Job, delayMs: number, signal: AbortSignal): Promise<void> =>
    new Promise((resolve, reject) => {
      const settle = (): void => {
        clearTimeout(timer);
        job.wake = null;
        signal.removeEventListener('abort', onAbort);
      };
      const done = (): void => {
        settle();
        resolve();
      };
      const onAbort = (): void => {
        settle();
        reject(signal.reason);
      };
      const timer = setTimeout(done, delayMs);
      job.wake = done;
      signal.addEventListener('abort', onAbort, { once: true });
    });

  /**
   * Whether `guard` still describes the layer. The layer's own paint landing under the job (an upload, more strokes)
   * runs again; anything else (removal, a document swap, an image swapped by undo or redo, another command) drops
   * the result, so history never re-runs detection.
   */
  const checkGuard = (
    guard: ControlDetectionGuard,
    recipe: CanvasControlLayerContract
  ): 'current' | 'drop' | 'restart' => {
    const current = deps.derivation.captureControlDetectionGuard(guard.layerId);
    if (!current || current.documentRevision !== guard.documentRevision) {
      return 'drop';
    }
    if (current.inputKey === guard.inputKey) {
      return 'current';
    }
    const layer = readLayer(guard.layerId);
    return layer && needsControlDetection(layer) && onlyOwnPaintChanged(recipe, layer) ? 'restart' : 'drop';
  };

  const commitWithRetry = async (
    job: Job,
    guard: ControlDetectionGuard,
    recipe: CanvasControlLayerContract,
    detection: CanvasControlDetectionContract,
    map: ControlDetectionMap,
    signal: AbortSignal
  ): Promise<JobOutcome> => {
    const startedAt = Date.now();
    let delay: number = CONTROL_DETECTION_COMMIT_RETRY_MS.initial;
    let durable = false;
    const makeDurable = async (): Promise<void> => {
      if (!durable) {
        await deps.makeDurable(map.image.imageName);
        durable = true;
      }
    };
    for (;;) {
      job.committing = true;
      let result: Awaited<ReturnType<ControlDerivationPort['commitControlMap']>>;
      try {
        result = await deps.derivation.commitControlMap({
          detection,
          guard,
          image: map.image,
          makeDurable,
          rect: map.rect,
          signal,
        });
      } finally {
        job.committing = false;
      }
      switch (result.status) {
        case 'committed':
          return { kind: 'committed' };
        case 'stale': {
          const check = checkGuard(guard, recipe);
          return { kind: check === 'restart' ? 'restart' : 'drop' };
        }
        case 'missing':
        case 'aborted':
          signal.throwIfAborted();
          return { kind: 'drop' };
        case 'over-budget':
          return errorOutcome('over-budget', result.message);
        case 'unrepresentable':
          return errorOutcome('unrepresentable', result.message);
        case 'failed':
          return errorOutcome('apply-failed', result.message);
        case 'busy':
        case 'locked':
        case 'not-ready': {
          const remaining = CONTROL_DETECTION_COMMIT_TIMEOUT_MS - (Date.now() - startedAt);
          if (remaining <= 0) {
            return errorOutcome('commit-busy', result.message);
          }
          setJobStatus(job, QUEUED);
          await waitForRetry(job, Math.min(delay, remaining), signal);
          delay = Math.min(delay * 2, CONTROL_DETECTION_COMMIT_RETRY_MS.max);
          break;
        }
      }
    }
  };

  const detect = async (job: Job, signal: AbortSignal): Promise<JobOutcome> => {
    let layer = readLayer(job.layerId);
    if (!layer || !wantsDetection(layer)) {
      return { kind: 'drop' };
    }
    if (isOwnPaintInput(layer)) {
      const flushed = await deps.flushLayerUploads(job.layerId, { signal });
      signal.throwIfAborted();
      if (flushed === 'aborted') {
        return { kind: 'drop' };
      }
      if (flushed === 'failed') {
        return errorOutcome('source-not-ready');
      }
      layer = readLayer(job.layerId);
    }
    if (!layer || !needsControlDetection(layer) || !layer.detection) {
      return { kind: 'drop' };
    }
    const input = getControlDetectionInput(layer);
    const guard = deps.derivation.captureControlDetectionGuard(job.layerId);
    if (!input || !guard) {
      return { kind: 'drop' };
    }
    job.inputKey = guard.inputKey;
    job.recipe = layer;
    const recipe = layer;
    const { detection } = layer;

    await acquireSlot(signal);
    let map: ControlDetectionMap;
    try {
      setJobStatus(job, { detector: detection.type, status: 'processing' });
      map = await runControlDetection({ detection, input, runFilterGraph: deps.runFilterGraph, signal });
    } finally {
      releaseSlot();
    }

    const check = checkGuard(guard, recipe);
    if (check !== 'current') {
      return { kind: check };
    }
    return commitWithRetry(job, guard, recipe, detection, map, signal);
  };

  const schedule = (layerId: string, delayMs: number, layer: CanvasControlLayerContract, restarts = 0): void => {
    if (disposed) {
      return;
    }
    if (suspended) {
      interrupted.add(layerId);
      statuses.set(layerId, QUEUED);
      return;
    }
    const inputKey = controlDetectionInputKey(layer);
    const existing = jobs.get(layerId);
    if (existing?.running && existing.inputKey === inputKey) {
      return;
    }
    cancelJob(layerId);
    const job: Job = {
      committing: false,
      controller: new AbortController(),
      inputKey,
      layerId,
      recipe: layer,
      restarts,
      running: false,
      timer: null,
      wake: null,
    };
    jobs.set(layerId, job);
    statuses.set(layerId, QUEUED);
    job.timer = setTimeout(() => {
      job.timer = null;
      void run(job);
    }, delayMs);
  };

  const finish = (job: Job, outcome: JobOutcome): void => {
    if (jobs.get(job.layerId) !== job) {
      return;
    }
    jobs.delete(job.layerId);
    if (outcome.kind === 'error') {
      statuses.set(
        job.layerId,
        outcome.detail
          ? { error: { code: outcome.code, detail: outcome.detail }, status: 'error' }
          : { error: { code: outcome.code }, status: 'error' }
      );
      return;
    }
    const layer = outcome.kind === 'restart' ? readLayer(job.layerId) : null;
    if (layer && job.restarts < CONTROL_DETECTION_MAX_RESTARTS) {
      schedule(job.layerId, FILTER_AUTO_PROCESS_DEBOUNCE_MS, layer, job.restarts + 1);
      return;
    }
    statuses.set(job.layerId, IDLE_CONTROL_DETECTION_STATUS);
  };

  const run = async (job: Job): Promise<void> => {
    job.running = true;
    const signal = AbortSignal.any([job.controller.signal, accounts.capture().signal]);
    let outcome: JobOutcome;
    try {
      outcome = await detect(job, signal);
    } catch (error) {
      if (signal.aborted) {
        // A cancel already settled the status; an account switch alone leaves nothing to show.
        if (jobs.get(job.layerId) === job) {
          jobs.delete(job.layerId);
          statuses.set(job.layerId, IDLE_CONTROL_DETECTION_STATUS);
        }
        return;
      }
      outcome =
        error instanceof ControlDetectionFailure
          ? errorOutcome(error.code, error.detail)
          : errorOutcome('process-failed', failureDetail(error));
    }
    finish(job, outcome);
  };

  const cancelAll = (): void => {
    for (const layerId of jobs.keys()) {
      cancelJob(layerId);
    }
  };

  const unsubscribeLocks = deps.subscribeLockChanges(() => {
    for (const job of jobs.values()) {
      job.wake?.();
    }
  });
  const unregisterAccount = accounts.register({
    clear: () => {
      cancelAll();
      interrupted.clear();
      statuses.clear();
    },
    name: `canvas-control-detection:${deps.projectId}:${String((coordinatorCount += 1))}`,
  });

  const settleWithoutDetection = (layerId: string): void => {
    cancelJob(layerId);
    interrupted.delete(layerId);
    statuses.set(layerId, IDLE_CONTROL_DETECTION_STATUS);
  };

  return {
    applyIntent: (layerId, intent) => {
      if (disposed) {
        return 'blocked';
      }
      const layer = readLayer(layerId);
      if (!layer) {
        return 'missing';
      }
      const next = applyControlLayerIntent(layer, intent, { detectorDefaults });
      if (next === 'no-detection-input') {
        return next;
      }
      if (next === layer) {
        return 'unchanged';
      }
      const result = deps.derivation.commitControlLayerPatch({
        coalesceKey:
          intent.kind === 'set-detector-settings'
            ? `${layerId}:${layer.detection?.type ?? 'none'}:settings`
            : undefined,
        expected: layer,
        label: INTENT_LABELS[intent.kind],
        next,
      });
      if (result.status !== 'committed') {
        return PATCH_REFUSALS[result.status];
      }
      if (wantsDetection(next)) {
        schedule(layerId, FILTER_AUTO_PROCESS_DEBOUNCE_MS, next);
      } else {
        settleWithoutDetection(layerId);
      }
      return 'updated';
    },
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      cancelAll();
      interrupted.clear();
      unsubscribeLocks();
      unregisterAccount();
      statuses.clear();
    },
    getStatus: statuses.get,
    reconcileWithDocument: () => {
      const tracked = new Set([...jobs.keys(), ...interrupted, ...statuses.layerIds()]);
      for (const layerId of tracked) {
        const layer = readLayer(layerId);
        if (!layer) {
          settleWithoutDetection(layerId);
          continue;
        }
        const job = jobs.get(layerId);
        if (job?.committing) {
          continue;
        }
        if (job) {
          // History moved the layer away from what the job detects: its result would be dropped, so stop the graph
          // now. The layer's own paint landing keeps it, to restart once it finishes.
          const inputChanged =
            controlDetectionInputKey(layer) !== job.inputKey && !onlyOwnPaintChanged(job.recipe, layer);
          if (!wantsDetection(layer) || inputChanged) {
            settleWithoutDetection(layerId);
          }
          continue;
        }
        // Only a status left behind by a failed or interrupted run is stale here.
        const { status } = statuses.get(layerId);
        if ((status === 'error' || status === 'queued') && !wantsDetection(layer)) {
          interrupted.delete(layerId);
          statuses.set(layerId, IDLE_CONTROL_DETECTION_STATUS);
        }
      }
    },
    resume: () => {
      if (disposed || !suspended) {
        return;
      }
      suspended = false;
      const layerIds = [...interrupted];
      interrupted.clear();
      for (const layerId of layerIds) {
        const layer = readLayer(layerId);
        if (layer && wantsDetection(layer)) {
          schedule(layerId, 0, layer);
        } else {
          statuses.set(layerId, IDLE_CONTROL_DETECTION_STATUS);
        }
      }
    },
    retry: (layerId) => {
      const layer = disposed ? null : readLayer(layerId);
      if (!layer) {
        return 'missing';
      }
      if (!wantsDetection(layer)) {
        settleWithoutDetection(layerId);
        return 'not-needed';
      }
      schedule(layerId, 0, layer);
      return 'scheduled';
    },
    subscribeStatus: statuses.subscribe,
    suspend: () => {
      if (disposed || suspended) {
        return;
      }
      suspended = true;
      for (const layerId of jobs.keys()) {
        interrupted.add(layerId);
        cancelJob(layerId);
        statuses.set(layerId, QUEUED);
      }
    },
  };
};
