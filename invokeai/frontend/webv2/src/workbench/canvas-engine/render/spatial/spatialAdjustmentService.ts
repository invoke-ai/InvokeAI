// SPDX-License-Identifier: Apache-2.0
/**
 * Runs spatial adjustment evaluations off the main thread, one at a time, with raster-memory accounting. This is
 * the only spatial module production code imports: the kernels load inside the worker, never in a page chunk.
 */

import type { RasterMemoryReservationResult } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { SpatialAdjustmentStep } from '@workbench/canvas-engine/render/spatial/spatialSteps';

/** One evaluation handed to an executor; `buffer` holds `width * height` unpremultiplied RGBA8 pixels. */
export interface SpatialExecutorJob {
  readonly width: number;
  readonly height: number;
  readonly buffer: ArrayBuffer;
  readonly steps: readonly SpatialAdjustmentStep[];
}

/** `error` reports a job that failed to send or threw while evaluating; the executor itself stays usable. */
export interface SpatialExecutorResult {
  readonly status: 'ok' | 'too-large' | 'error';
  readonly buffer: ArrayBuffer;
  readonly message?: string;
}

/**
 * Evaluates one job at a time. `run` rejects only when the executor broke (worker crash, undeliverable reply); the
 * service then disposes it and continues on a fresh one. `dispose` rejects any job still pending.
 */
export interface SpatialAdjustmentExecutor {
  run(job: SpatialExecutorJob): Promise<SpatialExecutorResult>;
  dispose(): void;
}

export type SpatialWorkerRequest = SpatialExecutorJob & { readonly id: number };
export type SpatialWorkerResponse = SpatialExecutorResult & { readonly id: number };

export type SpatialRunPriority = 'now' | 'display';

export interface SpatialRunRequest {
  /** Identifies the surface this result is for; a newer request for the same slot replaces a queued one. */
  readonly slotKey: string;
  readonly width: number;
  readonly height: number;
  /**
   * Unpremultiplied RGBA8, exactly `width * height * 4` bytes spanning a whole `ArrayBuffer` (`run` throws a
   * `RangeError` otherwise). The buffer is transferred to the worker: callers must not use this array, or any other
   * view of its buffer, after calling `run`, whatever the result.
   */
  readonly pixels: Uint8ClampedArray;
  readonly steps: readonly SpatialAdjustmentStep[];
  /** `now` jobs run before every queued `display` job. */
  readonly priority: SpatialRunPriority;
  readonly signal?: AbortSignal;
}

export type SpatialRunResult =
  | { readonly status: 'ok'; readonly pixels: Uint8ClampedArray }
  | { readonly status: 'superseded' | 'aborted' | 'too-large' | 'over-budget' | 'error' };

export interface SpatialAdjustmentService {
  run(request: SpatialRunRequest): Promise<SpatialRunResult>;
  /** Resolves queued jobs `aborted` and discards the running job's result. The service stays usable. */
  cancelAll(): void;
  /** Cancels everything and terminates the worker; later `run` calls resolve `aborted`. */
  dispose(): void;
}

export interface SpatialAdjustmentServiceOptions {
  /** Reserves raster memory for one job's pixel buffer; the caller binds the reservation purpose. */
  readonly reserve: (bytes: number) => RasterMemoryReservationResult;
  readonly createExecutor?: () => SpatialAdjustmentExecutor;
}

/** An idle worker holds its module and heap; recreating one costs a module load, so keep it through short pauses. */
export const SPATIAL_WORKER_IDLE_MS = 30_000;

/** The production executor: one module worker running the kernels. */
export const createWorkerExecutor = (): SpatialAdjustmentExecutor => {
  const worker = new Worker(new URL('./spatialAdjustment.worker.ts', import.meta.url), { type: 'module' });
  const pending = new Map<
    number,
    { resolve: (result: SpatialExecutorResult) => void; reject: (error: Error) => void }
  >();
  let nextId = 0;

  const rejectPending = (error: Error) => {
    const settlers = [...pending.values()];
    pending.clear();
    for (const { reject } of settlers) {
      reject(error);
    }
  };

  worker.addEventListener('message', (event: MessageEvent<SpatialWorkerResponse>) => {
    const { id, ...result } = event.data;
    const settler = pending.get(id);
    pending.delete(id);
    settler?.resolve(result);
  });
  worker.addEventListener('error', (event) => {
    rejectPending(new Error(event.message || 'Spatial adjustment worker failed.'));
  });
  worker.addEventListener('messageerror', () => {
    rejectPending(new Error('Spatial adjustment worker reply could not be deserialized.'));
  });

  return {
    dispose: () => {
      worker.terminate();
      rejectPending(new Error('Spatial adjustment worker was terminated.'));
    },
    run: (job) =>
      new Promise((resolve, reject) => {
        const id = nextId++;
        const request: SpatialWorkerRequest = { ...job, id };
        pending.set(id, { reject, resolve });
        try {
          worker.postMessage(request, [job.buffer]);
        } catch (error) {
          // The message never left (e.g. a detached buffer cannot be transferred); the worker is still healthy.
          pending.delete(id);
          resolve({
            buffer: job.buffer,
            message: error instanceof Error ? error.message : String(error),
            status: 'error',
          });
        }
      }),
  };
};

interface Job {
  readonly request: SpatialRunRequest;
  readonly buffer: ArrayBuffer;
  readonly release: () => void;
  readonly resolve: (result: SpatialRunResult) => void;
  /** Set once the caller has its result; a running job's late executor result is then discarded. */
  answered: boolean;
}

const ABORTED: SpatialRunResult = { status: 'aborted' };
const SUPERSEDED: SpatialRunResult = { status: 'superseded' };
const OVER_BUDGET: SpatialRunResult = { status: 'over-budget' };
const TOO_LARGE: SpatialRunResult = { status: 'too-large' };
const ERROR: SpatialRunResult = { status: 'error' };

/**
 * One job runs at a time. Queued jobs run `now` before `display`, FIFO within a priority. Each accepted job holds a
 * reservation for its pixel buffer until the buffer comes back from the executor (or the executor is disposed), so an
 * aborted running job keeps its reservation, and the single run slot, until the worker finishes.
 */
export const createSpatialAdjustmentService = ({
  createExecutor = createWorkerExecutor,
  reserve,
}: SpatialAdjustmentServiceOptions): SpatialAdjustmentService => {
  const queues: Record<SpatialRunPriority, Job[]> = { display: [], now: [] };
  let running: Job | null = null;
  let executor: SpatialAdjustmentExecutor | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;

  const answer = (job: Job, result: SpatialRunResult) => {
    if (!job.answered) {
      job.answered = true;
      job.resolve(result);
    }
  };

  /** Ends a job that never reached the executor. */
  const finishQueued = (job: Job, result: SpatialRunResult) => {
    answer(job, result);
    job.release();
  };

  const removeQueued = (job: Job): boolean => {
    const queue = queues[job.request.priority];
    const index = queue.indexOf(job);
    if (index < 0) {
      return false;
    }
    queue.splice(index, 1);
    return true;
  };

  const dropExecutor = () => {
    const current = executor;
    executor = null;
    current?.dispose();
  };

  const scheduleIdleTermination = () => {
    idleTimer = setTimeout(() => {
      idleTimer = null;
      dropExecutor();
    }, SPATIAL_WORKER_IDLE_MS);
  };

  const execute = async (job: Job) => {
    const { height, steps, width } = job.request;
    try {
      executor ??= createExecutor();
      const result = await executor.run({ buffer: job.buffer, height, steps, width });
      answer(
        job,
        result.status === 'ok'
          ? { pixels: new Uint8ClampedArray(result.buffer), status: 'ok' }
          : result.status === 'too-large'
            ? TOO_LARGE
            : ERROR
      );
    } catch {
      answer(job, ERROR);
      if (!disposed) {
        dropExecutor();
      }
    } finally {
      job.release();
      running = null;
      pump();
    }
  };

  const pump = () => {
    if (disposed || running) {
      return;
    }
    const next = queues.now.shift() ?? queues.display.shift();
    if (!next) {
      if (executor && !idleTimer) {
        scheduleIdleTermination();
      }
      return;
    }
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    running = next;
    void execute(next);
  };

  const cancelAll = () => {
    const queued = [...queues.now, ...queues.display];
    queues.now.length = 0;
    queues.display.length = 0;
    for (const job of queued) {
      finishQueued(job, ABORTED);
    }
    if (running) {
      answer(running, ABORTED);
    }
  };

  const run = (request: SpatialRunRequest): Promise<SpatialRunResult> => {
    const { height, pixels, signal, width } = request;
    const { buffer } = pixels;
    // Transfer moves the whole buffer: a partial view would detach memory its caller still uses.
    if (
      !(buffer instanceof ArrayBuffer) ||
      pixels.length !== width * height * 4 ||
      pixels.byteLength !== buffer.byteLength
    ) {
      throw new RangeError('Spatial adjustment pixels must be width * height RGBA8 bytes owning an ArrayBuffer.');
    }
    if (disposed || signal?.aborted) {
      return Promise.resolve(ABORTED);
    }
    // The raster budget accounts canvas surfaces resident on the main thread. Worker scratch is transient and bounded
    // separately: one job at a time, at most MAX_SPATIAL_PIXELS * SPATIAL_SCRATCH_BYTES_PER_PIXEL (about 400 MB at the
    // cap), freed when the job ends or the worker idles out.
    const reservation = reserve(width * height * 4);
    if (reservation.status !== 'ok') {
      return Promise.resolve(OVER_BUDGET);
    }

    return new Promise<SpatialRunResult>((resolve) => {
      const onAbort = () => {
        if (removeQueued(job)) {
          finishQueued(job, ABORTED);
        } else {
          answer(job, ABORTED);
        }
      };
      const job: Job = {
        answered: false,
        buffer,
        release: () => {
          signal?.removeEventListener('abort', onAbort);
          reservation.lease.release();
        },
        request,
        resolve,
      };
      signal?.addEventListener('abort', onAbort, { once: true });

      const queue = queues[request.priority];
      const replaced = [...queues.now, ...queues.display].find((queued) => queued.request.slotKey === request.slotKey);
      if (replaced && replaced.request.priority === request.priority) {
        // Take over the replaced job's place so a slot that keeps changing is not pushed behind later slots.
        queue.splice(queue.indexOf(replaced), 1, job);
      } else {
        if (replaced) {
          removeQueued(replaced);
        }
        queue.push(job);
      }
      if (replaced) {
        finishQueued(replaced, SUPERSEDED);
      }
      pump();
    });
  };

  return {
    cancelAll,
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      cancelAll();
      if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
      dropExecutor();
    },
    run,
  };
};
