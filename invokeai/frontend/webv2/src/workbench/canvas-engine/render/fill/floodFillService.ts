// SPDX-License-Identifier: Apache-2.0
/**
 * Computes paint bucket regions off the main thread. A press sends its sample once (`begin`), the worker keeps the
 * fill level map, and every tolerance or option change (`region`) is answered from that map until `end`. This is the
 * only fill module production code imports: the kernels load inside the worker, never in a page chunk.
 */

import type { FillSample } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import type { FillRegion, FillRegionParams } from '@workbench/canvas-engine/render/fill/regionMorphology';
import type { Vec2 } from '@workbench/canvas-engine/types';

export interface FloodFillBeginJob {
  readonly kind: 'begin';
  readonly pressId: number;
  readonly width: number;
  readonly height: number;
  /** `width * height` straight RGBA8 pixels; transferred. */
  readonly buffer: ArrayBuffer;
  readonly seedX: number;
  readonly seedY: number;
  readonly contiguous: boolean;
  readonly params: FillRegionParams;
}

export interface FloodFillRegionJob {
  readonly kind: 'region';
  readonly pressId: number;
  readonly params: FillRegionParams;
}

export type FloodFillJob = FloodFillBeginJob | FloodFillRegionJob;

/** `error` reports a job that failed (bad input, unknown press); the executor itself stays usable. */
export type FloodFillExecutorResult =
  | { readonly status: 'ok'; readonly region: FillRegion }
  | { readonly status: 'error'; readonly message: string };

/**
 * Runs one job at a time and keeps each begun press's level map until `release`. Messages are ordered: a release
 * takes effect after every job sent before it. `run` rejects only when the executor broke (worker crash,
 * undeliverable reply); every press it held is then lost. `dispose` rejects any job still pending.
 */
export interface FloodFillExecutor {
  run(job: FloodFillJob): Promise<FloodFillExecutorResult>;
  release(pressId: number): void;
  dispose(): void;
}

export type FloodFillWorkerRequest =
  | (FloodFillJob & { readonly id: number })
  | { readonly kind: 'release'; readonly pressId: number };
export type FloodFillWorkerResponse = FloodFillExecutorResult & { readonly id: number };

export interface FloodFillBeginOptions extends FillRegionParams {
  readonly contiguous: boolean;
}

/**
 * Every answer carries the `sequence` its request got, increasing across the service, so a caller can tell the
 * newest answer. `superseded`: a newer `region` for the same press replaced this one before it ran. `ended`: the
 * press ended (or the service was disposed) before the answer arrived; its region is dropped. `error`: the job
 * failed, or the press's level map was lost with a crashed worker; later regions for that press answer `error`.
 */
export type FloodFillAnswer =
  | { readonly status: 'ok'; readonly sequence: number; readonly region: FillRegion }
  | { readonly status: 'superseded' | 'ended' | 'error'; readonly sequence: number };

export interface FloodFillService {
  /**
   * Builds the press's level map and answers its first region. `sample.data` must span a whole `ArrayBuffer`
   * (`begin` throws a `RangeError` otherwise) and is transferred: callers must not use it, or any other view of its
   * buffer, after calling `begin`. `sampleLease`, the raster reservation covering the sample, is released once the
   * executor has taken the buffer, or when the job ends without running. Throws when `pressId` is already open.
   */
  begin(
    pressId: number,
    sample: FillSample,
    seed: Vec2,
    options: FloodFillBeginOptions,
    sampleLease?: { release(): void }
  ): Promise<FloodFillAnswer>;
  /** Answers a region from the press's map; replaces this press's queued, not yet running, region request. */
  region(pressId: number, params: FillRegionParams): Promise<FloodFillAnswer>;
  /** Frees the press's map; its queued and running requests answer `ended`. Unknown presses are ignored. */
  end(pressId: number): void;
  /** Ends every press and terminates the worker; later requests answer `ended`. */
  dispose(): void;
}

export interface FloodFillServiceOptions {
  readonly createExecutor?: () => FloodFillExecutor;
}

/** An idle worker holds its module; recreating one costs a module load, so keep it through short pauses. */
export const FLOOD_FILL_WORKER_IDLE_MS = 30_000;

/** The production executor: one module worker running the kernels. */
export const createFloodFillWorkerExecutor = (): FloodFillExecutor => {
  const worker = new Worker(new URL('./floodFill.worker.ts', import.meta.url), { type: 'module' });
  const pending = new Map<
    number,
    { resolve: (result: FloodFillExecutorResult) => void; reject: (error: Error) => void }
  >();
  let nextId = 0;
  /** Set by the first worker failure: the worker and the maps it held are gone, so every later job fails too. */
  let failure: Error | null = null;

  const fail = (error: Error) => {
    failure ??= error;
    const settlers = [...pending.values()];
    pending.clear();
    for (const { reject } of settlers) {
      reject(error);
    }
  };

  worker.addEventListener('message', (event: MessageEvent<FloodFillWorkerResponse>) => {
    const { id, ...result } = event.data;
    const settler = pending.get(id);
    pending.delete(id);
    settler?.resolve(result);
  });
  // An error with no job pending still means a dead worker: the next job must fail, not wait forever.
  worker.addEventListener('error', (event) => {
    fail(new Error(event.message || 'Flood fill worker failed.'));
  });
  worker.addEventListener('messageerror', () => {
    fail(new Error('Flood fill worker reply could not be deserialized.'));
  });

  return {
    dispose: () => {
      worker.terminate();
      fail(new Error('Flood fill worker was terminated.'));
    },
    release: (pressId) => {
      const request: FloodFillWorkerRequest = { kind: 'release', pressId };
      worker.postMessage(request);
    },
    run: (job) =>
      new Promise((resolve, reject) => {
        if (failure) {
          reject(failure);
          return;
        }
        const id = nextId++;
        const request: FloodFillWorkerRequest = { ...job, id };
        pending.set(id, { reject, resolve });
        try {
          worker.postMessage(request, job.kind === 'begin' ? [job.buffer] : []);
        } catch (error) {
          // The message never left (e.g. a detached buffer cannot be transferred); the worker is still healthy.
          pending.delete(id);
          resolve({ message: error instanceof Error ? error.message : String(error), status: 'error' });
        }
      }),
  };
};

interface Press {
  /** The executor that received this press's begin; only it holds the map. */
  holder: FloodFillExecutor | null;
  /** The map is gone (failed begin, crashed worker): later regions answer `error` without running. */
  broken: boolean;
}

interface Job {
  readonly press: Press;
  readonly job: FloodFillJob;
  readonly sequence: number;
  readonly resolve: (answer: FloodFillAnswer) => void;
  /** Set once the caller has its answer; a running job's late result is then discarded. */
  answered: boolean;
  /** A begin's sample reservation, released once the executor has the buffer or the job ends without running. */
  lease: { release(): void } | null;
}

/**
 * One job runs at a time, FIFO. A press's newer region request takes the place of its queued one. The worker stays
 * alive while any press is open (it holds their maps) and terminates after `FLOOD_FILL_WORKER_IDLE_MS` with no
 * press and no work.
 */
export const createFloodFillService = ({
  createExecutor = createFloodFillWorkerExecutor,
}: FloodFillServiceOptions = {}): FloodFillService => {
  const presses = new Map<number, Press>();
  const queue: Job[] = [];
  let running: Job | null = null;
  let executor: FloodFillExecutor | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;
  let nextSequence = 0;

  const releaseLease = (job: Job) => {
    const { lease } = job;
    job.lease = null;
    lease?.release();
  };

  const answer = (job: Job, result: FloodFillAnswer) => {
    releaseLease(job);
    if (!job.answered) {
      job.answered = true;
      job.resolve(result);
    }
  };

  const status = (job: Job, value: 'superseded' | 'ended' | 'error'): FloodFillAnswer => ({
    sequence: job.sequence,
    status: value,
  });

  const dropExecutor = () => {
    const current = executor;
    executor = null;
    current?.dispose();
  };

  const cancelIdleTermination = () => {
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  };

  const execute = async (job: Job) => {
    const { press } = job;
    let current: FloodFillExecutor | null = null;
    try {
      if (job.job.kind === 'region' && (press.broken || press.holder !== executor)) {
        press.broken = true;
        answer(job, status(job, 'error'));
        return;
      }
      current = executor ??= createExecutor();
      if (job.job.kind === 'begin') {
        press.holder = current;
      }
      const running = current.run(job.job);
      // The buffer has been posted (transferred) to the executor; its reservation has served.
      releaseLease(job);
      const result = await running;
      if (result.status === 'ok') {
        answer(job, { region: result.region, sequence: job.sequence, status: 'ok' });
      } else {
        if (job.job.kind === 'begin') {
          press.broken = true;
        }
        answer(job, status(job, 'error'));
      }
    } catch {
      answer(job, status(job, 'error'));
      if (!disposed && current && executor === current) {
        for (const open of presses.values()) {
          if (open.holder === current) {
            open.broken = true;
          }
        }
        dropExecutor();
      }
    } finally {
      running = null;
      pump();
    }
  };

  const pump = () => {
    if (disposed || running) {
      return;
    }
    const next = queue.shift();
    if (!next) {
      if (executor && !idleTimer && presses.size === 0) {
        idleTimer = setTimeout(() => {
          idleTimer = null;
          dropExecutor();
        }, FLOOD_FILL_WORKER_IDLE_MS);
      }
      return;
    }
    cancelIdleTermination();
    running = next;
    void execute(next);
  };

  const enqueue = (
    press: Press,
    job: FloodFillJob,
    lease: { release(): void } | null = null
  ): Promise<FloodFillAnswer> => {
    const sequence = nextSequence++;
    return new Promise<FloodFillAnswer>((resolve) => {
      const entry: Job = { answered: false, job, lease, press, resolve, sequence };
      const replaced =
        job.kind === 'region'
          ? queue.findIndex((queued) => queued.press === press && queued.job.kind === 'region')
          : -1;
      if (replaced >= 0) {
        const previous = queue[replaced];
        queue[replaced] = entry;
        answer(previous, status(previous, 'superseded'));
      } else {
        queue.push(entry);
      }
      pump();
    });
  };

  return {
    begin: (pressId, sample, seed, { contiguous, ...params }, sampleLease) => {
      const { data, height, width } = sample;
      const { buffer } = data;
      // Transfer moves the whole buffer: a partial view would detach memory its caller still uses.
      if (
        !(buffer instanceof ArrayBuffer) ||
        data.length !== width * height * 4 ||
        data.byteLength !== buffer.byteLength
      ) {
        throw new RangeError('Fill sample must be width * height RGBA8 bytes owning an ArrayBuffer.');
      }
      if (disposed) {
        sampleLease?.release();
        return Promise.resolve({ sequence: nextSequence++, status: 'ended' });
      }
      if (presses.has(pressId)) {
        throw new Error(`Fill press ${pressId} is already open.`);
      }
      const press: Press = { broken: false, holder: null };
      presses.set(pressId, press);
      return enqueue(
        press,
        {
          buffer,
          contiguous,
          height,
          kind: 'begin',
          params,
          pressId,
          seedX: seed.x,
          seedY: seed.y,
          width,
        },
        sampleLease ?? null
      );
    },
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      for (const job of queue.splice(0)) {
        answer(job, status(job, 'ended'));
      }
      if (running) {
        answer(running, status(running, 'ended'));
      }
      presses.clear();
      cancelIdleTermination();
      dropExecutor();
    },
    end: (pressId) => {
      const press = presses.get(pressId);
      if (!press) {
        return;
      }
      presses.delete(pressId);
      for (let index = queue.length - 1; index >= 0; index--) {
        const job = queue[index];
        if (job.press === press) {
          queue.splice(index, 1);
          answer(job, status(job, 'ended'));
        }
      }
      if (running?.press === press) {
        answer(running, status(running, 'ended'));
      }
      if (press.holder && press.holder === executor) {
        executor.release(pressId);
      }
      pump();
    },
    region: (pressId, params) => {
      const press = presses.get(pressId);
      if (disposed || !press) {
        return Promise.resolve({ sequence: nextSequence++, status: 'ended' });
      }
      if (press.broken) {
        return Promise.resolve({ sequence: nextSequence++, status: 'error' });
      }
      return enqueue(press, { kind: 'region', params, pressId });
    },
  };
};
