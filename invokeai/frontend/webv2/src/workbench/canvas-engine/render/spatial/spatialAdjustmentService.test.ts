import type {
  SpatialAdjustmentExecutor,
  SpatialExecutorJob,
  SpatialExecutorResult,
  SpatialRunPriority,
  SpatialRunResult,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import type { SpatialAdjustmentStep } from '@workbench/canvas-engine/render/spatial/spatialSteps';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';
import {
  createSpatialAdjustmentService,
  SPATIAL_WORKER_IDLE_MS,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { createInlineSpatialExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

const WIDTH = 4;
const HEIGHT = 2;
const PIXEL_BYTES = WIDTH * HEIGHT * 4;
const STEPS: readonly SpatialAdjustmentStep[] = [{ amount: 50, kind: 'clarity', size: 8 }];

/** Pixels whose first byte names the job, so the executor's view shows which job it got. */
const tagged = (tag: number): Uint8ClampedArray => {
  const pixels = new Uint8ClampedArray(PIXEL_BYTES);
  pixels[0] = tag;
  return pixels;
};

/** An executor whose jobs settle only when the test says so. */
const createManualExecutors = () => {
  const executors: { disposed: boolean; started: number[] }[] = [];
  const pending: { job: SpatialExecutorJob; settle: (status: SpatialExecutorResult['status'] | 'crash') => void }[] =
    [];
  const create = (): SpatialAdjustmentExecutor => {
    const state = { disposed: false, started: [] as number[] };
    executors.push(state);
    const own = new Set<(error: Error) => void>();
    return {
      dispose: () => {
        state.disposed = true;
        for (const reject of own) {
          reject(new Error('terminated'));
        }
      },
      run: (job) =>
        new Promise((resolve, reject) => {
          state.started.push(new Uint8ClampedArray(job.buffer)[0]);
          own.add(reject);
          pending.push({
            job,
            settle: (status) => {
              own.delete(reject);
              if (status === 'crash') {
                reject(new Error('worker crashed'));
              } else {
                const pixels = new Uint8ClampedArray(job.buffer);
                pixels[1] = 255;
                resolve({ buffer: job.buffer, status });
              }
            },
          });
        }),
    };
  };
  /** Settles the oldest unsettled job, then lets the service start the next one. */
  const settleNext = async (status: SpatialExecutorResult['status'] | 'crash' = 'ok') => {
    const next = pending.shift();
    if (!next) {
      throw new Error('no job is running');
    }
    next.settle(status);
    await flush();
  };
  const started = () => executors.flatMap((executor) => executor.started);
  return { create, executors, settleNext, started };
};

const flush = async () => {
  for (let i = 0; i < 5; i++) {
    await Promise.resolve();
  }
};

const setup = (budgetBytes = 1 << 20) => {
  const budget = new RasterMemoryBudgetController({ budgetBytes });
  const manual = createManualExecutors();
  const service = createSpatialAdjustmentService({
    createExecutor: manual.create,
    reserve: (bytes) => budget.reserveOperation(bytes, { purpose: 'layer-operation' }),
  });
  const reserved = () => budget.snapshot().reservedBytes;
  const run = (tag: number, options: { slotKey?: string; priority?: SpatialRunPriority; signal?: AbortSignal } = {}) =>
    service.run({
      height: HEIGHT,
      pixels: tagged(tag),
      priority: options.priority ?? 'display',
      signal: options.signal,
      slotKey: options.slotKey ?? `slot-${tag}`,
      steps: STEPS,
      width: WIDTH,
    });
  return { ...manual, reserved, run, service };
};

/** Records results as they settle so tests can assert resolution without awaiting the job. */
const track = (promise: Promise<SpatialRunResult>) => {
  const state: { result: SpatialRunResult | null } = { result: null };
  void promise.then((result) => {
    state.result = result;
  });
  return state;
};

afterEach(() => {
  vi.useRealTimers();
});

describe('createSpatialAdjustmentService', () => {
  it('returns the kernels’ output for the transferred pixels and releases the reservation', async () => {
    const budget = new RasterMemoryBudgetController({ budgetBytes: 1 << 20 });
    const service = createSpatialAdjustmentService({
      createExecutor: createInlineSpatialExecutor,
      reserve: (bytes) => budget.reserveOperation(bytes, { purpose: 'layer-operation' }),
    });
    const width = 24;
    const height = 16;
    const input = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < input.length; i++) {
      input[i] = (i * 37 + (i >> 5) * 11) % 256;
    }
    const steps: SpatialAdjustmentStep[] = [
      { amount: 60, kind: 'clarity', size: 6 },
      { amount: -40, kind: 'dehaze' },
    ];
    const expected = input.slice();
    evaluateSpatialSteps(expected, width, height, steps);

    const pending = service.run({ height, pixels: input.slice(), priority: 'now', slotKey: 'a', steps, width });
    expect(budget.snapshot().reservedBytes).toBe(width * height * 4);
    const result = await pending;

    expect(result.status).toBe('ok');
    expect(result.status === 'ok' && result.pixels).toEqual(expected);
    expect(expected).not.toEqual(input);
    expect(budget.snapshot().reservedBytes).toBe(0);
  });

  it('runs one job at a time, `now` before `display`, FIFO within a priority', async () => {
    const { run, settleNext, started } = setup();
    const results = [run(1), run(2), run(3, { priority: 'now' }), run(4), run(5, { priority: 'now' })];
    await flush();
    expect(started()).toEqual([1]);

    for (let i = 0; i < 5; i++) {
      await settleNext();
    }

    expect(started()).toEqual([1, 3, 5, 2, 4]);
    for (const result of await Promise.all(results)) {
      expect(result.status).toBe('ok');
      expect(result.status === 'ok' && result.pixels[1]).toBe(255);
    }
  });

  it('replaces a queued job for the same slot in its queue position, but never the running one', async () => {
    const { reserved, run, settleNext, started } = setup();
    const running = run(1, { slotKey: 'layer' });
    const replaced = track(run(2, { slotKey: 'layer' }));
    const other = run(3, { slotKey: 'other' });
    const replacement = run(4, { slotKey: 'layer' });
    await flush();

    expect(replaced.result).toEqual({ status: 'superseded' });
    expect(reserved()).toBe(3 * PIXEL_BYTES);

    for (let i = 0; i < 3; i++) {
      await settleNext();
    }

    expect(started()).toEqual([1, 4, 3]);
    expect((await running).status).toBe('ok');
    expect((await other).status).toBe('ok');
    expect((await replacement).status).toBe('ok');
    expect(reserved()).toBe(0);
  });

  it('moves a replaced slot to the replacement’s priority', async () => {
    const { run, settleNext, started } = setup();
    void run(1);
    void run(2, { slotKey: 'layer' });
    void run(3);
    void run(4, { priority: 'now', slotKey: 'layer' });
    for (let i = 0; i < 3; i++) {
      await settleNext();
    }

    expect(started()).toEqual([1, 4, 3]);
  });

  it('resolves an already-aborted request without reserving or running', async () => {
    const { reserved, run, started } = setup();
    const controller = new AbortController();
    controller.abort();

    expect(await run(1, { signal: controller.signal })).toEqual({ status: 'aborted' });
    await flush();
    expect(started()).toEqual([]);
    expect(reserved()).toBe(0);
  });

  it('drops a queued job when its signal aborts', async () => {
    const { reserved, run, settleNext, started } = setup();
    void run(1);
    const controller = new AbortController();
    const queued = run(2, { signal: controller.signal });
    await flush();

    controller.abort();
    expect(await queued).toEqual({ status: 'aborted' });
    expect(reserved()).toBe(PIXEL_BYTES);

    await settleNext();
    expect(started()).toEqual([1]);
    expect(reserved()).toBe(0);
  });

  it('answers an aborted running job at once but holds its reservation and slot until the worker returns', async () => {
    const { reserved, run, settleNext, started } = setup();
    const controller = new AbortController();
    const running = run(1, { signal: controller.signal });
    const next = run(2);
    await flush();

    controller.abort();
    expect(await running).toEqual({ status: 'aborted' });
    expect(reserved()).toBe(2 * PIXEL_BYTES);
    expect(started()).toEqual([1]);

    await settleNext();
    expect(reserved()).toBe(PIXEL_BYTES);
    expect(started()).toEqual([1, 2]);
    await settleNext();
    expect((await next).status).toBe('ok');
    expect(reserved()).toBe(0);
  });

  it('refuses a job whose pixels do not fit the raster budget and keeps serving others', async () => {
    const { reserved, run, settleNext, started } = setup(PIXEL_BYTES * 2);
    const first = run(1);
    const second = run(2);

    expect(await run(3)).toEqual({ status: 'over-budget' });

    await settleNext();
    await settleNext();
    expect((await first).status).toBe('ok');
    expect((await second).status).toBe('ok');
    expect(started()).toEqual([1, 2]);
    expect(reserved()).toBe(0);
  });

  it('maps executor outcomes and replaces a crashed executor for the rest of the queue', async () => {
    const { executors, reserved, run, settleNext, started } = setup();
    const tooLarge = run(1);
    const failed = run(2);
    const crashed = run(3);
    const after = run(4);

    await settleNext('too-large');
    await settleNext('error');
    expect(executors).toHaveLength(1);
    await settleNext('crash');
    await settleNext();

    expect(await tooLarge).toEqual({ status: 'too-large' });
    expect(await failed).toEqual({ status: 'error' });
    expect(await crashed).toEqual({ status: 'error' });
    expect((await after).status).toBe('ok');
    expect(executors.map((executor) => executor.disposed)).toEqual([true, false]);
    expect(started()).toEqual([1, 2, 3, 4]);
    expect(reserved()).toBe(0);
  });

  it('reports an executor that cannot be created as an error and tries again for the next job', async () => {
    const budget = new RasterMemoryBudgetController({ budgetBytes: 1 << 20 });
    const manual = createManualExecutors();
    let attempts = 0;
    const service = createSpatialAdjustmentService({
      createExecutor: () => {
        attempts++;
        if (attempts === 1) {
          throw new Error('workers unavailable');
        }
        return manual.create();
      },
      reserve: (bytes) => budget.reserveOperation(bytes, { purpose: 'layer-operation' }),
    });
    const request = { height: HEIGHT, priority: 'now', slotKey: 'a', steps: STEPS, width: WIDTH } as const;

    expect(await service.run({ ...request, pixels: tagged(1) })).toEqual({ status: 'error' });
    const second = service.run({ ...request, pixels: tagged(2) });
    await flush();
    await manual.settleNext();

    expect((await second).status).toBe('ok');
    expect(budget.snapshot().reservedBytes).toBe(0);
  });

  it('dispose aborts everything, terminates the executor, releases every reservation and refuses later work', async () => {
    const { executors, reserved, run, service, started } = setup();
    const running = run(1);
    const queued = run(2);
    await flush();

    service.dispose();
    expect(await running).toEqual({ status: 'aborted' });
    expect(await queued).toEqual({ status: 'aborted' });
    await flush();
    expect(executors.map((executor) => executor.disposed)).toEqual([true]);
    expect(reserved()).toBe(0);

    expect(await run(3)).toEqual({ status: 'aborted' });
    await flush();
    expect(started()).toEqual([1]);
    expect(executors).toHaveLength(1);
    expect(reserved()).toBe(0);
  });

  it('terminates the executor after an idle period and creates a fresh one on demand', async () => {
    vi.useFakeTimers();
    const { executors, run, settleNext } = setup();
    void run(1);
    await settleNext();

    await vi.advanceTimersByTimeAsync(SPATIAL_WORKER_IDLE_MS - 1);
    void run(2);
    await settleNext();
    await vi.advanceTimersByTimeAsync(SPATIAL_WORKER_IDLE_MS - 1);
    expect(executors.map((executor) => executor.disposed)).toEqual([false]);

    await vi.advanceTimersByTimeAsync(1);
    expect(executors.map((executor) => executor.disposed)).toEqual([true]);

    const revived = run(3);
    await settleNext();
    expect((await revived).status).toBe('ok');
    expect(executors.map((executor) => executor.disposed)).toEqual([true, false]);
  });

  it('rejects pixels that do not span exactly one whole buffer', () => {
    const { service } = setup();
    const request = { height: HEIGHT, priority: 'now', slotKey: 'a', steps: STEPS, width: WIDTH } as const;

    expect(() =>
      service.run({ ...request, pixels: new Uint8ClampedArray(PIXEL_BYTES * 2).subarray(0, PIXEL_BYTES) })
    ).toThrow(RangeError);
    expect(() => service.run({ ...request, pixels: new Uint8ClampedArray(PIXEL_BYTES + 4) })).toThrow(RangeError);
  });
});
