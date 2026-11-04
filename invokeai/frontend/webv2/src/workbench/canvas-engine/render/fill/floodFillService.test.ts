import type {
  FloodFillAnswer,
  FloodFillExecutor,
  FloodFillExecutorResult,
  FloodFillJob,
  FloodFillWorkerRequest,
} from '@workbench/canvas-engine/render/fill/floodFillService';
import type { FillRegionParams } from '@workbench/canvas-engine/render/fill/regionMorphology';

import { createLineArtSample, LINE_ART } from '@workbench/canvas-engine/render/fill/fillImages.testing';
import { buildFillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import {
  createFloodFillService,
  createFloodFillWorkerExecutor,
  FLOOD_FILL_WORKER_IDLE_MS,
} from '@workbench/canvas-engine/render/fill/floodFillService';
import { createInlineFloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService.testing';
import { computeFillRegion } from '@workbench/canvas-engine/render/fill/regionMorphology';
import { afterEach, describe, expect, it, vi } from 'vitest';

const PARAMS: FillRegionParams = { antiAlias: true, closeGaps: 0, expand: 0, tolerance: 20 };
const SEED = LINE_ART.narrow.seed;
const tolerance = (value: number): FillRegionParams => ({ ...PARAMS, tolerance: value });

const OK: FloodFillExecutorResult = {
  region: { bounds: { height: 1, width: 1, x: 0, y: 0 }, coverage: Uint8Array.of(255), pixelCount: 1 },
  status: 'ok',
};

/** An executor whose jobs settle only when the test says so. */
const createManualExecutors = () => {
  const executors: { disposed: boolean; released: number[] }[] = [];
  const started: FloodFillJob[] = [];
  const pending: ((outcome: FloodFillExecutorResult | 'crash') => void)[] = [];
  const create = (): FloodFillExecutor => {
    const state = { disposed: false, released: [] as number[] };
    executors.push(state);
    const own = new Set<(error: Error) => void>();
    return {
      dispose: () => {
        state.disposed = true;
        for (const reject of own) {
          reject(new Error('terminated'));
        }
      },
      release: (pressId) => state.released.push(pressId),
      run: (job) =>
        new Promise((resolve, reject) => {
          started.push(job);
          own.add(reject);
          pending.push((outcome) => {
            own.delete(reject);
            if (outcome === 'crash') {
              reject(new Error('worker crashed'));
            } else {
              resolve(outcome);
            }
          });
        }),
    };
  };
  const settleNext = async (outcome: FloodFillExecutorResult | 'crash' = OK) => {
    const settle = pending.shift();
    if (!settle) {
      throw new Error('no job is running');
    }
    settle(outcome);
    await flush();
  };
  /** Each started job as `begin` or its region tolerance. */
  const startedLabels = () =>
    started.map((job) => (job.kind === 'begin' ? `begin ${job.pressId}` : job.params.tolerance));
  return { create, executors, settleNext, startedLabels };
};

const flush = async () => {
  for (let i = 0; i < 5; i++) {
    await Promise.resolve();
  }
};

const setup = () => {
  const manual = createManualExecutors();
  const service = createFloodFillService({ createExecutor: manual.create });
  const begin = (pressId: number, params = PARAMS) =>
    service.begin(pressId, createLineArtSample(), SEED, { contiguous: true, ...params });
  return { ...manual, begin, service };
};

/** Records answers as they settle so tests can assert resolution without awaiting the job. */
const track = (promise: Promise<FloodFillAnswer>) => {
  const state: { answer: FloodFillAnswer | null } = { answer: null };
  void promise.then((answer) => {
    state.answer = answer;
  });
  return state;
};

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('createFloodFillService', () => {
  it('answers begin and later regions from the press’s level map', async () => {
    const service = createFloodFillService({ createExecutor: createInlineFloodFillExecutor });
    const sample = createLineArtSample();
    const map = buildFillLevelMap(createLineArtSample(), SEED.x, SEED.y, true);

    const first = await service.begin(1, sample, SEED, { contiguous: true, ...PARAMS });
    const closed = { ...PARAMS, closeGaps: 3 };
    const second = await service.region(1, closed);

    expect(first).toEqual({ region: computeFillRegion(map, PARAMS), sequence: 0, status: 'ok' });
    expect(second).toEqual({ region: computeFillRegion(map, closed), sequence: 1, status: 'ok' });
    service.end(1);
    expect(await service.region(1, PARAMS)).toEqual({ sequence: 2, status: 'ended' });
    service.dispose();
  });

  it('replaces a press’s queued region request with the newer one, never the running one', async () => {
    const { begin, service, settleNext, startedLabels } = setup();
    const first = begin(1);
    const running = service.region(1, tolerance(30));
    await settleNext();
    const replaced = track(service.region(1, tolerance(40)));
    const newest = service.region(1, tolerance(50));
    await flush();

    expect(replaced.answer).toEqual({ sequence: 2, status: 'superseded' });
    await settleNext();
    await settleNext();

    expect(startedLabels()).toEqual(['begin 1', 30, 50]);
    expect((await first).status).toBe('ok');
    expect(await running).toMatchObject({ sequence: 1, status: 'ok' });
    expect(await newest).toMatchObject({ sequence: 3, status: 'ok' });
  });

  it('does not let one press’s request replace another’s', async () => {
    const { begin, service, settleNext, startedLabels } = setup();
    void begin(1);
    void begin(2);
    const one = service.region(1, tolerance(30));
    const two = service.region(2, tolerance(40));
    for (let i = 0; i < 4; i++) {
      await settleNext();
    }

    expect(startedLabels()).toEqual(['begin 1', 'begin 2', 30, 40]);
    expect((await one).status).toBe('ok');
    expect((await two).status).toBe('ok');
  });

  it('drops answers for an ended press and releases its map after the running job', async () => {
    const { begin, executors, service, settleNext, startedLabels } = setup();
    const running = begin(1);
    const queued = service.region(1, tolerance(60));
    await flush();

    service.end(1);
    expect(await running).toEqual({ sequence: 0, status: 'ended' });
    expect(await queued).toEqual({ sequence: 1, status: 'ended' });
    expect(executors[0].released).toEqual([1]);

    await settleNext();
    expect(startedLabels()).toEqual(['begin 1']);
    expect(await service.region(1, PARAMS)).toEqual({ sequence: 2, status: 'ended' });
  });

  it('never asks the worker to release a press whose begin it never received', async () => {
    const { begin, executors, service, settleNext, startedLabels } = setup();
    void begin(1);
    const waiting = begin(2);
    await flush();
    service.end(2);
    await settleNext();

    expect(await waiting).toEqual({ sequence: 1, status: 'ended' });
    expect(startedLabels()).toEqual(['begin 1']);
    expect(executors[0].released).toEqual([]);
  });

  it('answers errors for a failed begin without running its later regions', async () => {
    const { begin, service, settleNext, startedLabels } = setup();
    const first = begin(1);
    await settleNext({ message: 'seed outside', status: 'error' });

    expect(await first).toEqual({ sequence: 0, status: 'error' });
    expect(await service.region(1, PARAMS)).toEqual({ sequence: 1, status: 'error' });
    expect(startedLabels()).toEqual(['begin 1']);
  });

  it('loses every press on a crashed worker and serves new presses on a fresh one', async () => {
    const { begin, executors, service, settleNext, startedLabels } = setup();
    void begin(1);
    await settleNext();
    void begin(2);
    const crashed = service.region(2, tolerance(30));
    await settleNext();
    await settleNext('crash');

    expect(await crashed).toEqual({ sequence: 2, status: 'error' });
    expect(await service.region(1, PARAMS)).toEqual({ sequence: 3, status: 'error' });
    expect(executors.map((executor) => executor.disposed)).toEqual([true]);

    const fresh = begin(3);
    await settleNext();
    expect((await fresh).status).toBe('ok');
    expect(executors.map((executor) => executor.disposed)).toEqual([true, false]);
    expect(startedLabels()).toEqual(['begin 1', 'begin 2', 30, 'begin 3']);
  });

  it('takes a worker error with no job pending as a dead worker, failing its next job and starting a fresh one', async () => {
    /** Answers every job until it errors; a dead worker answers nothing, as a killed one would. */
    class FakeWorker {
      static created: FakeWorker[] = [];
      readonly listeners = new Map<string, ((event: unknown) => void)[]>();
      dead = false;
      terminated = false;
      constructor() {
        FakeWorker.created.push(this);
      }
      addEventListener(type: string, listener: (event: unknown) => void) {
        this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
      }
      emit(type: string, event: unknown) {
        this.listeners.get(type)?.forEach((listener) => listener(event));
      }
      postMessage(request: FloodFillWorkerRequest) {
        if (request.kind !== 'release' && !this.dead) {
          queueMicrotask(() => this.emit('message', { data: { ...OK, id: request.id } }));
        }
      }
      terminate() {
        this.terminated = true;
      }
    }
    vi.stubGlobal('Worker', FakeWorker);
    const service = createFloodFillService({ createExecutor: createFloodFillWorkerExecutor });
    const options = { contiguous: true, ...PARAMS };
    expect((await service.begin(1, createLineArtSample(), SEED, options)).status).toBe('ok');

    const [first] = FakeWorker.created;
    first!.dead = true;
    first!.emit('error', { message: 'killed' });
    expect((await service.region(1, PARAMS)).status).toBe('error');
    expect(first!.terminated).toBe(true);

    expect((await service.begin(2, createLineArtSample(), SEED, options)).status).toBe('ok');
    expect(FakeWorker.created).toHaveLength(2);
    service.dispose();
  });

  it('holds a begin’s sample reservation until the executor has the buffer, or until the job ends unrun', async () => {
    const { executors, service, settleNext } = setup();
    const lease = (released: string[], name: string) => ({ release: () => released.push(name) });
    const released: string[] = [];
    void service.begin(1, createLineArtSample(), SEED, { contiguous: true, ...PARAMS }, lease(released, 'running'));
    void service.begin(2, createLineArtSample(), SEED, { contiguous: true, ...PARAMS }, lease(released, 'queued'));
    // The first begin went straight to the executor; the second waits behind it with its reservation held.
    expect(released).toEqual(['running']);
    service.end(2);
    expect(released).toEqual(['running', 'queued']);
    await settleNext();
    expect(executors).toHaveLength(1);
  });

  it('keeps the worker while a press is open and terminates it after an idle period without one', async () => {
    vi.useFakeTimers();
    const { begin, executors, service, settleNext } = setup();
    void begin(1);
    await settleNext();

    await vi.advanceTimersByTimeAsync(FLOOD_FILL_WORKER_IDLE_MS * 3);
    expect(executors.map((executor) => executor.disposed)).toEqual([false]);

    service.end(1);
    await vi.advanceTimersByTimeAsync(FLOOD_FILL_WORKER_IDLE_MS - 1);
    void begin(2);
    await settleNext();
    service.end(2);
    await vi.advanceTimersByTimeAsync(FLOOD_FILL_WORKER_IDLE_MS - 1);
    expect(executors.map((executor) => executor.disposed)).toEqual([false]);

    await vi.advanceTimersByTimeAsync(1);
    expect(executors.map((executor) => executor.disposed)).toEqual([true]);

    const revived = begin(3);
    await settleNext();
    expect((await revived).status).toBe('ok');
    expect(executors.map((executor) => executor.disposed)).toEqual([true, false]);
  });

  it('dispose ends everything, terminates the worker and refuses later work', async () => {
    const { begin, executors, service, startedLabels } = setup();
    const running = begin(1);
    const queued = service.region(1, PARAMS);
    await flush();

    service.dispose();
    expect(await running).toEqual({ sequence: 0, status: 'ended' });
    expect(await queued).toEqual({ sequence: 1, status: 'ended' });
    expect(executors.map((executor) => executor.disposed)).toEqual([true]);

    expect((await begin(2)).status).toBe('ended');
    expect((await service.region(1, PARAMS)).status).toBe('ended');
    await flush();
    expect(startedLabels()).toEqual(['begin 1']);
    expect(executors).toHaveLength(1);
  });

  it('rejects a sample that does not span one whole buffer and a press that is already open', () => {
    const { begin, service } = setup();
    const { height, width } = LINE_ART;
    const options = { contiguous: true, ...PARAMS };
    const partial = new Uint8ClampedArray(width * height * 8).subarray(0, width * height * 4);

    expect(() => service.begin(1, { data: partial, height, width }, SEED, options)).toThrow(RangeError);
    void begin(1);
    expect(() => begin(1)).toThrow(/already open/);
  });
});
