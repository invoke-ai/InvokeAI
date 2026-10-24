import type { WorkingMemoryPlan, WorkingMemoryRunCreated } from '@features/models/core/workingMemory';
import type { ConnectionListener } from '@platform/transport/socketHub';

import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { CalibrateRequest } from './workingMemoryApi';

const hub = vi.hoisted(() => ({ statusChanged: null as ((payload: unknown) => void) | null }));

vi.mock('@platform/transport/socketHub', () => ({
  socketHub: {
    on: (event: string, listener: (payload: unknown) => void) => {
      if (event === 'queue_item_status_changed') {
        hub.statusChanged = listener;
      }
      return () => {
        hub.statusChanged = null;
      };
    },
    onConnectionChange: (listener: ConnectionListener) => {
      listener('connected');
      return () => undefined;
    },
  },
}));

const api = vi.hoisted(() => ({
  acknowledgeCalibrationEnqueue: vi.fn((_idempotencyKey: string, _signal?: AbortSignal) => Promise.resolve()),
  calibrateWorkingMemory:
    vi.fn<(modelKey: string, request: CalibrateRequest, signal?: AbortSignal) => Promise<WorkingMemoryRunCreated>>(),
  cancelCalibrationRun: vi.fn(),
  getWorkingMemoryPlan: vi.fn(),
  getWorkingMemoryStatus: vi.fn(),
  getWorkingMemorySummary: vi.fn(),
  resetWorkingMemory: vi.fn(),
}));

vi.mock('./workingMemoryApi', () => api);

const { attachWorkingMemoryRealtime, startCalibration } = await import('./workingMemoryCalibration');
const { workingMemoryKeys } = await import('./workingMemoryQueries');
const { calibrationRunOrigin } = await import('@features/models/core/workingMemory');
const { accountLifecycle, captureAccountScope } = await import('@platform/state/accountLifecycle');
const { ApiError } = await import('@platform/transport/http');

let queryClient: QueryClient;
let detach: () => void;

beforeEach(() => {
  accountLifecycle.activate('wm-realtime-test', ':user:wm-realtime-test');
  queryClient = new QueryClient();
  detach = attachWorkingMemoryRealtime(queryClient);
});

afterEach(() => {
  detach();
  queryClient.clear();
  accountLifecycle.invalidate();
  vi.useRealTimers();
  vi.clearAllMocks();
});

it("marks every model's plan stale when a calibration item changes status, without reading one then", () => {
  const owner = captureAccountScope();
  const plans = [workingMemoryKeys.plan(owner, 'sdxl'), workingMemoryKeys.plan(owner, 'flux')];
  const read = vi.fn(() => Promise.resolve({}));
  const unsubscribe = plans.map((key) => {
    queryClient.setQueryData(key, { supported: true });
    // An observed plan, as an open section has: an invalidation that refetched would read it now.
    return new QueryObserver(queryClient, { queryFn: read, queryKey: key, staleTime: Infinity }).subscribe(
      () => undefined
    );
  });

  hub.statusChanged?.({ origin: 'generation' });
  expect(plans.map((key) => queryClient.getQueryState(key)?.isInvalidated)).toEqual([false, false]);

  hub.statusChanged?.({ origin: calibrationRunOrigin('run-1'), status: 'completed' });

  expect(plans.map((key) => queryClient.getQueryState(key)?.isInvalidated)).toEqual([true, true]);
  expect(read).not.toHaveBeenCalled();
  for (const stop of unsubscribe) {
    stop();
  }
});

describe('posting the steps of a calibration', () => {
  const MODEL_KEY = 'sdxl';
  // One case, one warm-up and two measured runs: three steps.
  const PLAN: WorkingMemoryPlan = {
    calibrated_through: null,
    cases: [
      {
        components: {},
        estimator: 'sdxl_vae_decode',
        height: 512,
        id: 'sdxl_vae_decode-decode-fp16-512x512',
        line: 'decode_fp16',
        mode: 'txt2img',
        model_key: MODEL_KEY,
        negative_prompt: '',
        positive_prompt: 'a lake',
        seed: 1,
        settings: { vaePrecision: 'fp16' },
        steps: 2,
        width: 512,
      },
    ],
    device_class: { id: '0123456789abcdef', name: 'Fixture GPU' },
    estimated_seconds: 60,
    estimators: ['sdxl_vae_decode'],
    model_key: MODEL_KEY,
    not_covered: [],
    reason: null,
    repetitions: { measured: 2, warm_up: 1 },
    shared_with: [],
    supported: true,
  };
  const graphs = PLAN.cases.map((item) => ({ caseId: item.id, data: [], graph: { id: item.id } }));

  /**
   * Answers like the server: a new key queues its items once, a repeat of a key returns the run it queued. The first
   * `times` requests of the first key get `failure` once its items are queued, as a failed store write does.
   */
  const serve = (failure: InstanceType<typeof ApiError>, times = 1) => {
    const queued = new Map<string, WorkingMemoryRunCreated>();
    let firstKeyRequests = 0;
    api.calibrateWorkingMemory.mockImplementation((_modelKey, request) => {
      const known = queued.get(request.idempotencyKey);
      const created = known ?? {
        device_class_id: PLAN.device_class!.id,
        item_ids: [queued.size + 1],
        run_id: `run-${String(queued.size + 1)}`,
      };
      queued.set(request.idempotencyKey, created);

      if (queued.size === 1) {
        firstKeyRequests += 1;
      }
      return queued.size === 1 && firstKeyRequests <= times ? Promise.reject(failure) : Promise.resolve(created);
    });
    return queued;
  };

  const postedKeys = () => api.calibrateWorkingMemory.mock.calls.map(([, request]) => request.idempotencyKey);

  it('names the attempt and the step count it plans on every step, so the server can tell an aborted attempt', async () => {
    serve(new ApiError('{"detail":"unused"}', 503), 0);

    await startCalibration({ graphs, modelKey: MODEL_KEY, plan: PLAN, queryClient });

    const requests = api.calibrateWorkingMemory.mock.calls.map(([, request]) => request);
    expect(requests).toHaveLength(3);
    expect(requests.map((request) => request.attemptSteps)).toEqual([3, 3, 3]);
    expect(new Set(requests.map((request) => request.attemptId)).size).toBe(1);
  });

  it('repeats a step the server answered with 503 after queuing it, with the same key, then posts the later steps', async () => {
    vi.useFakeTimers();
    const queued = serve(new ApiError('{"detail":"The calibration store is locked"}', 503));

    const started = startCalibration({ graphs, modelKey: MODEL_KEY, plan: PLAN, queryClient });
    await vi.runAllTimersAsync();
    await started;

    const keys = postedKeys();
    // The first step twice under one key, then each later step once.
    expect(keys).toHaveLength(4);
    expect(keys[1]).toBe(keys[0]);
    expect(new Set(keys).size).toBe(3);
    expect(queued.size).toBe(3);
    // Every step's response arrived, so every key is acknowledged once.
    expect(api.acknowledgeCalibrationEnqueue.mock.calls.map(([key]) => key)).toEqual([...new Set(keys)]);
  });

  it('gives up on a step the server keeps answering with 503 after three requests, posting no later step', async () => {
    vi.useFakeTimers();
    serve(new ApiError('{"detail":"The calibration store is locked"}', 503), Infinity);

    const started = startCalibration({ graphs, modelKey: MODEL_KEY, plan: PLAN, queryClient });
    await vi.runAllTimersAsync();
    await started;

    const keys = postedKeys();
    expect(keys).toHaveLength(3);
    expect(new Set(keys).size).toBe(1);
    expect(api.acknowledgeCalibrationEnqueue).not.toHaveBeenCalled();
  });

  it('stops at any other answer of the server without repeating it or posting the later steps', async () => {
    vi.useFakeTimers();
    serve(new ApiError('{"detail":"The queue has no room for the run"}', 409));

    const started = startCalibration({ graphs, modelKey: MODEL_KEY, plan: PLAN, queryClient });
    await vi.runAllTimersAsync();
    await started;

    expect(postedKeys()).toHaveLength(1);
    expect(api.acknowledgeCalibrationEnqueue).not.toHaveBeenCalled();
  });
});
