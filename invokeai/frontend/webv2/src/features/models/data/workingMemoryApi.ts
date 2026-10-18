import type {
  WorkingMemoryPlan,
  WorkingMemoryRunCreated,
  WorkingMemoryStatus,
  WorkingMemorySummary,
} from '@features/models/core/workingMemory';

import { calibrationRunOrigin } from '@features/models/core/workingMemory';

import { browserModelsDataPort } from './transport';

/** REST client for working-memory calibration, admin-only on the server. */

const { request, requestJson } = browserModelsDataPort;

const modelPath = (key: string): string => `/api/v2/models/i/${encodeURIComponent(key)}/working_memory`;

/** Calibration items are ordinary queue items; the queue routes act on them by origin. */
const QUEUE_BASE = '/api/v1/queue/default';

export const getWorkingMemoryPlan = (key: string, signal?: AbortSignal): Promise<WorkingMemoryPlan> =>
  requestJson<WorkingMemoryPlan>(`${modelPath(key)}/plan`, { signal });

/** 409 when there is no current device class (no GPU, or several generation devices). */
export const getWorkingMemorySummary = (key: string, signal?: AbortSignal): Promise<WorkingMemorySummary> =>
  requestJson<WorkingMemorySummary>(`${modelPath(key)}/summary`, { signal });

export const getWorkingMemoryStatus = (signal?: AbortSignal): Promise<WorkingMemoryStatus> =>
  requestJson<WorkingMemoryStatus>('/api/v2/models/working_memory/status', { signal });

export interface CalibrateRequest {
  /** Groups the runs of one Calibrate click; the server returns it on every run. */
  attemptId: string;
  graph: unknown;
  data: unknown;
  idempotencyKey: string;
}

/** Enqueues one queue item as one server run. Resending with the same key returns that run and enqueues nothing. */
export const calibrateWorkingMemory = (
  key: string,
  { attemptId, data, graph, idempotencyKey }: CalibrateRequest,
  signal?: AbortSignal
): Promise<WorkingMemoryRunCreated> =>
  requestJson<WorkingMemoryRunCreated>(`${modelPath(key)}/calibrate`, {
    body: JSON.stringify({ attempt_id: attemptId, data, graph, idempotency_key: idempotencyKey }),
    method: 'POST',
    signal,
  });

/** Releases the server's receipt for a key whose response arrived; until then it counts toward a per-owner limit. */
export const acknowledgeCalibrationEnqueue = async (idempotencyKey: string, signal?: AbortSignal): Promise<void> => {
  await request(`${QUEUE_BASE}/enqueue_batch/acknowledge`, {
    body: JSON.stringify({ idempotency_key: idempotencyKey }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
    signal,
  });
};

/** Cancels a run's items, the running one included. */
export const cancelCalibrationRun = async (runId: string, signal?: AbortSignal): Promise<void> => {
  const params = new URLSearchParams({ origin_prefix: calibrationRunOrigin(runId) });

  await request(`${QUEUE_BASE}/cancel_all?${params.toString()}`, { method: 'PUT', signal });
};

/** Removes the model's runs and measurements on the current device class. */
export const resetWorkingMemory = async (key: string, signal?: AbortSignal): Promise<void> => {
  await request(modelPath(key), { method: 'DELETE', signal });
};
