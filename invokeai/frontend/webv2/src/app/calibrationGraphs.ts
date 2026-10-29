/**
 * The graphs a working-memory calibration queues, compiled exactly as `pnpm graphs:export` compiles its cases, which
 * is how Generate compiles them: over the model records the caller read and the capability table the server serves.
 * The Model Manager loads this module only once Calibrate is clicked, to check that the plan compiles before its
 * confirmation opens (`compileCalibrationGraphs` in `router.tsx`); the compiler itself runs in
 * `calibrationGraphs.worker.ts`.
 */

import type { CalibrationGraph, ModelConfig, WorkingMemoryCase } from '@features/models';

import { apiFetchJson } from '@platform/transport/http';

import type { CalibrationWorkerReply } from './calibrationGraphs.worker';
import type { GraphExportResponse } from './graphExport';

const GRAPH_EXPORT_VERSION = 1;

/**
 * A plan case in the export's snake_case case format. The plan adds what the case measures (`estimator`, `line`),
 * which the export reader rejects as unknown fields.
 */
export const toGraphExportCase = ({
  estimator: _estimator,
  line: _line,
  ...record
}: WorkingMemoryCase): Omit<WorkingMemoryCase, 'estimator' | 'line'> => record;

/** Each case's graph and batch data; rejects naming the first case that cannot compile, before anything is queued. */
export const toCalibrationGraphs = (
  cases: readonly WorkingMemoryCase[],
  response: GraphExportResponse
): CalibrationGraph[] =>
  cases.map((item) => {
    const result = response.cases.find((candidate) => candidate.id === item.id);

    if (!result) {
      throw new Error(`Case ${item.id} was not compiled.`);
    }

    if ('error' in result) {
      throw new Error(`Case ${item.id}: ${result.error}`);
    }

    return { caseId: item.id, data: result.batch.data, graph: result.batch.graph };
  });

/** Compiling a large plan takes seconds; a worker that has not answered by then is stuck. */
export const COMPILE_TIMEOUT_MS = 60_000;

/** Settles exactly once: with the worker's answer, or rejecting on a failure, the timeout or `signal`. */
const runWorker = (request: unknown, signal: AbortSignal): Promise<GraphExportResponse> =>
  new Promise((resolve, reject) => {
    signal.throwIfAborted();

    const worker = new Worker(new URL('./calibrationGraphs.worker.ts', import.meta.url), {
      name: 'calibration-graphs',
      type: 'module',
    });
    const settle = (error: Error | null, response?: GraphExportResponse) => {
      worker.terminate();
      clearTimeout(timeout);
      signal.removeEventListener('abort', onAbort);

      if (error) {
        reject(error);
      } else {
        resolve(response!);
      }
    };
    const onAbort = () => settle(signal.reason instanceof Error ? signal.reason : new Error('Compiling was stopped.'));
    const timeout = setTimeout(
      () => settle(new Error(`The graph compiler did not answer within ${String(COMPILE_TIMEOUT_MS / 1_000)} s.`)),
      COMPILE_TIMEOUT_MS
    );

    signal.addEventListener('abort', onAbort, { once: true });
    worker.addEventListener('message', (event: MessageEvent<CalibrationWorkerReply>) => {
      if (event.data.ok) {
        settle(null, event.data.response);
      } else {
        settle(new Error(event.data.error));
      }
    });
    worker.addEventListener('messageerror', () => settle(new Error('The graph compiler’s answer could not be read.')));
    worker.addEventListener('error', (event) => {
      settle(new Error(event.message || 'The graph compiler did not start.'));
    });
    worker.postMessage(request);
  });

export const compileCalibrationGraphs = async (
  cases: readonly WorkingMemoryCase[],
  models: Promise<readonly ModelConfig[]>,
  signal: AbortSignal
): Promise<CalibrationGraph[]> => {
  const [capabilities, records] = await Promise.all([
    apiFetchJson<unknown>('/api/v2/models/capabilities', { signal }),
    models,
  ]);
  const response = await runWorker(
    { capabilities, cases: cases.map(toGraphExportCase), models: records, version: GRAPH_EXPORT_VERSION },
    signal
  );

  return toCalibrationGraphs(cases, response);
};
