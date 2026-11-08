/**
 * Compiles working-memory calibration cases with the offline graph export (`graphExport.ts`), in a worker. Its own
 * bundle keeps Generate's compiler out of both routes' chunk graphs, and its own capability table cannot replace the
 * one an open editor holds.
 */

import { exportGraphs } from './graphExport';

export type CalibrationWorkerReply =
  | { ok: true; response: ReturnType<typeof exportGraphs> }
  | { ok: false; error: string };

interface WorkerScope {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: CalibrationWorkerReply): void;
}

const scope = globalThis as unknown as WorkerScope;

scope.addEventListener('message', (event) => {
  try {
    scope.postMessage({ ok: true, response: exportGraphs(event.data) });
  } catch (error) {
    scope.postMessage({ error: error instanceof Error ? error.message : String(error), ok: false });
  }
});
