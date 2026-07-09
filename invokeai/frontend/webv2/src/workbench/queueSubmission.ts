/**
 * Which pending queue items the {@link import('./WorkbenchRuntime').WorkbenchRuntime}
 * submission bridge is allowed to POST to the backend queue.
 *
 * This is the gate that decides whether a locally-enqueued `pending` queue item
 * gets handed to the coordinator (`submitGenerate` / `submitWorkflow`). It lives
 * in its own pure module — separate from the React effect that consumes it — so
 * it can be unit-tested under node-env vitest (the runtime component cannot).
 *
 * ## Why `canvas` MUST be here
 * Every Invoke — generate, workflow, AND canvas — first lands an optimistic local
 * `pending` queue item (see `enqueueCompiledSnapshot`). The runtime loop then
 * enqueues each pending item to the backend exactly once. If a source id is
 * missing from this set, its items are created locally but **never POSTed**, so
 * they stack in the queue as `pending` forever and nothing ever generates. That
 * was the canvas→canvas stall: canvas snapshots carry `sourceId: 'canvas'`, which
 * was absent from the loop's allow-list, so each Invoke only produced a dead
 * local row. Canvas items use the generate-style backend submission path: the
 * graph is compiled ahead of time, but prompt/seed batch data still has to ride
 * along so repeated invokes are not silently run with backend primitive defaults.
 *
 * `upscale` is intentionally excluded: it is not yet an available source
 * (`invocation.ts` marks it `available: false`), so it never produces a snapshot.
 */

import type { InvocationSourceId, QueueItem } from './types';

/** Source kinds whose pending queue items the runtime enqueues to the backend. */
export const BACKEND_SUBMITTABLE_SOURCE_IDS = [
  'generate',
  'workflow',
  'canvas',
] as const satisfies readonly InvocationSourceId[];

const BACKEND_SUBMITTABLE_SOURCE_ID_SET: ReadonlySet<InvocationSourceId> = new Set(BACKEND_SUBMITTABLE_SOURCE_IDS);

/** Whether a queue item's source is one the runtime knows how to enqueue. */
export const isBackendSubmittableSourceId = (sourceId: InvocationSourceId): boolean =>
  BACKEND_SUBMITTABLE_SOURCE_ID_SET.has(sourceId);

/**
 * Whether a queue item is a fresh, backend-submittable `pending` item — the
 * runtime still applies its own "already started this id" dedupe on top of this.
 */
export const shouldSubmitPendingQueueItem = (queueItem: QueueItem): boolean =>
  queueItem.status === 'pending' && isBackendSubmittableSourceId(queueItem.snapshot.sourceId);
