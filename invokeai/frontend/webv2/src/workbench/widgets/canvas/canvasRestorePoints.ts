/**
 * Restore points make the canvas's wholesale losses recoverable. Clear canvas and Discard all each keep one (the
 * reducer holds the newest `MAX_CANVAS_SNAPSHOTS` of each kind), the History pane lists them, and restoring a canvas
 * point keeps the canvas it replaces as a point of its own, so a restore can be undone too. A swap of the document
 * (clearing, restoring a canvas) first crosses the paint barrier, so neither the point nor the kept canvas misses
 * strokes still uploading; when they cannot be saved, nothing changes. Everything here acts on one project, captured by
 * its caller, never "the active project".
 */

import type {
  CanvasDocumentContractV3,
  CanvasProjectMutation,
  CanvasSnapshotKind,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

import { createNewCanvasState } from '@workbench/canvasMigration';
import { isCancellableQueueItem } from '@workbench/canvasStagingView';

/** The project's canvas engine as restore points use it: the owner of document swaps and of the paint barrier. */
export interface RestorePointEngine {
  readonly document: {
    replaceDocument(document: CanvasDocumentContractV3): boolean;
    restoreSnapshot(snapshotId: string, keepCurrentAs?: { id: string; name: string; createdAt: string }): boolean;
  };
  readonly lifecycle: { flushPendingUploads(options?: { readonly waitForHeldPixels?: boolean }): Promise<void> };
}

export interface RestorePointPorts {
  /** Applies a mutation to the caller's project; false when it changed nothing (the project closed). */
  apply(mutation: CanvasProjectMutation): boolean;
  /** The project's live canvas engine; null once its canvas closed. */
  engine(): RestorePointEngine | null;
  /** Whether the canvas holds nothing to bring back: no layer but empty masks, no staged result. */
  isBlank(): boolean;
  /**
   * Why swapping the document must wait: an operation holds it (`editing`), or a generation runs or results wait for
   * review (`canvas`), whose results a swap would throw away. Null when nothing does.
   */
  lockReason(): 'editing' | 'canvas' | null;
  createId(): string;
  now(): string;
}

/** Keeps a restore point of the canvas as it is now, its staged results included. Null when the project is gone. */
export const saveRestorePoint = (ports: RestorePointPorts, kind: CanvasSnapshotKind, name: string): string | null => {
  const id = ports.createId();
  return ports.apply({ createdAt: ports.now(), id, includeStaging: true, kind, name, type: 'saveCanvasSnapshot' })
    ? id
    : null;
};

/** The canvas generations still queued or running for this canvas session; a Clear canvas would drop their results. */
export const getInFlightCanvasItems = (
  canvas: Pick<CanvasStateContractV3, 'documentRevision'>,
  queueItems: readonly QueueItem[]
): QueueItem[] =>
  queueItems.filter(
    (item) =>
      item.snapshot.destination === 'canvas' &&
      item.snapshot.canvas.documentRevision === canvas.documentRevision &&
      isCancellableQueueItem(item)
  );

/** Saves the strokes still uploading; false when they cannot be, so a swap now would lose them. */
const crossPaintBarrier = async (engine: RestorePointEngine): Promise<boolean> => {
  try {
    await engine.lifecycle.flushPendingUploads({ waitForHeldPixels: false });
    return true;
  } catch {
    return false;
  }
};

export interface ClearCanvasRequest {
  readonly ports: RestorePointPorts;
  /** The restore point's name. */
  readonly name: string;
  /**
   * The document whose size the new canvas keeps, and the queue items to stop first (empty to let them finish; their
   * results are then dropped). Read once the strokes have saved, so a run enqueued or a resize made meanwhile counts.
   * Null when the project closed.
   */
  readonly readTarget: () => {
    readonly document: Pick<CanvasDocumentContractV3, 'height' | 'width'>;
    readonly stop: readonly string[];
  } | null;
  readonly cancelQueueItem: (queueItemId: string) => void;
}

export type ClearCanvasOutcome =
  /** `restorePointId` is null for a blank canvas, which had nothing to keep; `stopped` counts the items stopped. */
  | { status: 'cleared'; restorePointId: string | null; stopped: number }
  /** The latest strokes could not be saved, so clearing would lose them; nothing changed. */
  | { status: 'unsaved' }
  /** The canvas did not take the new document; its restore point is withdrawn, nothing changed. */
  | { status: 'failed' }
  /** The project or its canvas closed before the canvas could be cleared. */
  | { status: 'gone' };

/**
 * Clears the canvas behind a restore point; a blank canvas has nothing to keep. A wholesale replacement drops canvas
 * history, staging and Discard all points, as any new canvas session does.
 */
export const clearCanvasWithRestorePoint = async (request: ClearCanvasRequest): Promise<ClearCanvasOutcome> => {
  const engine = request.ports.engine();
  if (!engine) {
    return { status: 'gone' };
  }
  if (!(await crossPaintBarrier(engine))) {
    return { status: 'unsaved' };
  }
  // Everything from here runs synchronously, so nothing can be enqueued or resized between this read and the swap.
  const target = request.readTarget();
  if (!target) {
    return { status: 'gone' };
  }
  const isBlank = request.ports.isBlank();
  const restorePointId = isBlank ? null : saveRestorePoint(request.ports, 'clear-canvas', request.name);
  if (!isBlank && restorePointId === null) {
    return { status: 'gone' };
  }
  for (const queueItemId of target.stop) {
    request.cancelQueueItem(queueItemId);
  }
  const replaced = engine.document.replaceDocument(
    createNewCanvasState(target.document.width, target.document.height).document
  );
  if (!replaced) {
    if (restorePointId) {
      request.ports.apply({ snapshotId: restorePointId, type: 'deleteCanvasSnapshot' });
    }
    return { status: 'failed' };
  }
  return { restorePointId, status: 'cleared', stopped: target.stop.length };
};

export type RestoreOutcome =
  /**
   * `undoPointId` names the kept canvas that undoes a canvas restore; null for a Discard all point, or when the
   * replaced canvas was blank.
   */
  | { status: 'restored'; undoPointId: string | null }
  | { status: 'locked'; reason: 'editing' | 'canvas' }
  | { status: 'unsaved' }
  /** A Discard all point whose results are all staged already; it stays. */
  | { status: 'nothing' }
  | { status: 'gone' };

/**
 * Restores a point. A canvas point waits while an operation, a generation or a review holds the canvas, both before
 * and after saving the strokes still uploading, then takes the replaced canvas's place among the points
 * (`undoPointId`) unless that canvas was blank. A Discard all point only re-stages its results, so it neither waits
 * nor keeps the canvas.
 */
export const restoreFromPoint = async (
  ports: RestorePointPorts,
  point: { id: string; kind?: CanvasSnapshotKind },
  beforeName: string
): Promise<RestoreOutcome> => {
  const restagesOnly = point.kind === 'discard-results';
  const lock = restagesOnly ? null : ports.lockReason();
  if (lock) {
    return { reason: lock, status: 'locked' };
  }
  const engine = ports.engine();
  if (!engine) {
    return { status: 'gone' };
  }
  if (restagesOnly) {
    return engine.document.restoreSnapshot(point.id)
      ? { status: 'restored', undoPointId: null }
      : { status: 'nothing' };
  }
  if (!(await crossPaintBarrier(engine))) {
    return { status: 'unsaved' };
  }
  // A run enqueued or a result staged while the strokes saved would be thrown away by the swap, which runs
  // synchronously from here on.
  const lockAfterBarrier = ports.lockReason();
  if (lockAfterBarrier) {
    return { reason: lockAfterBarrier, status: 'locked' };
  }
  if (ports.isBlank()) {
    if (!engine.document.restoreSnapshot(point.id)) {
      return { status: 'gone' };
    }
    // Used up as a kept canvas would be, with nothing worth keeping in its place.
    ports.apply({ snapshotId: point.id, type: 'deleteCanvasSnapshot' });
    return { status: 'restored', undoPointId: null };
  }
  const undoPointId = ports.createId();
  return engine.document.restoreSnapshot(point.id, { createdAt: ports.now(), id: undoPointId, name: beforeName })
    ? { status: 'restored', undoPointId }
    : { status: 'gone' };
};
