import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasStackForests } from '@workbench/canvas-engine/contracts';
import type { RasterMemoryReservationResult } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { LayerStackKind } from '@workbench/canvas-engine/document/layerStacks';
import type {
  CanvasEditConcurrency,
  CanvasTransactionOutcome,
  DocumentEditPermit,
  SubsetOf,
} from '@workbench/canvas-engine/editConcurrency';
import type { History, HistoryEntry, HistoryEntryToken } from '@workbench/canvas-engine/history/history';
import type {
  CanvasEditIntent,
  CanvasMutationOrigin,
  CanvasProjectMutation,
} from '@workbench/canvas-engine/mutationContracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { EMPTY_STACKS } from '@workbench/canvas-engine/document/documentTree';
import { captureInsertionAnchor, captureRestoreAnchor } from '@workbench/canvas-engine/document/insertionAnchors';

/**
 * One visible change: an optional document mutation verified on reducer and mirror, then in-memory publication of
 * prepared pixels. Publication and history replay apply steps the same way.
 */
export interface EditStep {
  readonly mutation?: CanvasProjectMutation;
  /** The postcondition `mutation` must establish on the reducer document; the mirror must then follow it. */
  readonly accepted?: (document: CanvasDocumentContractV3 | null) => boolean;
  /** Restores the reducer when the mutation landed but its postconditions failed. */
  readonly rollback?: {
    readonly mutation: CanvasProjectMutation;
    readonly restored: (document: CanvasDocumentContractV3 | null) => boolean;
  };
  /** Publishes prepared pixels once the document accepted the mutation. Must not fail partially. */
  readonly install?: () => void;
  /** Ancillary consequences (dirty marks, selection mirrors); a failure cannot undo the accepted step. */
  readonly notify?: () => void;
}

/** Why a step did not land: the reducer left the document unchanged, or its postconditions failed. */
export class EditStepError extends Error {
  constructor(
    readonly outcome: 'rejected' | 'reverted' | 'reverted-unmirrored' | 'unreverted',
    options?: { cause?: unknown }
  ) {
    super(`Canvas edit step ${outcome === 'rejected' ? 'was rejected' : 'failed its postconditions'}.`, options);
    this.name = 'EditStepError';
  }
}

export type EditRefusal = SubsetOf<CanvasTransactionOutcome, 'busy' | 'gesture-active' | 'not-ready' | 'over-budget'>;

export type EditPublishResult =
  | { readonly status: 'committed'; readonly token: HistoryEntryToken }
  | { readonly status: EditRefusal }
  | { readonly status: 'dispatch-rejected' }
  | { readonly status: 'postcondition-failed'; readonly recovered: 'reverted' | 'reverted-unmirrored' | 'unreverted' };

/** The undo entry a publication records; its label comes from the publication. */
export type EditEntry = Omit<HistoryEntry, 'label'>;

/**
 * An admitted edit. Admission happens before any mutation, so an edit that could never keep its undo entry is
 * refused while the document, pixels and history are still untouched. Every resource the transaction reserves or
 * holds is released by `end`, which callers run in `finally`.
 */
export interface EditTransaction {
  /** Whether publication would still be accepted (permit, gesture, replay, disposal). */
  isCurrent(): boolean;
  /** Admits `bytes` more history; false when the edit could no longer keep its undo entry. */
  growHistory(bytes: number): boolean;
  /** Reserves raster bytes for preparation allocations until the transaction ends; false when they do not fit. */
  reserveRaster(bytes: number): boolean;
  /** Releases `resource` when the transaction ends. */
  hold<T extends { release(): void }>(resource: T): T;
  /** Applies `step`, records `entry` and routes the edit; refusals and failed steps leave history untouched. */
  publish(
    label: string,
    step: EditStep,
    entry: EditEntry,
    options?: {
      readonly replacing?: HistoryEntryToken;
      readonly origin?: CanvasMutationOrigin;
      /** The mutation this edit represents for routing; defaults to the step's own. */
      readonly routed?: CanvasProjectMutation;
    }
  ): EditPublishResult;
  end(): void;
}

/** The engine's single edit protocol: permits, admission, verified steps, history publication and cleanup. */
export interface CanvasMutationContext extends CanvasEditConcurrency {
  /**
   * The mirror document an edit starts from. An open structural preview ends first, restoring its baseline, so a
   * snapshot taken here (and the step or inverse built from it) never carries previewed values.
   */
  getDocument(): CanvasDocumentContractV3 | null;
  /** The reducer document as it is, previews included; for postconditions and preview baselines. */
  getReducerDocument(): CanvasDocumentContractV3 | null;
  /** Where a new `stack` layer lands: above `aboveId` when it belongs to the stack, else the stack top. */
  captureInsertionAnchor(stack: LayerStackKind, aboveId: string | null): CanvasNodeInsertionAnchor;
  /** The anchor that restores `layerId` between its current same-stack neighbours; null when absent. */
  captureRestoreAnchor(layerId: string): CanvasNodeInsertionAnchor | null;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  createLayerId(): string;
  /** The newest history entry, for coalescing edits that amend it. */
  historyTop(): HistoryEntryToken | null;
  /** Unrecorded dispatch for previews and reconciliation. */
  dispatch(action: CanvasProjectMutation, origin?: CanvasMutationOrigin): boolean;
  /**
   * Admits an edit. `gesture` marks an edit that belongs to the interaction in progress (a live stroke, a float landing
   * as the user moves on), which an active gesture therefore neither refuses nor holds back.
   */
  begin(options: {
    readonly historyBytes: number;
    readonly owner?: symbol;
    readonly gesture?: boolean;
  }): EditTransaction | { status: EditRefusal };
  /** Applies a step with verified postconditions; throws {@link EditStepError} when it does not land. */
  applyStep(step: EditStep): void;
  /** Reserves raster bytes for a replay's preparation; null when they do not fit. */
  reserveRaster(bytes: number): { release(): void } | null;
  preparePixels(layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement;
  installPrepared(prepared: PreparedLayerCacheReplacement, persist?: boolean): void;
}

/** Engine-side wiring for {@link createCanvasMutationContext}. */
export interface CanvasMutationContextDeps {
  readonly projectId: string;
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV3 | null;
  readonly subscribeReducer: (listener: () => void) => () => void;
  readonly dispatch: (action: CanvasProjectMutation, origin?: CanvasMutationOrigin) => boolean;
  readonly commitEdit: (intent: CanvasEditIntent) => void;
  readonly refreshMirror: () => void;
  readonly editingLocked: { get(): boolean; subscribe(listener: () => void): () => void };
  readonly editOwner: symbol;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly reserveRaster: (bytes: number) => RasterMemoryReservationResult;
  readonly isGestureActive: () => boolean;
  /** Ends an open structural preview, restoring its baseline; see {@link CanvasMutationContext.getDocument}. */
  readonly endStructuralPreview?: () => void;
  readonly createLayerId: () => string;
  readonly report?: (error: EditStepError, label: string) => void;
}

const runBestEffort = (notify: (() => void) | undefined): void => {
  try {
    notify?.();
  } catch {
    // The step is already accepted; a later render, dirty mark or store sync reconciles ancillary state.
  }
};

/**
 * Owns edit-permit epochs and the transaction protocol. The editing-lock and reducer subscriptions last until
 * {@link dispose}.
 */
export const createCanvasMutationContext = (
  deps: CanvasMutationContextDeps
): CanvasMutationContext & { dispose(): void } => {
  let disposed = false;
  let documentEditEpoch = 0;
  let documentEditingLocked = false;
  const syncDocumentEditingLock = (): void => {
    const nextLocked = deps.editingLocked.get();
    if (nextLocked !== documentEditingLocked) {
      documentEditingLocked = nextLocked;
      documentEditEpoch += 1;
    }
  };
  const unsubscribeDocumentEditingLock = deps.editingLocked.subscribe(syncDocumentEditingLock);
  let editRevision = 0;
  let observedDocument = deps.getReducerDocument();
  const syncEditRevision = (): void => {
    const document = deps.getReducerDocument();
    if (document !== observedDocument) {
      observedDocument = document;
      editRevision += 1;
    }
  };
  const unsubscribeReducer = deps.subscribeReducer(syncEditRevision);
  const getEditRevision = (): number => {
    syncEditRevision();
    return editRevision;
  };
  const currentStacks = (): CanvasStackForests => deps.getDocument()?.stacks ?? EMPTY_STACKS;
  const canEdit = (owner?: symbol): boolean =>
    !disposed && !deps.history.isReplaying() && (owner === deps.editOwner || !deps.editingLocked.get());
  const capturePermit = (owner?: symbol): DocumentEditPermit | null =>
    canEdit(owner) ? { epoch: documentEditEpoch, owner } : null;
  const isPermitCurrent = (permit: DocumentEditPermit): boolean =>
    !disposed &&
    !deps.history.isReplaying() &&
    (permit.owner === deps.editOwner || (!deps.editingLocked.get() && permit.epoch === documentEditEpoch));

  /** A postcondition that throws (a faulty state read) does not hold. */
  const holds = (check: () => boolean): boolean => {
    try {
      return check();
    } catch {
      return false;
    }
  };
  const mirrored = (): boolean => holds(() => deps.getDocument() === deps.getReducerDocument());

  /** Re-runs the mirror subscriber when an interrupted notification left it behind the reducer. */
  const reconcileMirror = (): boolean => {
    if (mirrored()) {
      return true;
    }
    try {
      deps.refreshMirror();
    } catch {
      // Judged by the identity check below.
    }
    return mirrored();
  };

  const applyStep = (step: EditStep): void => {
    if (step.mutation) {
      const before = deps.getReducerDocument();
      const accepted = step.accepted ?? ((document) => document !== before);
      let dispatchError: unknown;
      try {
        deps.dispatch(step.mutation, 'system');
      } catch (error) {
        // An observer may throw after the reducer committed; the postconditions decide.
        dispatchError = error;
      }
      let reducerDocument: CanvasDocumentContractV3 | null | undefined;
      try {
        reducerDocument = deps.getReducerDocument();
      } catch {
        // Unreadable: neither rejected nor accepted, so the rollback decides.
      }
      const landed = reducerDocument !== undefined && holds(() => accepted(reducerDocument!));
      if (reducerDocument === before && !landed) {
        throw new EditStepError('rejected', { cause: dispatchError });
      }
      if (!landed || !reconcileMirror()) {
        throw new EditStepError(rollBack(step, reducerDocument), { cause: dispatchError });
      }
    } else if (!reconcileMirror()) {
      throw new EditStepError('unreverted');
    }
    step.install?.();
    runBestEffort(step.notify);
  };

  /** Dispatches the step's rollback and reports how far the reducer and mirror returned. */
  const rollBack = (
    step: EditStep,
    /** The document the step left, or undefined when it could not be read. */
    failed: CanvasDocumentContractV3 | null | undefined
  ): 'reverted' | 'reverted-unmirrored' | 'unreverted' => {
    const { rollback } = step;
    if (!rollback) {
      return 'unreverted';
    }
    try {
      deps.dispatch(rollback.mutation, 'system');
    } catch {
      // Judged by the restoration postcondition below.
    }
    const restored = holds(() => {
      const reducerDocument = deps.getReducerDocument();
      return reducerDocument !== failed && rollback.restored(reducerDocument);
    });
    if (!restored) {
      return 'unreverted';
    }
    return reconcileMirror() ? 'reverted' : 'reverted-unmirrored';
  };

  const refusal = (owner?: symbol, gesture = false): EditRefusal | null => {
    if (disposed) {
      return 'not-ready';
    }
    if (!canEdit(owner)) {
      return 'busy';
    }
    if (!gesture && deps.isGestureActive()) {
      return 'gesture-active';
    }
    return deps.getReducerDocument() ? null : 'not-ready';
  };

  const begin = (options: {
    readonly historyBytes: number;
    readonly owner?: symbol;
    readonly gesture?: boolean;
  }): EditTransaction | { status: EditRefusal } => {
    const refused = refusal(options.owner, options.gesture);
    if (refused) {
      return { status: refused };
    }
    const permit: DocumentEditPermit = { epoch: documentEditEpoch, owner: options.owner };
    const admission = deps.history.admit(options.historyBytes);
    if (!admission) {
      return { status: 'over-budget' };
    }
    const held: { release(): void }[] = [admission];
    let ended = false;
    // An observer of the step may end the transaction mid-publication; its resources outlive that publication.
    let publishing = false;
    // A gesture's own edit may publish within it; any other waits until no gesture runs.
    const isCurrent = (): boolean =>
      !ended && isPermitCurrent(permit) && (options.gesture === true || !deps.isGestureActive());
    const release = (): void => {
      for (const resource of held.splice(0).reverse()) {
        try {
          resource.release();
        } catch {
          // Release every remaining resource regardless.
        }
      }
    };
    return {
      end: () => {
        if (ended) {
          return;
        }
        ended = true;
        if (!publishing) {
          release();
        }
      },
      growHistory: (bytes) => !ended && admission.grow(bytes),
      hold: (resource) => {
        if (ended) {
          resource.release();
        } else {
          held.push(resource);
        }
        return resource;
      },
      isCurrent,
      publish: (label, step, entry, publishOptions = {}) => {
        if (ended) {
          throw new Error('Canvas edit transaction has already ended.');
        }
        if (!isCurrent()) {
          return { status: refusal(options.owner) ?? 'busy' };
        }
        if (entry.bytes > admission.bytes && !admission.grow(entry.bytes - admission.bytes)) {
          return { status: 'over-budget' };
        }
        if (publishOptions.replacing !== undefined && deps.history.top() !== publishOptions.replacing) {
          return { status: 'busy' };
        }
        publishing = true;
        let token: HistoryEntryToken;
        try {
          applyStep({ ...step, notify: undefined });
          token = admission.publish({ ...entry, label }, publishOptions.replacing);
        } catch (error) {
          if (!(error instanceof EditStepError)) {
            throw error;
          }
          if (error.outcome === 'rejected') {
            return { status: 'dispatch-rejected' };
          }
          if (error.outcome !== 'reverted') {
            deps.report?.(error, label);
          }
          return { recovered: error.outcome, status: 'postcondition-failed' };
        } finally {
          publishing = false;
          if (ended) {
            release();
          }
        }
        const routed = publishOptions.routed ?? step.mutation;
        if (routed && (publishOptions.origin ?? 'user') === 'user') {
          runBestEffort(() => deps.commitEdit({ kind: 'mutation', mutation: routed }));
        }
        runBestEffort(step.notify);
        return { status: 'committed', token };
      },
      reserveRaster: (bytes) => {
        if (ended) {
          return false;
        }
        const reservation = deps.reserveRaster(bytes);
        if (reservation.status !== 'ok') {
          return false;
        }
        held.push(reservation.lease);
        return true;
      },
    };
  };

  return {
    applyStep,
    begin,
    canEdit,
    captureInsertionAnchor: (stack, aboveId) =>
      captureInsertionAnchor(currentStacks(), {
        aboveId,
        editRevision: getEditRevision(),
        projectId: deps.projectId,
        stack,
      }),
    capturePermit,
    captureRestoreAnchor: (layerId) =>
      captureRestoreAnchor(currentStacks(), layerId, deps.projectId, getEditRevision()),
    createLayerId: () => deps.createLayerId(),
    dispatch: (action, origin) => deps.dispatch(action, origin),
    dispose: () => {
      disposed = true;
      unsubscribeDocumentEditingLock();
      unsubscribeReducer();
    },
    getDocument: () => {
      deps.endStructuralPreview?.();
      return deps.getDocument();
    },
    getEditRevision,
    getReducerDocument: () => deps.getReducerDocument(),
    historyTop: () => deps.history.top(),
    installPrepared: (prepared, persist) => deps.installPrepared(prepared, persist),
    isGestureActive: () => deps.isGestureActive(),
    isGuardCurrent: (guard) => deps.isGuardCurrent(guard),
    isPermitCurrent,
    preparePixels: (layerId, rect, pixels) => deps.preparePixels(layerId, rect, pixels),
    projectId: deps.projectId,
    reserveRaster: (bytes) => {
      const reservation = deps.reserveRaster(bytes);
      return reservation.status === 'ok' ? reservation.lease : null;
    },
  };
};
