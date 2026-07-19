import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

/**
 * A claim on the current document-edit epoch. Captured before an async
 * operation and re-checked ({@link CanvasMutationContext.isPermitCurrent})
 * before publishing its result: any editing-lock transition in between bumps
 * the epoch and invalidates outstanding permits. A permit captured with the
 * engine's own edit-owner symbol bypasses the lock (the lock holder may edit).
 */
export interface DocumentEditPermit {
  readonly epoch: number;
  readonly owner?: symbol;
}

/**
 * The shared mutation substrate handed to canvas controllers: the guarded
 * document-mutation protocol (edit permits, prepared-cache dispatch with
 * reducer/mirror postconditions, layer-cache replacement install), plus the
 * small set of engine services every mutating controller needs.
 */
export interface CanvasMutationContext {
  readonly history: History;
  getDocument(): CanvasDocumentContractV2 | null;
  getReducerDocument(): CanvasDocumentContractV2 | null;
  canEdit(owner?: symbol): boolean;
  capturePermit(owner?: symbol): DocumentEditPermit | null;
  isPermitCurrent(permit: DocumentEditPermit): boolean;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  dispatch(action: CanvasProjectMutation): boolean;
  dispatchPrepared(action: CanvasProjectMutation, reducerAccepted: () => boolean, mirrorAccepted: () => boolean): void;
  preparePixels(layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement;
  installPrepared(prepared: PreparedLayerCacheReplacement, persist?: boolean): void;
  endBurst(): void;
  isGestureActive(): boolean;
  createLayerId(): string;
}

/** Engine-side wiring for {@link createCanvasMutationContext}. */
export interface CanvasMutationContextDeps {
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly dispatch: (action: CanvasProjectMutation) => boolean;
  readonly refreshMirror: () => void;
  readonly editingLocked: { get(): boolean; subscribe(listener: () => void): () => void };
  readonly editOwner: symbol;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly endBurst: () => void;
  readonly isGestureActive: () => boolean;
  readonly createLayerId: () => string;
}

/**
 * Creates the shared mutation substrate. Owns the document-edit permit epoch
 * machine (subscribed to the editing-lock store until {@link dispose}) and the
 * prepared-mutation dispatch postcondition protocol; everything else delegates
 * to the engine through `deps`.
 */
export const createCanvasMutationContext = (
  deps: CanvasMutationContextDeps
): CanvasMutationContext & { dispose(): void } => {
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
  const canEdit = (owner?: symbol): boolean => owner === deps.editOwner || !deps.editingLocked.get();
  const capturePermit = (owner?: symbol): DocumentEditPermit | null =>
    canEdit(owner) ? { epoch: documentEditEpoch, owner } : null;
  const isPermitCurrent = (permit: DocumentEditPermit): boolean =>
    permit.owner === deps.editOwner || (!deps.editingLocked.get() && permit.epoch === documentEditEpoch);

  const dispatchPrepared = (
    action: CanvasProjectMutation,
    isApplied: () => boolean,
    isMirrored: () => boolean
  ): void => {
    try {
      if (!deps.dispatch(action)) {
        throw new Error('Canvas document mutation was rejected');
      }
    } catch (error) {
      // Store subscribers run after the reducer has accepted an action. A
      // faulty observer must not strand an applied document mutation before
      // its matching engine state and history are published. Preserve real
      // reducer/dispatch failures by swallowing only when the exact intended
      // postcondition is visible in the authoritative reducer state.
      if (!isApplied()) {
        throw error;
      }
      // Notification may have been interrupted before DocumentMirror's
      // subscriber ran. Reconcile it synchronously from authoritative state
      // before publishing follow-up state or history.
      try {
        deps.refreshMirror();
      } catch (refreshError) {
        if (!isMirrored()) {
          throw refreshError;
        }
      }
      if (!isMirrored()) {
        throw error;
      }
      return;
    }

    // A reducer may reject a guarded transaction by returning the unchanged
    // state without throwing. Do not install its prepared cache or consume a
    // failure-atomic history entry unless the authoritative postcondition
    // actually landed.
    if (!isApplied()) {
      throw new Error('Canvas document mutation was rejected');
    }
    if (!isMirrored()) {
      try {
        deps.refreshMirror();
      } catch (refreshError) {
        if (!isMirrored()) {
          throw refreshError;
        }
      }
      if (!isMirrored()) {
        throw new Error('Canvas document mutation was not mirrored');
      }
    }
  };

  return {
    canEdit,
    capturePermit,
    createLayerId: () => deps.createLayerId(),
    dispatch: (action) => deps.dispatch(action),
    dispatchPrepared,
    dispose: () => unsubscribeDocumentEditingLock(),
    endBurst: () => deps.endBurst(),
    getDocument: () => deps.getDocument(),
    getReducerDocument: () => deps.getReducerDocument(),
    history: deps.history,
    installPrepared: (prepared, persist) => deps.installPrepared(prepared, persist),
    isGestureActive: () => deps.isGestureActive(),
    isGuardCurrent: (guard) => deps.isGuardCurrent(guard),
    isPermitCurrent,
    preparePixels: (layerId, rect, pixels) => deps.preparePixels(layerId, rect, pixels),
  };
};
