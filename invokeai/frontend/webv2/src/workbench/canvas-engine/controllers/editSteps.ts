/**
 * Edit-step building blocks shared by the layer controllers: byte accounting, replay reservations, the steps that
 * add, replace and remove layers with their pixels, and how a publication's outcome is reported to each kind of
 * caller.
 */

import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { CanvasTransactionOutcome, SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { HeldAssetRefs } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer, isNodeAbsent } from '@workbench/canvas-engine/document/documentIndex';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';

import type {
  CanvasMutationContext,
  EditPublishResult,
  EditRefusal,
  EditStep,
  EditTransaction,
} from './mutationContext';

/** Layer-local pixels an edit or its undo step installs. */
export interface LayerPixels {
  readonly pixels: RasterSurface;
  readonly rect: Rect;
}

export const rgbaBytes = (rect: Rect): number => Math.max(0, rect.width) * Math.max(0, rect.height) * 4;

// ---- Publication outcomes ----------------------------------------------------------------------------------

/**
 * Whether a layer operation can read a layer's cache now: `not-ready` while its pixels load (waiting helps),
 * `unreadable` once rasterizing its current source failed (waiting does not; the operation refuses and says so).
 */
export type LayerCacheReadiness = 'ready' | 'not-ready' | 'unreadable';

/** Layers read together: one that cannot be read refuses the set even while others still load. */
export const readinessOfAll = (readiness: readonly LayerCacheReadiness[]): LayerCacheReadiness =>
  readiness.includes('unreadable') ? 'unreadable' : readiness.includes('not-ready') ? 'not-ready' : 'ready';

/** A refused admission as callers report it; a gesture in progress reads as contention. */
export type LayerEditRefusal = SubsetOf<CanvasTransactionOutcome, 'busy' | 'not-ready' | 'over-budget'>;

export const layerEditRefusal = (status: EditRefusal): LayerEditRefusal =>
  status === 'gesture-active' ? 'busy' : status;

/** How an ordinary layer operation's publication ended; a reducer refusal or failed postcondition applied nothing. */
export type LayerOperationStatus = 'committed' | LayerEditRefusal | 'failed';

/** Ordinary layer operations (merge, crop, copy, rasterize, insert) report a step that did not land as `failed`. */
export const layerOperationStatus = (result: EditPublishResult): LayerOperationStatus => {
  switch (result.status) {
    case 'committed':
      return 'committed';
    case 'dispatch-rejected':
    case 'postcondition-failed':
      return 'failed';
    default:
      return layerEditRefusal(result.status);
  }
};

/** How a guarded result operation reports a publication that did not commit. */
export type GuardedResultRefusal = LayerEditRefusal | SubsetOf<CanvasTransactionOutcome, 'stale'>;

/**
 * Guarded result operations (generated, filter, mask and staged results, mask edits, duplicate) were prepared
 * against a guard, so a step that did not land means the target moved on: `stale`.
 */
export const guardedResultRefusal = (
  result: Exclude<EditPublishResult, { status: 'committed' }>
): GuardedResultRefusal => {
  switch (result.status) {
    case 'dispatch-rejected':
    case 'postcondition-failed':
      return 'stale';
    default:
      return layerEditRefusal(result.status);
  }
};

// ---- Replays -------------------------------------------------------------------------------------------------

export type ReplayContext = Pick<CanvasMutationContext, 'reserveRaster'>;

/** Why a replay that cannot reserve its preparation memory stays where it is. */
export const REPLAY_RASTER_REFUSAL = 'Not enough raster memory to replay this step.';

/** Runs a replay's preparation under a raster reservation; a replay that does not fit throws and stays in place. */
export const withReplayReservation = <T>(ctx: ReplayContext, bytes: number, run: () => T): T => {
  const lease = ctx.reserveRaster(bytes);
  if (!lease) {
    throw new Error(REPLAY_RASTER_REFUSAL);
  }
  try {
    return run();
  } finally {
    lease.release();
  }
};

export type LayerStepContext = Pick<CanvasMutationContext, 'applyStep' | 'installPrepared' | 'preparePixels'> &
  ReplayContext;

/** Replays a step that installs `snapshot`, preparing its copy under a reservation. */
export const replayWithPixels = (
  ctx: LayerStepContext,
  layerId: string,
  snapshot: LayerPixels | null,
  step: (prepared: PreparedLayerCacheReplacement | null) => EditStep
): void => {
  if (!snapshot) {
    ctx.applyStep(step(null));
    return;
  }
  withReplayReservation(ctx, rgbaBytes(snapshot.rect), () =>
    ctx.applyStep(step(ctx.preparePixels(layerId, snapshot.rect, snapshot.pixels)))
  );
};

// ---- Steps ---------------------------------------------------------------------------------------------------

/** Options shared by the in-place replace steps. */
export interface ReplaceLayerStepOptions {
  readonly persist?: boolean;
  /** Runs just before each layer's pixels install, once the document accepted the contracts. */
  readonly beforeInstall?: (layerId: string) => void;
  readonly notify?: () => void;
}

/** Whether `document` holds exactly these contracts, by identity. */
const holdsContracts =
  (contracts: readonly CanvasLayerContract[]) =>
  (document: CanvasDocumentContractV3 | null): boolean =>
    contracts.every((contract) => getDocumentLayer(document, contract.id) === contract);

/** Installs each replacement's prepared pixels in order, after its `beforeInstall`; nothing to do yields undefined. */
const installReplacements = (
  ctx: Pick<CanvasMutationContext, 'installPrepared'>,
  replacements: readonly LayerReplacement[],
  options: ReplaceLayerStepOptions
): (() => void) | undefined =>
  options.beforeInstall || replacements.some(({ prepared }) => prepared)
    ? () => {
        for (const { contract, prepared } of replacements) {
          options.beforeInstall?.(contract.id);
          if (prepared) {
            ctx.installPrepared(prepared, options.persist);
          }
        }
      }
    : undefined;

/** A layer's replacement contract and the pixels that land with it. */
export interface LayerReplacement {
  readonly contract: CanvasLayerContract;
  readonly prepared: PreparedLayerCacheReplacement | null;
}

/**
 * Swaps `contract` in for its layer and, once the document accepted it, installs the prepared pixels. A step with a
 * `restore` contract rolls back to it when its postconditions fail.
 */
export const replaceLayerStep = (
  ctx: Pick<CanvasMutationContext, 'installPrepared'>,
  contract: CanvasLayerContract,
  prepared: PreparedLayerCacheReplacement | null,
  options: ReplaceLayerStepOptions & { readonly restore: CanvasLayerContract | undefined }
): EditStep => ({
  accepted: holdsContracts([contract]),
  install: installReplacements(ctx, [{ contract, prepared }], options),
  mutation: { layer: contract, layerId: contract.id, type: 'replaceCanvasLayer' },
  notify: options.notify,
  rollback: options.restore
    ? {
        mutation: { layer: options.restore, layerId: contract.id, type: 'replaceCanvasLayer' },
        restored: holdsContracts([options.restore]),
      }
    : undefined,
});

/**
 * {@link replaceLayerStep} for several same-type leaves at once, as one `replaceCanvasLayers`. It rolls back to
 * `restore` (one contract per replaced layer) when its postconditions fail; without a complete restore set it has no
 * rollback.
 */
export const replaceLayersStep = (
  ctx: Pick<CanvasMutationContext, 'installPrepared'>,
  replacements: readonly LayerReplacement[],
  options: ReplaceLayerStepOptions & { readonly restore: readonly CanvasLayerContract[] | undefined }
): EditStep => {
  const contracts = replacements.map(({ contract }) => contract);
  const restore = options.restore?.length === contracts.length ? options.restore : undefined;
  return {
    accepted: holdsContracts(contracts),
    install: installReplacements(ctx, replacements, options),
    mutation: { replacements: contracts.map((layer) => ({ layer, layerId: layer.id })), type: 'replaceCanvasLayers' },
    notify: options.notify,
    rollback: restore
      ? {
          mutation: {
            replacements: restore.map((layer) => ({ layer, layerId: layer.id })),
            type: 'replaceCanvasLayers',
          },
          restored: holdsContracts(restore),
        }
      : undefined,
  };
};

const removeLayerMutation = (layerId: string, selectedLayerId: string | null) =>
  ({ enabledUpdates: [], removeIds: [layerId], selectedLayerId, type: 'applyCanvasLayerStackMutation' }) as const;

/** Inserts `layer` (selecting it) and installs its prepared pixels; rolls back to the prior selection. */
export const addLayerStep = (
  ctx: Pick<CanvasMutationContext, 'installPrepared'>,
  layer: CanvasLayerContract,
  anchor: CanvasNodeInsertionAnchor,
  prepared: PreparedLayerCacheReplacement | null,
  options: { readonly persist: boolean; readonly previousSelectedLayerId: string | null }
): EditStep => ({
  accepted: (document) => getDocumentLayer(document, layer.id) === layer,
  install: prepared ? () => ctx.installPrepared(prepared, options.persist) : undefined,
  mutation: { anchor, layer, type: 'addCanvasLayer' },
  rollback: {
    mutation: removeLayerMutation(layer.id, options.previousSelectedLayerId),
    restored: (document) => isNodeAbsent(document, layer.id),
  },
});

/** Removes a layer an edit added and restores the selection it replaced. */
export const removeLayerStep = (layerId: string, selectedLayerId: string | null): EditStep => ({
  accepted: (document) => isNodeAbsent(document, layerId) && document?.selectedLayerId === selectedLayerId,
  mutation: removeLayerMutation(layerId, selectedLayerId),
});

// ---- New paint layers ----------------------------------------------------------------------------------------

/** A raster paint layer placing pixels at `rect` with an identity transform. */
export const paintLayerAt = (id: string, name: string, rect: Rect): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name,
  opacity: 1,
  source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

type EnabledUpdates = readonly { readonly id: string; readonly isEnabled: boolean }[];

/** A new paint layer whose pixels the undo entry owns, inserted and selected in one step. */
export interface AddedRasterLayerEdit {
  readonly label: string;
  /** A raster or control paint layer whose source offset places `pixels` at `rect`. */
  readonly layer: CanvasLayerContract;
  readonly rect: Rect;
  readonly pixels: RasterSurface;
  readonly anchor: CanvasNodeInsertionAnchor;
  /** The selection undo restores. */
  readonly selectedLayerId: string | null;
  /** Enabled flags the edit sets on other layers, and the values undo restores. */
  readonly enabled?: { readonly before: EnabledUpdates; readonly after: EnabledUpdates };
  readonly heldAssetRefs?: HeldAssetRefs;
}

/** Bytes an {@link AddedRasterLayerEdit} entry retains. */
export const addedLayerHistoryBytes = (rect: Rect): number => rgbaBytes(rect) + HISTORY_ENTRY_OVERHEAD_BYTES;

/**
 * Inserts the layer with its prepared pixels as one undo step. The cache replacement is reserved and prepared
 * before the document changes; replays prepare again under their own reservation.
 */
export const publishAddedRasterLayer = (
  ctx: LayerStepContext,
  txn: EditTransaction,
  edit: AddedRasterLayerEdit
): EditPublishResult => {
  const { anchor, layer, pixels, rect, selectedLayerId } = edit;
  const after = edit.enabled?.after ?? [];
  const before = edit.enabled?.before ?? [];
  const forward: CanvasProjectMutation = {
    add: [{ anchor, nodes: [layer] }],
    enabledUpdates: after,
    selectedLayerId: layer.id,
    type: 'applyCanvasLayerStackMutation',
  };
  const inverse: CanvasProjectMutation = {
    enabledUpdates: before,
    removeIds: [layer.id],
    selectedLayerId,
    type: 'applyCanvasLayerStackMutation',
  };
  const hasEnabled = (document: CanvasDocumentContractV3 | null, updates: EnabledUpdates): boolean =>
    updates.every((update) => getDocumentLayer(document, update.id)?.isEnabled === update.isEnabled);
  const added = (document: CanvasDocumentContractV3 | null): boolean =>
    document?.selectedLayerId === layer.id &&
    getDocumentLayer(document, layer.id) === layer &&
    hasEnabled(document, after);
  const removed = (document: CanvasDocumentContractV3 | null): boolean =>
    document !== null &&
    document.selectedLayerId === selectedLayerId &&
    isNodeAbsent(document, layer.id) &&
    hasEnabled(document, before);
  const addStep = (): EditStep => {
    const prepared = ctx.preparePixels(layer.id, rect, pixels);
    return {
      accepted: added,
      install: () => ctx.installPrepared(prepared),
      mutation: forward,
      rollback: { mutation: inverse, restored: removed },
    };
  };
  if (!txn.reserveRaster(rgbaBytes(rect))) {
    return { status: 'over-budget' };
  }
  return txn.publish(edit.label, addStep(), {
    bytes: addedLayerHistoryBytes(rect),
    heldAssetRefs: edit.heldAssetRefs ?? collectHistoryMediaRefs(layer),
    redo: () => withReplayReservation(ctx, rgbaBytes(rect), () => ctx.applyStep(addStep())),
    undo: () =>
      ctx.applyStep({ accepted: removed, mutation: inverse, rollback: { mutation: forward, restored: added } }),
  });
};
