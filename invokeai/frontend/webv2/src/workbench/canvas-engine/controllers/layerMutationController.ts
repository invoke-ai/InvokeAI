import type { DuplicateLayersResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import {
  getDocumentIndex,
  getDocumentLayer,
  hasDocumentNode,
  isNodeAbsent,
  outermostNodes,
  type CanvasNodeEntry,
} from '@workbench/canvas-engine/document/documentIndex';
import { cloneSubtree, collectSubtreeLeaves } from '@workbench/canvas-engine/document/documentTree';
import { insertNodesAtAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import {
  guardedResultRefusal,
  layerEditRefusal,
  type LayerPixels,
  rgbaBytes,
  withReplayReservation,
} from './editSteps';

export type CapturedLayerCache = LayerPixels | null | 'not-ready' | 'over-budget';

export type DuplicateLayerRasterPlan =
  | {
      readonly captureBytes: number;
      readonly initialReserveBytes: number;
      readonly replayReserveBytes: number;
      readonly retainForHistory: boolean;
      readonly type: 'capture';
    }
  | { readonly type: 'empty' }
  | { readonly type: 'reference' }
  | { readonly type: 'not-ready' };

type DuplicateRasterPreparationResult =
  | { readonly status: 'ready'; readonly layer: CanvasLayerContract }
  | { readonly status: 'not-ready' | 'over-budget' };

export interface LayerMutationControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'canEdit'
    | 'captureInsertionAnchor'
    | 'capturePermit'
    | 'captureRestoreAnchor'
    | 'createLayerId'
    | 'getDocument'
    | 'getEditRevision'
    | 'installPrepared'
    | 'isGestureActive'
    | 'isPermitCurrent'
    | 'preparePixels'
    | 'reserveRaster'
  >;
  /** Copies a layer's live cache; `admit` accounts its bytes first and refuses the copy as `over-budget`. */
  readonly captureCache: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3,
    admit?: (rect: Rect) => boolean
  ) => CapturedLayerCache;
  readonly discardPersisted: (layerId: string) => void;
  readonly getDuplicateRasterPlan: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3
  ) => DuplicateLayerRasterPlan;
  readonly getSelectedLayerIds: (document: CanvasDocumentContractV3) => readonly string[];
  readonly hasPendingPixelWork: (layerId: string) => boolean;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly publishSelectedLayerIds: (primaryId: string | null, selectedIds: readonly string[]) => void;
  readonly prepareDuplicateRasterSource: (layerId: string) => Promise<DuplicateRasterPreparationResult>;
  readonly pinDuplicateRasterSources: (layerIds: readonly string[]) => { release(): void };
  readonly scheduleDuplicateRasterization: (layerIds: readonly string[]) => void;
  readonly sameContract: (document: CanvasDocumentContractV3 | null, layer: CanvasLayerContract) => boolean;
  readonly trackDetached: (bytes: number) => { release(): void };
}

type StackMutation = Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>;

/** A stack edit verified by `accepted`, rolled back by `rollback` until `restored` holds. */
const stackStep = (
  mutation: StackMutation,
  accepted: (document: CanvasDocumentContractV3 | null) => boolean,
  rollback: StackMutation,
  restored: (document: CanvasDocumentContractV3 | null) => boolean,
  effects: Pick<EditStep, 'install' | 'notify'> = {}
): EditStep => ({ ...effects, accepted, mutation, rollback: { mutation: rollback, restored } });

/** The outermost requested nodes in document order, or `null` when any id is absent or none is given. */
const duplicateRoots = (document: CanvasDocumentContractV3, ids: readonly string[]): CanvasNodeEntry[] | null => {
  const index = getDocumentIndex(document);
  const unique = [...new Set(ids)];
  if (unique.length === 0 || unique.some((id) => !index.byId.has(id))) {
    return null;
  }
  return outermostNodes(index, unique);
};

/** Owns admitted, failure-atomic copy and cross-type conversion edits. */
export class LayerMutationController {
  private duplicateInFlight = false;

  constructor(private readonly options: LayerMutationControllerOptions) {}

  async duplicate(layerIds: readonly string[]): Promise<DuplicateLayersResult> {
    const { ctx, ...o } = this.options;
    const permit = ctx.capturePermit();
    if (this.duplicateInFlight || !permit || ctx.isGestureActive()) {
      return { status: 'busy' };
    }
    const document = ctx.getDocument();
    if (!document) {
      return { status: 'nothing' };
    }
    const roots = duplicateRoots(document, layerIds);
    if (!roots) {
      return { status: 'nothing' };
    }
    const sources = roots.flatMap((entry) => collectSubtreeLeaves(entry.node));
    const plans = sources.map((source) => o.getDuplicateRasterPlan(source, document));
    const notReadySources = sources.filter((_source, index) => plans[index]?.type === 'not-ready');
    if (notReadySources.length === 0) {
      return this.commitDuplicate(layerIds);
    }

    this.duplicateInFlight = true;
    const pinLease = o.pinDuplicateRasterSources(sources.map((source) => source.id));
    try {
      for (const source of notReadySources) {
        if (o.hasPendingPixelWork(source.id)) {
          return { status: 'not-ready' };
        }
        const prepared = await o.prepareDuplicateRasterSource(source.id);
        if (prepared.status !== 'ready') {
          return { status: prepared.status };
        }
        if (prepared.layer !== source) {
          return { status: 'stale' };
        }
      }
      if (!ctx.isPermitCurrent(permit) || ctx.isGestureActive() || ctx.getDocument() !== document) {
        return { status: 'stale' };
      }
      return this.commitDuplicate(layerIds);
    } finally {
      pinLease.release();
      this.duplicateInFlight = false;
    }
  }

  private commitDuplicate(layerIds: readonly string[]): DuplicateLayersResult {
    const { ctx, ...o } = this.options;
    const document = ctx.getDocument();
    if (!document) {
      return { status: 'nothing' };
    }
    const roots = duplicateRoots(document, layerIds);
    if (!roots) {
      return { status: 'nothing' };
    }
    const sources = roots.flatMap((entry) => collectSubtreeLeaves(entry.node));
    const plans = sources.map((source) => o.getDuplicateRasterPlan(source, document));
    let retainedBytes = 0;
    let reserveBytes = 0;
    for (const plan of plans) {
      if (plan.type === 'not-ready') {
        return { status: 'not-ready' };
      }
      if (plan.type === 'capture') {
        if (plan.retainForHistory) {
          retainedBytes += plan.captureBytes;
        }
        reserveBytes += plan.initialReserveBytes;
      }
    }
    const historyBytes = retainedBytes + sources.length * HISTORY_ENTRY_OVERHEAD_BYTES;
    const txn = ctx.begin({ historyBytes });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      // Immutable durable sources rebuild caches on redo, needing one insertion copy. Unpersisted paint/mask
      // pixels need separate history and live copies; both remain within the raster reservation.
      if (!txn.reserveRaster(reserveBytes)) {
        return { status: 'over-budget' };
      }
      const captures = sources.map((source, index) =>
        plans[index]?.type === 'capture' ? o.captureCache(source, document) : null
      );
      if (captures.some((capture) => capture === 'not-ready' || capture === 'over-budget')) {
        return { status: 'not-ready' };
      }
      const existingIds = new Set(getDocumentIndex(document).byId.keys());
      const idMap = new Map<string, string>();
      const clones = roots.map((entry) => {
        const { node } = cloneSubtree(entry.node, ctx.createLayerId, idMap);
        return { ...node, name: `${entry.node.name} copy` };
      });
      const cloneLeaves = new Map(
        clones.flatMap((clone) => collectSubtreeLeaves(clone)).map((leaf) => [leaf.id, leaf])
      );
      // The cloned leaf for each pixel source, in source order; clones are fresh objects, so an
      // empty plan can strip their durable pixel reference in place before they enter the document.
      const duplicates = sources.map((source, index) => {
        const duplicate = cloneLeaves.get(idMap.get(source.id)!)!;
        if (plans[index]?.type === 'empty') {
          if (duplicate.type === 'raster' || duplicate.type === 'control') {
            if (duplicate.source.type === 'paint') {
              duplicate.source = { bitmap: null, type: 'paint' };
            }
          } else {
            duplicate.mask = { ...duplicate.mask, bitmap: null, offset: { x: 0, y: 0 } };
          }
        }
        return duplicate;
      });
      const createdIds = [...idMap.values()];
      if (createdIds.some((id) => existingIds.has(id)) || new Set(createdIds).size !== createdIds.length) {
        return { status: 'stale' };
      }
      const insertions = roots.map((entry, index) => ({
        anchor: ctx.captureInsertionAnchor(entry.stack, entry.node.id),
        nodes: [clones[index]!],
      }));
      const expectedStacks = insertions.reduce(
        (stacks, insertion) => insertNodesAtAnchor(stacks, insertion.anchor, insertion.nodes),
        document.stacks
      );
      const selectedLayerId =
        (document.selectedLayerId ? idMap.get(document.selectedLayerId) : undefined) ?? clones[0]!.id;
      const previousSelectedLayerId = document.selectedLayerId;
      const previousSelectedIds = [...o.getSelectedLayerIds(document)];
      const duplicateIds = clones.map((clone) => clone.id);
      const hasDuplicates = (candidate: CanvasDocumentContractV3 | null): boolean =>
        candidate?.selectedLayerId === selectedLayerId && haveSameStructure(candidate.stacks, expectedStacks);
      const hasOriginals = (candidate: CanvasDocumentContractV3 | null): boolean =>
        candidate?.selectedLayerId === previousSelectedLayerId && haveSameStructure(candidate.stacks, document.stacks);
      const addDuplicates: StackMutation = {
        add: insertions,
        enabledUpdates: [],
        selectedLayerId,
        type: 'applyCanvasLayerStackMutation',
      };
      const removeDuplicates: StackMutation = {
        enabledUpdates: [],
        removeIds: duplicateIds,
        selectedLayerId: previousSelectedLayerId,
        type: 'applyCanvasLayerStackMutation',
      };
      const added = (
        prepared: readonly { duplicate: CanvasLayerContract; replacement: PreparedLayerCacheReplacement }[]
      ): EditStep =>
        stackStep(addDuplicates, hasDuplicates, removeDuplicates, hasOriginals, {
          install: () => {
            for (const { duplicate, replacement } of prepared) {
              ctx.installPrepared(replacement, o.needsPixelPersistence(duplicate));
            }
          },
          notify: () => o.publishSelectedLayerIds(selectedLayerId, duplicateIds),
        });
      const retainedCaptures = captures.map((capture, index) =>
        plans[index]?.type === 'capture' && plans[index].retainForHistory ? capture : null
      );
      const initialPrepared = captures.flatMap((capture, index) => {
        if (!capture || typeof capture === 'string') {
          return [];
        }
        const duplicate = duplicates[index]!;
        const plan = plans[index]!;
        return [
          {
            duplicate,
            replacement:
              plan.type === 'capture' && plan.retainForHistory
                ? ctx.preparePixels(duplicate.id, capture.rect, capture.pixels)
                : { layerId: duplicate.id, rect: capture.rect, surface: capture.pixels },
          },
        ];
      });
      let detachedLease: { release(): void } | null = null;
      const redo = (): void => {
        const replayPlans = sources.map((_source, index) => {
          const originalPlan = plans[index]!;
          if (originalPlan.type === 'capture' && originalPlan.retainForHistory) {
            return originalPlan;
          }
          return originalPlan.type === 'capture' ? ({ type: 'reference' } as const) : originalPlan;
        });
        const replayBytes = replayPlans.reduce(
          (total, plan) => total + (plan.type === 'capture' ? plan.replayReserveBytes : 0),
          0
        );
        withReplayReservation(ctx, replayBytes, () => {
          const prepared = replayPlans.flatMap((plan, index) => {
            if (plan.type !== 'capture') {
              return [];
            }
            const duplicate = duplicates[index]!;
            const retained = retainedCaptures[index];
            if (!retained || typeof retained === 'string') {
              throw new Error('Layer pixels are not ready to restore duplicated layers');
            }
            return [{ duplicate, replacement: ctx.preparePixels(duplicate.id, retained.rect, retained.pixels) }];
          });
          ctx.applyStep(added(prepared));
        });
        o.scheduleDuplicateRasterization(
          replayPlans.flatMap((plan, index) =>
            plan.type === 'reference' && plans[index]?.type === 'capture' ? [duplicates[index]!.id] : []
          )
        );
      };
      const result = txn.publish(
        duplicates.length === 1 ? 'Duplicate layer' : 'Duplicate layers',
        added(initialPrepared),
        {
          bytes: historyBytes,
          dispose: () => detachedLease?.release(),
          heldAssetRefs: collectHistoryMediaRefs(duplicates),
          redo,
          undo: () =>
            ctx.applyStep(
              stackStep(removeDuplicates, hasOriginals, addDuplicates, hasDuplicates, {
                notify: () => o.publishSelectedLayerIds(previousSelectedLayerId, previousSelectedIds),
              })
            ),
        }
      );
      if (result.status !== 'committed') {
        return { status: guardedResultRefusal(result) };
      }
      if (retainedBytes > 0) {
        detachedLease = o.trackDetached(retainedBytes);
      }
      return { duplicateIds, selectedLayerId, status: 'duplicated' };
    } finally {
      txn.end();
    }
  }

  copy(label: string, sourceLayerId: string, layer: CanvasLayerContract, anchor: CanvasNodeInsertionAnchor): boolean {
    const { ctx, ...o } = this.options;
    const document = ctx.getDocument();
    const source = getDocumentLayer(document, sourceLayerId);
    if (
      !document ||
      !source ||
      anchor.capturedEditRevision !== ctx.getEditRevision() ||
      hasDocumentNode(document, layer.id)
    ) {
      return false;
    }
    const txn = ctx.begin({ historyBytes: HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return false;
    }
    try {
      const captured = o.captureCache(
        source,
        document,
        (rect) => txn.growHistory(rgbaBytes(rect)) && txn.reserveRaster(rgbaBytes(rect))
      );
      if (captured === 'not-ready' || captured === 'over-budget') {
        return false;
      }
      const selectedLayerId = document.selectedLayerId;
      const hasCopy = (candidate: CanvasDocumentContractV3 | null): boolean =>
        candidate?.selectedLayerId === layer.id && getDocumentLayer(candidate, layer.id) === layer;
      const hasNoCopy = (candidate: CanvasDocumentContractV3 | null): boolean =>
        candidate?.selectedLayerId === selectedLayerId && isNodeAbsent(candidate, layer.id);
      const addCopy: StackMutation = {
        add: [{ anchor, nodes: [layer] }],
        enabledUpdates: [],
        selectedLayerId: layer.id,
        type: 'applyCanvasLayerStackMutation',
      };
      const removeCopy: StackMutation = {
        enabledUpdates: [],
        removeIds: [layer.id],
        selectedLayerId,
        type: 'applyCanvasLayerStackMutation',
      };
      const added = (prepared: PreparedLayerCacheReplacement | null): EditStep =>
        stackStep(addCopy, hasCopy, removeCopy, hasNoCopy, {
          install: prepared ? () => ctx.installPrepared(prepared, o.needsPixelPersistence(layer)) : undefined,
        });
      const prepare = (): PreparedLayerCacheReplacement | null =>
        captured ? ctx.preparePixels(layer.id, captured.rect, captured.pixels) : null;
      const result = txn.publish(label, added(prepare()), {
        bytes: (captured ? rgbaBytes(captured.rect) : 0) + HISTORY_ENTRY_OVERHEAD_BYTES,
        heldAssetRefs: collectHistoryMediaRefs(layer),
        redo: () => replayPrepared(ctx, captured, () => ctx.applyStep(added(prepare()))),
        undo: () => ctx.applyStep(stackStep(removeCopy, hasNoCopy, addCopy, hasCopy)),
      });
      return result.status === 'committed';
    } finally {
      txn.end();
    }
  }

  convert(label: string, expected: CanvasLayerContract, after: CanvasLayerContract): boolean {
    const { ctx, ...o } = this.options;
    if (expected.id !== after.id || expected.type === after.type) {
      return false;
    }
    const document = ctx.getDocument();
    const current = getDocumentLayer(document, expected.id);
    if (
      !document ||
      !current ||
      current !== expected ||
      current.type !== expected.type ||
      lookupDocumentLeaf(document, current.id)?.effectiveLocked !== false
    ) {
      return false;
    }
    const txn = ctx.begin({ historyBytes: HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return false;
    }
    try {
      const captured = o.captureCache(
        current,
        document,
        (rect) => txn.growHistory(rgbaBytes(rect)) && txn.reserveRaster(rgbaBytes(rect))
      );
      if (captured === 'not-ready' || captured === 'over-budget') {
        return false;
      }
      // A conversion changes stacks, so undo carries the leaf back to its captured place.
      const restoreAnchor = ctx.captureRestoreAnchor(current.id) ?? undefined;
      const before = structuredClone(current);
      const converted = (
        layer: CanvasLayerContract,
        restore: CanvasLayerContract,
        anchor?: CanvasNodeInsertionAnchor
      ): EditStep => {
        const prepared = captured ? ctx.preparePixels(layer.id, captured.rect, captured.pixels) : null;
        return {
          accepted: (candidate) => o.sameContract(candidate, layer),
          install: () => {
            try {
              o.discardPersisted(layer.id);
            } catch {
              /* Ancillary after reducer acceptance. */
            }
            if (prepared) {
              ctx.installPrepared(prepared, o.needsPixelPersistence(layer));
            }
          },
          mutation: { anchor, id: layer.id, layer, targetType: layer.type, type: 'convertCanvasLayer' },
          rollback: {
            mutation: { id: layer.id, layer: restore, targetType: restore.type, type: 'convertCanvasLayer' },
            restored: (candidate) => o.sameContract(candidate, restore),
          },
        };
      };
      const result = txn.publish(label, converted(after, current), {
        bytes: (captured ? rgbaBytes(captured.rect) : 0) + HISTORY_ENTRY_OVERHEAD_BYTES,
        heldAssetRefs: collectHistoryMediaRefs(before, after),
        redo: () => replayPrepared(ctx, captured, () => ctx.applyStep(converted(after, before))),
        undo: () => replayPrepared(ctx, captured, () => ctx.applyStep(converted(before, after, restoreAnchor))),
      });
      return result.status === 'committed';
    } finally {
      txn.end();
    }
  }

  dispose(): void {}
}

/** Runs a replay whose step prepares a copy of `captured`, reserving that copy first. */
const replayPrepared = (
  ctx: Pick<CanvasMutationContext, 'reserveRaster'>,
  captured: CapturedLayerCache,
  replay: () => void
): void => {
  withReplayReservation(ctx, captured && typeof captured !== 'string' ? rgbaBytes(captured.rect) : 0, replay);
};
