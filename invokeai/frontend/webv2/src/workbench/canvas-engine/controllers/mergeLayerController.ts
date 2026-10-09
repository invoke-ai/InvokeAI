import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import {
  lookupDocumentLayer,
  mergeDownEligibility,
  compileDocumentLeaves,
} from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer, getDocumentLeaves, isNodeAbsent } from '@workbench/canvas-engine/document/documentIndex';
import { removeNodes } from '@workbench/canvas-engine/document/documentTree';
import { insertNodesAtAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { mergeDownMatrix } from '@workbench/canvas-engine/document/mergeDown';
import { canMergeSelectedRasters, getMergeVisibleRasterLeaves } from '@workbench/canvas-engine/document/mergeVisible';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  collectCompositedGroups,
  planGroupCompositeScopes,
  type GroupCompositeScope,
} from '@workbench/canvas-engine/render/groupCompositeScopes';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import {
  addedLayerHistoryBytes,
  layerEditRefusal,
  layerOperationStatus,
  paintLayerAt,
  publishAddedRasterLayer,
  rgbaBytes,
  withReplayReservation,
  type LayerEditRefusal,
  replaceLayerStep,
  type LayerPixels,
} from './editSteps';

export type MergeVisibleResult = 'merged' | LayerEditRefusal | 'nothing' | 'failed';

export type MergeDownResult =
  | 'merged'
  | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'>
  | LayerEditRefusal
  | 'failed';

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export interface MergeLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly ctx: CanvasMutationContext;
  readonly layers: LayerCacheStore;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly publishSelectedLayerIds: (primaryId: string | null, selectedIds: readonly string[]) => void;
}

const EMPTY_RECT: Rect = { height: 0, width: 0, x: 0, y: 0 };

const countScopes = (scopes: readonly GroupCompositeScope[]): number =>
  scopes.reduce((count, scope) => count + 1 + countScopes(scope.children), 0);

/** Owns destructive merge-down and non-destructive merge-visible pixel operations. */
export class MergeLayerController {
  private disposed = false;

  constructor(private readonly deps: MergeLayerControllerOptions) {}

  /**
   * Bakes the upper layer into the one below as one undo step: the lower layer keeps its id and takes the merged
   * pixels. Undo restores the lower contract and pixels, then reinserts the upper layer with its own.
   */
  mergeDown(upperLayerId: string): MergeDownResult {
    const { ctx } = this.deps;
    if (this.disposed) {
      return 'not-ready';
    }
    const document = ctx.getDocument();
    if (!document) {
      return 'missing';
    }
    const eligibility = mergeDownEligibility(document, upperLayerId);
    if (eligibility.status !== 'eligible') {
      return eligibility.status === 'missing' ? 'missing' : 'unsupported';
    }
    const upper = lookupDocumentLayer(document, eligibility.upperId)!;
    const below = lookupDocumentLayer(document, eligibility.lowerId)!;
    const upperCache = this.deps.layers.get(upper.id);
    const belowCache = this.deps.layers.get(below.id);
    const upperHasContent = this.deps.hasExportableContent(upper.id);
    const belowHasContent = this.deps.hasExportableContent(below.id);
    if (
      (upperHasContent && !upperCache) ||
      (belowHasContent && !belowCache) ||
      !this.deps.isCacheReady(upper, document) ||
      !this.deps.isCacheReady(below, document)
    ) {
      return 'not-ready';
    }
    const matrix = mergeDownMatrix(below.transform, upper.transform);
    if (!matrix) {
      return 'unsupported';
    }
    const upperRect = upperHasContent ? { ...upperCache!.rect } : EMPTY_RECT;
    const belowRect = belowHasContent ? { ...belowCache!.rect } : EMPTY_RECT;
    const mergedRect =
      !belowHasContent && !upperHasContent
        ? EMPTY_RECT
        : roundOut(
            belowHasContent && upperHasContent
              ? union(belowRect, transformBounds(matrix, upperRect))
              : belowHasContent
                ? belowRect
                : transformBounds(matrix, upperRect)
          );
    const snapshotBytes = rgbaBytes(upperRect) + rgbaBytes(belowRect);
    const txn = ctx.begin({
      historyBytes: snapshotBytes + rgbaBytes(mergedRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
    });
    if (!('publish' in txn)) {
      return layerEditRefusal(txn.status);
    }
    try {
      // Undo snapshots, the merged pixels and their prepared cache replacement.
      if (!txn.reserveRaster(snapshotBytes + rgbaBytes(mergedRect) * 2)) {
        return 'over-budget';
      }
      const snapshot = (rect: Rect, layerId: string): LayerPixels => {
        const pixels = this.deps.backend.createSurface(rect.width, rect.height);
        if (!isEmpty(rect)) {
          pixels.ctx.drawImage(this.deps.layers.get(layerId)!.surface.canvas, 0, 0);
        }
        return { pixels, rect };
      };
      const upperPixels = snapshot(upperRect, upper.id);
      const belowPixels = snapshot(belowRect, below.id);
      const merged = this.deps.backend.createSurface(mergedRect.width, mergedRect.height);
      const context = merged.ctx;
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, mergedRect.width, mergedRect.height);
      if (belowHasContent) {
        context.drawImage(belowPixels.pixels.canvas, belowRect.x - mergedRect.x, belowRect.y - mergedRect.y);
      }
      if (upperHasContent) {
        context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - mergedRect.x, matrix.f - mergedRect.y);
        context.globalAlpha = upper.opacity;
        context.globalCompositeOperation = blendToComposite(upper.blendMode);
        context.drawImage(upperPixels.pixels.canvas, upperRect.x, upperRect.y);
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.globalAlpha = 1;
        context.globalCompositeOperation = 'source-over';
      }
      const mergedPixels: LayerPixels = { pixels: merged, rect: mergedRect };
      const source = { bitmap: null, offset: { x: mergedRect.x, y: mergedRect.y }, type: 'paint' } as const;
      const mergeMutation: CanvasProjectMutation = { source, type: 'mergeCanvasLayersDown', upperLayerId: upper.id };
      const upperAnchor = ctx.captureRestoreAnchor(upper.id)!;
      const selectedLayerId = document.selectedLayerId;
      const persist = (layer: CanvasLayerContract) => this.deps.needsPixelPersistence(layer);

      // A merge the reducer accepts cannot be expressed back as one mutation; replays split undo into two steps.
      const mergeStep = (): EditStep => {
        const prepared = ctx.preparePixels(below.id, mergedRect, merged);
        return {
          accepted: (candidate) =>
            isNodeAbsent(candidate, upper.id) &&
            (getDocumentLayer(candidate, below.id) as { source?: unknown } | undefined)?.source === source,
          install: () => ctx.installPrepared(prepared),
          mutation: mergeMutation,
        };
      };
      const replaceBelowStep = (contract: CanvasLayerContract, pixels: LayerPixels): EditStep =>
        replaceLayerStep(ctx, contract, ctx.preparePixels(below.id, pixels.rect, pixels.pixels), {
          persist: persist(contract),
          restore: getDocumentLayer(ctx.getReducerDocument(), below.id) ?? undefined,
        });
      const reinsertUpperStep = (): EditStep => {
        const prepared = ctx.preparePixels(upper.id, upperRect, upperPixels.pixels);
        return {
          accepted: (candidate) =>
            getDocumentLayer(candidate, upper.id) === upper && candidate?.selectedLayerId === selectedLayerId,
          install: () => ctx.installPrepared(prepared, persist(upper)),
          mutation: {
            add: [{ anchor: upperAnchor, nodes: [upper] }],
            enabledUpdates: [],
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          rollback: {
            mutation: { enabledUpdates: [], removeIds: [upper.id], type: 'applyCanvasLayerStackMutation' },
            restored: (candidate) => isNodeAbsent(candidate, upper.id),
          },
        };
      };
      const undo = (): void =>
        withReplayReservation(ctx, snapshotBytes, () => {
          const mergedBelow = getDocumentLayer(ctx.getReducerDocument(), below.id);
          ctx.applyStep(replaceBelowStep(below, belowPixels));
          try {
            ctx.applyStep(reinsertUpperStep());
          } catch (error) {
            if (mergedBelow) {
              ctx.applyStep(replaceBelowStep(mergedBelow, mergedPixels));
            }
            throw error;
          }
        });
      const status = layerOperationStatus(
        txn.publish('Merge down', mergeStep(), {
          bytes: snapshotBytes + rgbaBytes(mergedRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
          heldAssetRefs: collectHistoryMediaRefs(upper, below),
          redo: () => withReplayReservation(ctx, rgbaBytes(mergedRect), () => ctx.applyStep(mergeStep())),
          undo,
        })
      );
      return status === 'committed' ? 'merged' : status;
    } finally {
      txn.end();
    }
  }

  async mergeVisible(): Promise<MergeVisibleResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return 'busy';
    }
    const document = ctx.getDocument();
    if (!document) {
      return 'nothing';
    }
    const contributorLeaves = getMergeVisibleRasterLeaves(
      compileDocumentLeaves(document),
      this.deps.hasExportableContent
    );
    const contributors = contributorLeaves.map((leaf) => leaf.layer);
    if (contributors.length < 2) {
      return 'nothing';
    }
    const owned: Extract<ExportResult, { status: 'ok' }>[] = [];
    const acquire = async (layerId: string): Promise<ExportResult> => {
      const result = await this.deps.exportBaked(layerId);
      if (result.status === 'ok') {
        owned.push(result);
      }
      return result;
    };
    try {
      const settled = await Promise.allSettled(contributors.map((layer) => acquire(layer.id)));
      const rejected = settled.find((result) => result.status === 'rejected');
      if (rejected?.status === 'rejected') {
        throw rejected.reason instanceof Error ? rejected.reason : new Error(String(rejected.reason));
      }
      const exports = settled.map((result) => (result as PromiseFulfilledResult<ExportResult>).value);
      if (!ctx.isPermitCurrent(permit)) {
        return 'busy';
      }
      if (exports.some((result) => result.status === 'over-budget')) {
        return 'over-budget';
      }
      if (exports.some((result) => result.status !== 'ok')) {
        return 'not-ready';
      }
      for (let index = 0; index < exports.length; index += 1) {
        const exported = exports[index];
        const contributor = contributors[index];
        if (
          !exported ||
          exported.status !== 'ok' ||
          !contributor ||
          exported.guard.layer !== contributor ||
          !ctx.isGuardCurrent(exported.guard)
        ) {
          return 'not-ready';
        }
      }

      const liveDocument = ctx.getDocument();
      const liveLeaves = liveDocument
        ? getMergeVisibleRasterLeaves(compileDocumentLeaves(liveDocument), this.deps.hasExportableContent)
        : [];
      if (
        !liveDocument ||
        liveLeaves.length !== contributors.length ||
        liveLeaves.some((leaf, index) => leaf.layer !== contributors[index])
      ) {
        return 'not-ready';
      }
      // Leaf identity misses a mid-await ancestor-stack edit or an order-preserving
      // re-parent; the scope plan folds both.
      const scopes = planGroupCompositeScopes(contributorLeaves, collectCompositedGroups(document));
      const liveScopes = planGroupCompositeScopes(liveLeaves, collectCompositedGroups(liveDocument));
      if (JSON.stringify(liveScopes) !== JSON.stringify(scopes)) {
        return 'not-ready';
      }

      const successful = exports as Extract<ExportResult, { status: 'ok' }>[];
      let rect = successful[0]!.rect;
      for (let index = 1; index < successful.length; index += 1) {
        rect = union(rect, successful[index]!.rect);
      }
      rect = roundOut(rect);
      if (isEmpty(rect)) {
        return 'nothing';
      }
      const txn = ctx.begin({ historyBytes: addedLayerHistoryBytes(rect) });
      if (!('publish' in txn)) {
        return layerEditRefusal(txn.status);
      }
      try {
        // The flattened result plus one buffer per isolated group scope.
        if (!txn.reserveRaster(rgbaBytes(rect) * (1 + countScopes(scopes)))) {
          return 'over-budget';
        }
        const pixels = this.deps.backend.createSurface(rect.width, rect.height);
        const context = pixels.ctx;
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, rect.width, rect.height);
        // Composite the group to preserve member opacity and blending in merged pixels.
        const drawMerged = (
          target: RasterSurface['ctx'],
          from: number,
          to: number,
          range: readonly GroupCompositeScope[]
        ): void => {
          let scopeIndex = range.length - 1;
          for (let index = to - 1; index >= from;) {
            const scope = scopeIndex >= 0 ? range[scopeIndex]! : null;
            if (scope && index >= scope.start && index < scope.end) {
              const buffer = this.deps.backend.createSurface(rect.width, rect.height);
              buffer.ctx.setTransform(1, 0, 0, 1, 0, 0);
              buffer.ctx.clearRect(0, 0, rect.width, rect.height);
              drawMerged(buffer.ctx, scope.start, scope.end, scope.children);
              if (!isIdentityAdjustments(scope.adjustments)) {
                const scoped = buffer.ctx.getImageData(0, 0, rect.width, rect.height);
                applyAdjustments(scoped, scope.adjustments);
                buffer.ctx.putImageData(scoped, 0, 0);
              }
              target.globalAlpha = scope.opacity;
              target.globalCompositeOperation = blendToComposite(scope.blendMode);
              target.drawImage(buffer.canvas, 0, 0);
              index = scope.start - 1;
              scopeIndex -= 1;
              continue;
            }
            const exported = successful[index]!;
            const contributor = contributors[index]!;
            target.globalAlpha = contributor.opacity;
            target.globalCompositeOperation = blendToComposite(contributor.blendMode);
            target.drawImage(exported.surface.canvas, exported.rect.x - rect.x, exported.rect.y - rect.y);
            index -= 1;
          }
        };
        drawMerged(context, 0, successful.length, scopes);
        context.globalAlpha = 1;
        context.globalCompositeOperation = 'source-over';

        const status = layerOperationStatus(
          publishAddedRasterLayer(ctx, txn, {
            anchor: ctx.captureInsertionAnchor('raster', null),
            heldAssetRefs: collectHistoryMediaRefs(contributors),
            label: 'Merge visible',
            layer: paintLayerAt(ctx.createLayerId(), `${contributors[0]!.name} merged`, rect),
            pixels,
            rect,
            selectedLayerId: liveDocument.selectedLayerId,
          })
        );
        return status === 'committed' ? 'merged' : status;
      } finally {
        txn.end();
      }
    } finally {
      for (const result of owned) {
        result.release();
      }
    }
  }

  async mergeSelected(layerIds: readonly string[]): Promise<MergeVisibleResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return 'busy';
    }
    const document = ctx.getDocument();
    if (!document) {
      return 'nothing';
    }
    const selectedIds = new Set(layerIds);
    const contributors = getDocumentLeaves(document).filter((layer) => selectedIds.has(layer.id));
    if (
      !canMergeSelectedRasters(document, compileDocumentLeaves(document), selectedIds, this.deps.hasExportableContent)
    ) {
      return 'nothing';
    }
    const owned: Extract<ExportResult, { status: 'ok' }>[] = [];
    const acquire = async (layerId: string): Promise<ExportResult> => {
      const result = await this.deps.exportBaked(layerId);
      if (result.status === 'ok') {
        owned.push(result);
      }
      return result;
    };
    try {
      const settled = await Promise.allSettled(contributors.map((layer) => acquire(layer.id)));
      const rejected = settled.find((result) => result.status === 'rejected');
      if (rejected?.status === 'rejected') {
        throw rejected.reason instanceof Error ? rejected.reason : new Error(String(rejected.reason));
      }
      const exports = settled.map((result) => (result as PromiseFulfilledResult<ExportResult>).value);
      if (!ctx.isPermitCurrent(permit)) {
        return 'busy';
      }
      if (exports.some((result) => result.status === 'over-budget')) {
        return 'over-budget';
      }
      if (exports.some((result) => result.status !== 'ok')) {
        return 'not-ready';
      }
      for (let index = 0; index < exports.length; index += 1) {
        const exported = exports[index];
        const contributor = contributors[index];
        if (
          !exported ||
          exported.status !== 'ok' ||
          !contributor ||
          exported.guard.layer !== contributor ||
          !ctx.isGuardCurrent(exported.guard)
        ) {
          return 'not-ready';
        }
      }
      // The merge is built from `document`: its structure and selection, the contributors' identities and their
      // eligibility must still hold. A preview-tolerant recheck; see `CanvasMutationContext.getReducerDocument`.
      const liveDocument = ctx.getReducerDocument();
      const liveContributors = liveDocument
        ? getDocumentLeaves(liveDocument).filter((layer) => selectedIds.has(layer.id))
        : [];
      if (
        !liveDocument ||
        liveDocument.selectedLayerId !== document.selectedLayerId ||
        !haveSameStructure(liveDocument.stacks, document.stacks) ||
        liveContributors.length !== contributors.length ||
        liveContributors.some((layer, index) => layer !== contributors[index]) ||
        !canMergeSelectedRasters(
          liveDocument,
          compileDocumentLeaves(liveDocument),
          selectedIds,
          this.deps.hasExportableContent
        )
      ) {
        return 'not-ready';
      }

      const rawEntries = contributors.map((contributor) => this.deps.layers.get(contributor.id));
      if (rawEntries.some((entry) => !entry || entry.stale)) {
        return 'not-ready';
      }
      const successful = exports as Extract<ExportResult, { status: 'ok' }>[];
      let rect = successful[0]!.rect;
      for (let index = 1; index < successful.length; index += 1) {
        rect = union(rect, successful[index]!.rect);
      }
      rect = roundOut(rect);
      if (isEmpty(rect)) {
        return 'nothing';
      }
      const rawBytes = rawEntries.reduce((bytes, entry) => bytes + rgbaBytes(entry!.rect), 0);
      const mergedBytes = rgbaBytes(rect);
      const txn = ctx.begin({
        historyBytes: rawBytes + mergedBytes + contributors.length * HISTORY_ENTRY_OVERHEAD_BYTES,
      });
      if (!('publish' in txn)) {
        return layerEditRefusal(txn.status);
      }
      try {
        // Besides already-reserved baked exports, the transaction simultaneously
        // owns raw undo snapshots, the flattened result, and its prepared cache.
        if (!txn.reserveRaster(rawBytes + mergedBytes * 2)) {
          return 'over-budget';
        }
        const rawSnapshots = contributors.map((contributor, index) => {
          const entry = rawEntries[index]!;
          const snapshotPixels = this.deps.backend.createSurface(entry.rect.width, entry.rect.height);
          snapshotPixels.ctx.drawImage(entry.surface.canvas, 0, 0);
          return { layer: contributor, pixels: snapshotPixels, rect: { ...entry.rect } };
        });
        const pixels = this.deps.backend.createSurface(rect.width, rect.height);
        const context = pixels.ctx;
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, rect.width, rect.height);
        for (let index = successful.length - 1; index >= 0; index -= 1) {
          const exported = successful[index]!;
          const contributor = contributors[index]!;
          context.globalAlpha = contributor.opacity;
          context.globalCompositeOperation = 'source-over';
          context.drawImage(exported.surface.canvas, exported.rect.x - rect.x, exported.rect.y - rect.y);
        }
        context.globalAlpha = 1;
        context.globalCompositeOperation = 'source-over';

        const resultLayer = paintLayerAt(ctx.createLayerId(), `${contributors[0]!.name} merged`, rect);
        const resultId = resultLayer.id;
        const contributorIds = contributors.map((layer) => layer.id);
        const anchor = ctx.captureInsertionAnchor('raster', contributors[0]!.id);
        const restoreInsertions = contributors.map((layer) => ({
          anchor: ctx.captureRestoreAnchor(layer.id)!,
          nodes: [layer],
        }));
        const mergedStacks = removeNodes(insertNodesAtAnchor(document.stacks, anchor, [resultLayer]), selectedIds);
        const selectedLayerId = document.selectedLayerId;
        const hasMerged = (candidate: CanvasDocumentContractV3 | null): boolean =>
          candidate?.selectedLayerId === resultId && haveSameStructure(candidate.stacks, mergedStacks);
        const hasOriginals = (candidate: CanvasDocumentContractV3 | null): boolean =>
          candidate?.selectedLayerId === selectedLayerId && haveSameStructure(candidate.stacks, document.stacks);
        const merge: CanvasProjectMutation = {
          add: [{ anchor, nodes: [resultLayer] }],
          enabledUpdates: [],
          removeIds: contributorIds,
          selectedLayerId: resultId,
          type: 'applyCanvasLayerStackMutation',
        };
        const restore: CanvasProjectMutation = {
          add: restoreInsertions,
          enabledUpdates: [],
          removeIds: [resultId],
          selectedLayerId,
          type: 'applyCanvasLayerStackMutation',
        };
        const mergeStep = (): EditStep => {
          const prepared = ctx.preparePixels(resultId, rect, pixels);
          return {
            accepted: hasMerged,
            install: () => ctx.installPrepared(prepared),
            mutation: merge,
            notify: () => this.deps.publishSelectedLayerIds(resultId, [resultId]),
            rollback: { mutation: restore, restored: hasOriginals },
          };
        };
        const restoreStep = (): EditStep => {
          const prepared = rawSnapshots.map((captured) => ({
            layer: captured.layer,
            replacement: ctx.preparePixels(captured.layer.id, captured.rect, captured.pixels),
          }));
          return {
            accepted: hasOriginals,
            install: () => {
              for (const { layer, replacement } of prepared) {
                ctx.installPrepared(replacement, this.deps.needsPixelPersistence(layer));
              }
            },
            mutation: restore,
            notify: () => this.deps.publishSelectedLayerIds(selectedLayerId, contributorIds),
            rollback: { mutation: merge, restored: hasMerged },
          };
        };
        const status = layerOperationStatus(
          txn.publish('Merge selected layers', mergeStep(), {
            bytes: rawBytes + mergedBytes + contributors.length * HISTORY_ENTRY_OVERHEAD_BYTES,
            heldAssetRefs: collectHistoryMediaRefs(rawSnapshots.map(({ layer }) => layer)),
            redo: () => withReplayReservation(ctx, mergedBytes, () => ctx.applyStep(mergeStep())),
            undo: () => withReplayReservation(ctx, rawBytes, () => ctx.applyStep(restoreStep())),
          })
        );
        return status === 'committed' ? 'merged' : status;
      } finally {
        txn.end();
      }
    } finally {
      for (const result of owned) {
        result.release();
      }
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
