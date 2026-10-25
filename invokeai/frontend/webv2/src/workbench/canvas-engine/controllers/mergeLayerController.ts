import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
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
import {
  applyAdjustments,
  hasSpatialAdjustments,
  isIdentityAdjustments,
} from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  adjustExactly,
  type AdjustmentRefusal,
  adjustmentRefusal,
  adjustSurfaceExactly,
  imageDataOf,
  isAdjustmentRefusal,
  type SpatialRunner,
} from '@workbench/canvas-engine/render/exactAdjustments';
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

export type MergeVisibleResult = 'merged' | LayerEditRefusal | 'nothing' | 'failed' | AdjustmentRefusal;

export type MergeDownResult =
  | 'merged'
  | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'>
  | LayerEditRefusal
  | 'failed'
  | AdjustmentRefusal;

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' }
  | AdjustmentRefusal;

export interface MergeLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly ctx: CanvasMutationContext;
  readonly layers: LayerCacheStore;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  /** The display's current, exact adjusted copy of a raster layer's live cache, or null; never builds one. */
  readonly peekAdjustedSurface: (layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null;
  /** Evaluates spatial adjustment runs; without it a spatial stack refuses `error`. */
  readonly runSpatial?: SpatialRunner;
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
   * pixels, with each layer's own adjustment stack baked in layer-local pixels before the upper one is resampled, so
   * the merged layer (whose contract has no stack) shows what the canvas showed. Undo restores the lower contract and
   * raw pixels, then reinserts the upper layer with its own. A spatial stack without an exact display copy is
   * evaluated in the spatial worker first; the merge then lands only if neither layer nor the document changed.
   */
  async mergeDown(upperLayerId: string): Promise<MergeDownResult> {
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
    // Each stack applies to its layer's raw pixels before compositing: the display's exact adjusted copy when it
    // has one, else a layer-local copy (surface plus ImageData) adjusted here.
    const ownStack = (layer: CanvasLayerContract, cache: LayerCacheEntry | undefined) =>
      cache && layer.type === 'raster' && layer.adjustments && !isIdentityAdjustments(layer.adjustments)
        ? { adjustments: layer.adjustments, displayed: this.deps.peekAdjustedSurface(layer, cache) }
        : null;
    const upperStack = upperHasContent ? ownStack(upper, upperCache) : null;
    const belowStack = belowHasContent ? ownStack(below, belowCache) : null;
    const adjustedCopyBytes =
      (upperStack && !upperStack.displayed ? rgbaBytes(upperRect) * 2 : 0) +
      (belowStack && !belowStack.displayed ? rgbaBytes(belowRect) * 2 : 0);
    const txn = ctx.begin({
      historyBytes: snapshotBytes + rgbaBytes(mergedRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
    });
    if (!('publish' in txn)) {
      return layerEditRefusal(txn.status);
    }
    try {
      // Undo snapshots, adjusted copies, the merged pixels and their prepared cache replacement.
      if (!txn.reserveRaster(snapshotBytes + adjustedCopyBytes + rgbaBytes(mergedRect) * 2)) {
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
      const revision = ctx.getEditRevision();
      const versions = [upperCache?.version, belowCache?.version];
      // A layer's stack works in layer-local pixels; surface (0,0) sits at its cache rect's origin.
      const adjusted = async (
        layer: CanvasLayerContract,
        pixels: LayerPixels,
        stack: ReturnType<typeof ownStack>
      ): Promise<RasterSurface | AdjustmentRefusal | 'aborted'> => {
        if (!stack || stack.displayed) {
          return stack?.displayed ?? pixels.pixels;
        }
        const result = await adjustSurfaceExactly(
          (width, height) => this.deps.backend.createSurface(width, height),
          pixels.pixels,
          pixels.rect,
          stack.adjustments,
          { run: this.deps.runSpatial }
        );
        return result.status === 'ok'
          ? result.surface
          : result.status === 'aborted'
            ? 'aborted'
            : adjustmentRefusal(document, layer.id, result.status);
      };
      const upperDrawn = await adjusted(upper, upperPixels, upperStack);
      if (upperDrawn === 'aborted' || isAdjustmentRefusal(upperDrawn)) {
        return upperDrawn === 'aborted' ? 'not-ready' : upperDrawn;
      }
      const belowDrawn = await adjusted(below, belowPixels, belowStack);
      if (belowDrawn === 'aborted' || isAdjustmentRefusal(belowDrawn)) {
        return belowDrawn === 'aborted' ? 'not-ready' : belowDrawn;
      }
      // Evaluation yields; the merge stands only for the document and pixels it read, and the display copies drawn
      // below are still those pixels' exact copies.
      if (
        this.disposed ||
        !txn.isCurrent() ||
        ctx.getEditRevision() !== revision ||
        this.deps.layers.peek(upper.id) !== upperCache ||
        this.deps.layers.peek(below.id) !== belowCache ||
        upperCache?.version !== versions[0] ||
        belowCache?.version !== versions[1] ||
        (upperStack?.displayed && this.deps.peekAdjustedSurface(upper, upperCache!) !== upperStack.displayed) ||
        (belowStack?.displayed && this.deps.peekAdjustedSurface(below, belowCache!) !== belowStack.displayed)
      ) {
        return 'not-ready';
      }
      const merged = this.deps.backend.createSurface(mergedRect.width, mergedRect.height);
      const context = merged.ctx;
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, mergedRect.width, mergedRect.height);
      if (belowHasContent) {
        context.drawImage(belowDrawn.canvas, belowRect.x - mergedRect.x, belowRect.y - mergedRect.y);
      }
      if (upperHasContent) {
        context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - mergedRect.x, matrix.f - mergedRect.y);
        context.globalAlpha = upper.opacity;
        context.globalCompositeOperation = blendToComposite(upper.blendMode);
        context.drawImage(upperDrawn.canvas, upperRect.x, upperRect.y);
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
      const refused = exports.find(isAdjustmentRefusal);
      if (refused) {
        return refused;
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
        // Composite the group to preserve member opacity and blending in merged pixels. A group stack works in
        // document pixels; the buffer is `rect` at scale 1. `rect` holds every member, so a spatial stack sees all
        // of its group's content (its alpha-bounds domain ignores the transparent margin).
        const drawMerged = async (
          target: RasterSurface['ctx'],
          from: number,
          to: number,
          range: readonly GroupCompositeScope[]
        ): Promise<AdjustmentRefusal | 'aborted' | null> => {
          let scopeIndex = range.length - 1;
          for (let index = to - 1; index >= from;) {
            const scope = scopeIndex >= 0 ? range[scopeIndex]! : null;
            if (scope && index >= scope.start && index < scope.end) {
              const buffer = this.deps.backend.createSurface(rect.width, rect.height);
              buffer.ctx.setTransform(1, 0, 0, 1, 0, 0);
              buffer.ctx.clearRect(0, 0, rect.width, rect.height);
              const inner = await drawMerged(buffer.ctx, scope.start, scope.end, scope.children);
              if (inner) {
                return inner;
              }
              if (hasSpatialAdjustments(scope.adjustments)) {
                const scoped = buffer.ctx.getImageData(0, 0, rect.width, rect.height);
                const result = await adjustExactly(scoped, scope.adjustments, rect, { run: this.deps.runSpatial });
                if (result.status !== 'ok') {
                  return result.status === 'aborted' ? 'aborted' : adjustmentRefusal(document, scope.id, result.status);
                }
                buffer.ctx.putImageData(imageDataOf(result.data, rect.width, rect.height), 0, 0);
              } else if (!isIdentityAdjustments(scope.adjustments)) {
                const scoped = buffer.ctx.getImageData(0, 0, rect.width, rect.height);
                applyAdjustments(scoped, scope.adjustments, rect);
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
          return null;
        };
        const refusal = await drawMerged(context, 0, successful.length, scopes);
        if (refusal) {
          return refusal === 'aborted' ? 'not-ready' : refusal;
        }
        // A group job yields: the merged layer stands only for contributors that still look as they were read.
        if (this.disposed || successful.some((exported) => !ctx.isGuardCurrent(exported.guard))) {
          return 'not-ready';
        }
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
      const refused = exports.find(isAdjustmentRefusal);
      if (refused) {
        return refused;
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
