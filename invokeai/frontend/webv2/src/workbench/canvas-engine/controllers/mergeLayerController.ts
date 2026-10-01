import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterMemoryReservationResult } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
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
import { collectHistoryMediaRefs } from '@workbench/canvas-engine/history/history';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  collectCompositedGroups,
  planGroupCompositeScopes,
  type GroupCompositeScope,
} from '@workbench/canvas-engine/render/groupCompositeScopes';

import type { CanvasMutationContext } from './mutationContext';

export type MergeVisibleResult = 'merged' | 'not-ready' | 'over-budget' | 'busy' | 'nothing';

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export interface MergeLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly ctx: CanvasMutationContext;
  readonly layers: LayerCacheStore;
  readonly canEdit: () => boolean;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly notifyPainted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly publishSelectedLayerIds: (primaryId: string | null, selectedIds: readonly string[]) => void;
  readonly reserve: (bytes: number) => RasterMemoryReservationResult;
}

/** Owns destructive merge-down and non-destructive merge-visible pixel operations. */
export class MergeLayerController {
  private disposed = false;

  constructor(private readonly deps: MergeLayerControllerOptions) {}

  mergeDown(upperLayerId: string): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.ctx.isGestureActive()) {
      return false;
    }
    this.deps.ctx.endBurst();
    const document = this.deps.ctx.getDocument();
    if (!document) {
      return false;
    }
    const eligibility = mergeDownEligibility(document, upperLayerId);
    if (eligibility.status !== 'eligible') {
      return false;
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
      return false;
    }
    const matrix = mergeDownMatrix(below.transform, upper.transform);
    if (!matrix) {
      return false;
    }
    if (!belowHasContent && !upperHasContent) {
      this.deps.ctx.dispatch({
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
        type: 'mergeCanvasLayersDown',
        upperLayerId,
      });
      // An empty live cache, not a missing one: the paint barrier treats a dirty layer without a cache as lost.
      this.deps.layers.delete(below.id);
      this.deps.layers.getOrCreateRect(below.id, { height: 0, width: 0, x: 0, y: 0 }).stale = false;
      this.deps.notifyPainted(below.id);
      this.deps.markDirty(below.id);
      return true;
    }
    const mergedRect = roundOut(
      belowHasContent && upperHasContent
        ? union(belowCache!.rect, transformBounds(matrix, upperCache!.rect))
        : belowHasContent
          ? belowCache!.rect
          : transformBounds(matrix, upperCache!.rect)
    );
    const merged = this.deps.backend.createSurface(mergedRect.width, mergedRect.height);
    const context = merged.ctx;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, mergedRect.width, mergedRect.height);
    if (belowHasContent) {
      context.drawImage(
        belowCache!.surface.canvas,
        belowCache!.rect.x - mergedRect.x,
        belowCache!.rect.y - mergedRect.y
      );
    }
    if (upperHasContent) {
      context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - mergedRect.x, matrix.f - mergedRect.y);
      context.globalAlpha = upper.opacity;
      context.globalCompositeOperation = blendToComposite(upper.blendMode);
      context.drawImage(upperCache!.surface.canvas, upperCache!.rect.x, upperCache!.rect.y);
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.globalAlpha = 1;
      context.globalCompositeOperation = 'source-over';
    }
    this.deps.ctx.dispatch({
      source: { bitmap: null, offset: { x: mergedRect.x, y: mergedRect.y }, type: 'paint' },
      type: 'mergeCanvasLayersDown',
      upperLayerId,
    });
    this.deps.layers.delete(below.id);
    const target = this.deps.layers.getOrCreateRect(below.id, mergedRect);
    target.surface.ctx.drawImage(merged.canvas, 0, 0);
    target.stale = false;
    this.deps.notifyPainted(below.id);
    this.deps.markDirty(below.id);
    return true;
  }

  async mergeVisible(): Promise<MergeVisibleResult> {
    const permit = this.deps.ctx.capturePermit();
    if (this.disposed || !permit || !this.deps.canEdit() || this.deps.ctx.isGestureActive()) {
      return 'busy';
    }
    this.deps.ctx.endBurst();
    const document = this.deps.ctx.getDocument();
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
      if (!this.deps.ctx.isPermitCurrent(permit)) {
        return 'busy';
      }
      if (exports.some((result) => result.status === 'over-budget')) {
        return 'over-budget';
      }
      if (exports.some((result) => result.status !== 'ok')) {
        return 'not-ready';
      }
      if (this.deps.ctx.isGestureActive()) {
        return 'busy';
      }
      for (let index = 0; index < exports.length; index += 1) {
        const exported = exports[index];
        const contributor = contributors[index];
        if (
          !exported ||
          exported.status !== 'ok' ||
          !contributor ||
          exported.guard.layer !== contributor ||
          !this.deps.ctx.isGuardCurrent(exported.guard)
        ) {
          return 'not-ready';
        }
      }

      const liveDocument = this.deps.ctx.getDocument();
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

      const resultId = this.deps.ctx.createLayerId();
      const resultLayer: CanvasLayerContract = {
        blendMode: 'normal',
        id: resultId,
        isEnabled: true,
        isLocked: false,
        name: `${contributors[0]!.name} merged`,
        opacity: 1,
        source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'raster',
      };
      const selectedLayerId = liveDocument.selectedLayerId;
      const anchor = this.deps.ctx.captureInsertionAnchor('raster', null);
      const hasResult = (doc: CanvasDocumentContractV3 | null): boolean =>
        doc?.selectedLayerId === resultId && getDocumentLayer(doc, resultLayer.id) === resultLayer;
      const apply = (): void => {
        const prepared = this.deps.ctx.preparePixels(resultId, rect, pixels);
        this.deps.ctx.dispatchPrepared(
          {
            add: [{ anchor, nodes: [resultLayer] }],
            enabledUpdates: [],
            selectedLayerId: resultId,
            type: 'applyCanvasLayerStackMutation',
          },
          () => hasResult(this.deps.ctx.getReducerDocument()),
          () => hasResult(this.deps.ctx.getDocument())
        );
        this.deps.ctx.installPrepared(prepared);
      };
      if (!this.deps.ctx.isPermitCurrent(permit)) {
        return 'busy';
      }
      apply();
      this.deps.ctx.history.push({
        bytes: rect.width * rect.height * 4 + 256,
        heldAssetRefs: collectHistoryMediaRefs(contributors),
        label: 'Merge visible',
        redo: apply,
        replayFailureAtomic: true,
        undo: () =>
          this.deps.ctx.dispatchPrepared(
            { enabledUpdates: [], removeIds: [resultId], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
            () =>
              this.deps.ctx.getReducerDocument()?.selectedLayerId === selectedLayerId &&
              isNodeAbsent(this.deps.ctx.getReducerDocument(), resultId),
            () =>
              this.deps.ctx.getDocument()?.selectedLayerId === selectedLayerId &&
              isNodeAbsent(this.deps.ctx.getDocument(), resultId)
          ),
      });
      return 'merged';
    } finally {
      for (const result of owned) {
        result.release();
      }
    }
  }

  async mergeSelected(layerIds: readonly string[]): Promise<MergeVisibleResult> {
    const permit = this.deps.ctx.capturePermit();
    if (this.disposed || !permit || !this.deps.canEdit() || this.deps.ctx.isGestureActive()) {
      return 'busy';
    }
    this.deps.ctx.endBurst();
    const document = this.deps.ctx.getDocument();
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
      if (!this.deps.ctx.isPermitCurrent(permit)) {
        return 'busy';
      }
      if (exports.some((result) => result.status === 'over-budget')) {
        return 'over-budget';
      }
      if (exports.some((result) => result.status !== 'ok') || this.deps.ctx.isGestureActive()) {
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
          !this.deps.ctx.isGuardCurrent(exported.guard)
        ) {
          return 'not-ready';
        }
      }
      const liveDocument = this.deps.ctx.getDocument();
      if (liveDocument !== document) {
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
      const rawBytes = rawEntries.reduce((bytes, entry) => bytes + entry!.rect.width * entry!.rect.height * 4, 0);
      const mergedBytes = rect.width * rect.height * 4;
      const historyBytes = rawBytes + mergedBytes + contributors.length * 256;
      if (!this.deps.ctx.history.canRetain(historyBytes)) {
        return 'over-budget';
      }
      // Besides already-reserved baked exports, the transaction simultaneously
      // owns raw undo snapshots, the flattened result, and its prepared cache.
      const reservation = this.deps.reserve(rawBytes + mergedBytes * 2);
      if (reservation.status === 'over-budget') {
        return 'over-budget';
      }
      try {
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

        const resultId = this.deps.ctx.createLayerId();
        const resultLayer: CanvasLayerContract = {
          blendMode: 'normal',
          id: resultId,
          isEnabled: true,
          isLocked: false,
          name: `${contributors[0]!.name} merged`,
          opacity: 1,
          source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
          type: 'raster',
        };
        const contributorIds = contributors.map((layer) => layer.id);
        const anchor = this.deps.ctx.captureInsertionAnchor('raster', contributors[0]!.id);
        const restoreInsertions = contributors.map((layer) => ({
          anchor: this.deps.ctx.captureRestoreAnchor(layer.id)!,
          nodes: [layer],
        }));
        const mergedStacks = removeNodes(insertNodesAtAnchor(document.stacks, anchor, [resultLayer]), selectedIds);
        const selectedLayerId = document.selectedLayerId;
        const hasMerged = (candidate: CanvasDocumentContractV3 | null): boolean =>
          candidate?.selectedLayerId === resultId && haveSameStructure(candidate.stacks, mergedStacks);
        const hasOriginals = (candidate: CanvasDocumentContractV3 | null): boolean =>
          candidate?.selectedLayerId === selectedLayerId && haveSameStructure(candidate.stacks, document.stacks);
        const applyPrepared = (): void => {
          const prepared = this.deps.ctx.preparePixels(resultId, rect, pixels);
          this.deps.ctx.dispatchPrepared(
            {
              add: [{ anchor, nodes: [resultLayer] }],
              enabledUpdates: [],
              removeIds: contributorIds,
              selectedLayerId: resultId,
              type: 'applyCanvasLayerStackMutation',
            },
            () => hasMerged(this.deps.ctx.getReducerDocument()),
            () => hasMerged(this.deps.ctx.getDocument())
          );
          this.deps.ctx.installPrepared(prepared);
          this.deps.publishSelectedLayerIds(resultId, [resultId]);
        };
        const redo = (): void => {
          const replayReservation = this.deps.reserve(mergedBytes);
          if (replayReservation.status === 'over-budget') {
            throw new Error('Not enough raster memory to restore the merged layer');
          }
          try {
            applyPrepared();
          } finally {
            replayReservation.lease.release();
          }
        };
        const undo = (): void => {
          const replayReservation = this.deps.reserve(rawBytes);
          if (replayReservation.status === 'over-budget') {
            throw new Error('Not enough raster memory to restore the source layers');
          }
          try {
            const prepared = rawSnapshots.map((captured) => ({
              layer: captured.layer,
              replacement: this.deps.ctx.preparePixels(captured.layer.id, captured.rect, captured.pixels),
            }));
            this.deps.ctx.dispatchPrepared(
              {
                add: restoreInsertions,
                enabledUpdates: [],
                removeIds: [resultId],
                selectedLayerId,
                type: 'applyCanvasLayerStackMutation',
              },
              () => hasOriginals(this.deps.ctx.getReducerDocument()),
              () => hasOriginals(this.deps.ctx.getDocument())
            );
            for (const { layer, replacement } of prepared) {
              this.deps.ctx.installPrepared(replacement, this.deps.needsPixelPersistence(layer));
            }
            this.deps.publishSelectedLayerIds(selectedLayerId, contributorIds);
          } finally {
            replayReservation.lease.release();
          }
        };
        if (!this.deps.ctx.isPermitCurrent(permit)) {
          return 'busy';
        }
        applyPrepared();
        this.deps.ctx.history.push({
          bytes: historyBytes,
          heldAssetRefs: collectHistoryMediaRefs(rawSnapshots.map(({ layer }) => layer)),
          label: 'Merge selected layers',
          redo,
          replayFailureAtomic: true,
          undo,
        });
        return 'merged';
      } finally {
        reservation.lease.release();
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
