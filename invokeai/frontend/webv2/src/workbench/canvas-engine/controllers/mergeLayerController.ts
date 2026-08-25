import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterMemoryReservationResult } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { mergeDownMatrix } from '@workbench/canvas-engine/document/mergeDown';
import { canMergeSelectedRasters, getMergeVisibleRasterLayers } from '@workbench/canvas-engine/document/mergeVisible';
import { isMergeableRasterLayer } from '@workbench/canvas-engine/document/sources';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

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
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => boolean;
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
    const upperIndex = document.layers.findIndex((layer) => layer.id === upperLayerId);
    const upper = document.layers[upperIndex];
    const below = document.layers[upperIndex + 1];
    if (upperIndex < 0 || !upper || !below || !isMergeableRasterLayer(upper) || !isMergeableRasterLayer(below)) {
      return false;
    }
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
      this.deps.layers.delete(below.id);
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
    const contributors = getMergeVisibleRasterLayers(document.layers, this.deps.hasExportableContent);
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
      const liveContributors = liveDocument
        ? getMergeVisibleRasterLayers(liveDocument.layers, this.deps.hasExportableContent)
        : [];
      if (
        !liveDocument ||
        liveContributors.length !== contributors.length ||
        liveContributors.some((layer, index) => layer !== contributors[index])
      ) {
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
      for (let index = successful.length - 1; index >= 0; index -= 1) {
        const exported = successful[index]!;
        const contributor = contributors[index]!;
        context.globalAlpha = contributor.opacity;
        context.globalCompositeOperation = blendToComposite(contributor.blendMode);
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
      const selectedLayerId = liveDocument.selectedLayerId;
      const hasResult = (doc: CanvasDocumentContractV2 | null): boolean =>
        doc?.selectedLayerId === resultId && doc.layers[0] === resultLayer;
      const apply = (): void => {
        const prepared = this.deps.ctx.preparePixels(resultId, rect, pixels);
        this.deps.ctx.dispatchPrepared(
          {
            add: { index: 0, layers: [resultLayer] },
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
        label: 'Merge visible',
        redo: apply,
        replayFailureAtomic: true,
        undo: () =>
          this.deps.ctx.dispatchPrepared(
            { enabledUpdates: [], removeIds: [resultId], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
            () =>
              this.deps.ctx.getReducerDocument()?.selectedLayerId === selectedLayerId &&
              this.deps.ctx.getReducerDocument()?.layers.some((layer) => layer.id === resultId) === false,
            () =>
              this.deps.ctx.getDocument()?.selectedLayerId === selectedLayerId &&
              this.deps.ctx.getDocument()?.layers.some((layer) => layer.id === resultId) === false
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
    const contributors = document.layers.filter((layer) => selectedIds.has(layer.id));
    if (!canMergeSelectedRasters(document.layers, selectedIds, this.deps.hasExportableContent)) {
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
        const originalIds = document.layers.map((layer) => layer.id);
        const contributorIds = contributors.map((layer) => layer.id);
        const topIndex = document.layers.indexOf(contributors[0]!);
        const mergedIds = originalIds.filter((id) => !selectedIds.has(id));
        mergedIds.splice(topIndex, 0, resultId);
        const selectedLayerId = document.selectedLayerId;
        const hasMerged = (candidate: CanvasDocumentContractV2 | null): boolean =>
          candidate?.selectedLayerId === resultId &&
          candidate.layers.length === mergedIds.length &&
          candidate.layers.every((layer, index) => layer.id === mergedIds[index]);
        const hasOriginals = (candidate: CanvasDocumentContractV2 | null): boolean =>
          candidate?.selectedLayerId === selectedLayerId &&
          candidate.layers.length === originalIds.length &&
          candidate.layers.every((layer, index) => layer.id === originalIds[index]);
        const applyPrepared = (): void => {
          const prepared = this.deps.ctx.preparePixels(resultId, rect, pixels);
          this.deps.ctx.dispatchPrepared(
            {
              add: { index: topIndex, layers: [resultLayer] },
              enabledUpdates: [],
              orderedIds: mergedIds,
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
                add: { index: topIndex, layers: contributors },
                enabledUpdates: [],
                orderedIds: originalIds,
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
