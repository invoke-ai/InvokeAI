import type { ExportBakedLayerPixelsOptions, LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { AdjustmentRefusal } from '@workbench/canvas-engine/render/exactAdjustments';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { intersect, isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import {
  layerEditRefusal,
  layerOperationStatus,
  rgbaBytes,
  withReplayReservation,
  type LayerEditRefusal,
  replaceLayerStep,
  replaceLayersStep,
  type LayerPixels,
} from './editSteps';

export type CropLayerResult =
  | { status: 'cropped' }
  | { status: 'missing' | 'locked' | 'unsupported' | 'empty' | LayerEditRefusal }
  | { status: 'failed'; message: string }
  | AdjustmentRefusal;

/**
 * `skippedLocked` counts locked leaves (own or ancestor lock) whose content reaches past the frame. `nothing` when
 * every unlocked leaf already fits; every refusal leaves document, pixels and history alone.
 */
export type CropAllLayersResult =
  | { status: 'cropped'; cropped: number; skippedLocked: number }
  | { status: 'nothing'; skippedLocked: number }
  | { status: 'stale' | 'failed' | LayerEditRefusal }
  | AdjustmentRefusal;

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' }
  | AdjustmentRefusal;

export interface CropLayerControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'capturePermit'
    | 'getDocument'
    | 'getReducerDocument'
    | 'installPrepared'
    | 'isPermitCurrent'
    | 'preparePixels'
    | 'reserveRaster'
  >;
  readonly backend: RasterBackend;
  readonly isSupportedSource: (source: CanvasLayerSourceContract) => boolean;
  readonly exportBaked: (
    layerId: string,
    options?: Pick<ExportBakedLayerPixelsOptions, 'applyAdjustments'>
  ) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly captureCache: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3,
    admit?: (rect: Rect) => boolean
  ) => LayerPixels | null | 'not-ready' | 'over-budget';
  readonly discardPersisted: (layerId: string) => void;
}

const failedCrop: CropLayerResult = { message: 'The crop could not be applied.', status: 'failed' };

/**
 * The contract of `layer` cropped to `rect`: its baked pixels now sit at `rect` in a paint source (or mask bitmap) with
 * an identity transform, so transform, raster adjustments and control filter no longer apply. The bitmap is
 * written by persistence once the cropped pixels land.
 */
const croppedLayerContract = (layer: CanvasLayerContract, rect: Rect): CanvasLayerContract => {
  const before = structuredClone(layer);
  const identity = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
  const offset = { x: rect.x, y: rect.y };
  if (before.type === 'raster') {
    const { adjustments: _adjustments, ...rest } = before;
    return { ...rest, source: { bitmap: null, offset, type: 'paint' }, transform: identity };
  }
  if (before.type === 'control') {
    const { filter: _filter, ...rest } = before;
    return { ...rest, source: { bitmap: null, offset, type: 'paint' }, transform: identity };
  }
  return { ...before, mask: { ...before.mask, bitmap: null, offset }, transform: identity };
};

/** `exported` pixels (placed at `exportedRect`) copied into a new surface covering exactly `rect`. */
const cropPixels = (backend: RasterBackend, exported: RasterSurface, exportedRect: Rect, rect: Rect): RasterSurface => {
  const cropped = backend.createSurface(rect.width, rect.height);
  if (rect.width > 0 && rect.height > 0) {
    cropped.ctx.setTransform(1, 0, 0, 1, 0, 0);
    cropped.ctx.clearRect(0, 0, rect.width, rect.height);
    cropped.ctx.drawImage(exported.canvas, exportedRect.x - rect.x, exportedRect.y - rect.y);
  }
  return cropped;
};

/** One leaf's crop: its contract and pixels before and after. */
interface LayerCrop {
  readonly before: CanvasLayerContract;
  readonly beforePixels: LayerPixels;
  readonly after: CanvasLayerContract;
  readonly afterPixels: LayerPixels;
}

/** A leaf's cropped pixels, copied out of its export under their own reservation, so the export is already released. */
interface CroppedLeaf {
  readonly layer: CanvasLayerContract;
  readonly pixels: LayerPixels;
  readonly reservation: { release(): void };
}

/** Owns guarded crop-to-frame conversion, of one layer or all of them, and replayable pixel snapshots. */
export class CropLayerController {
  private disposed = false;
  constructor(private readonly deps: CropLayerControllerOptions) {}

  async crop(layerId: string): Promise<CropLayerResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return { status: 'busy' };
    }
    const document = ctx.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (layer.isLocked) {
      return { status: 'locked' };
    }
    const source = renderableSourceOf(layer);
    if (!source || !this.deps.isSupportedSource(source)) {
      return { status: 'unsupported' };
    }
    try {
      const exported = await this.deps.exportBaked(layerId);
      if (exported.status === 'adjustment-refused') {
        return exported;
      }
      if (exported.status !== 'ok') {
        return { status: exported.status === 'disabled' ? 'not-ready' : exported.status };
      }
      try {
        if (!ctx.isPermitCurrent(permit)) {
          return { status: 'busy' };
        }
        const liveDocument = ctx.getDocument();
        const liveLayer = getDocumentLayer(liveDocument, layerId);
        if (!liveDocument || !liveLayer) {
          return { status: 'missing' };
        }
        if (liveLayer.isLocked) {
          return { status: 'locked' };
        }
        if (!this.deps.isGuardCurrent(exported.guard)) {
          return { status: 'not-ready' };
        }
        const liveSource = renderableSourceOf(liveLayer);
        if (!liveSource || !this.deps.isSupportedSource(liveSource)) {
          return { status: 'unsupported' };
        }
        const overlap = intersect(exported.rect, roundOut(liveDocument.bbox));
        if (!overlap || isEmpty(overlap)) {
          return { status: 'empty' };
        }
        const cropRect = roundOut(overlap);
        const txn = ctx.begin({ historyBytes: rgbaBytes(cropRect) + HISTORY_ENTRY_OVERHEAD_BYTES });
        if (!('publish' in txn)) {
          return { status: layerEditRefusal(txn.status) };
        }
        try {
          // The cropped pixels and their prepared cache replacement.
          if (!txn.reserveRaster(rgbaBytes(cropRect) * 2)) {
            return { status: 'over-budget' };
          }
          const beforePixels = this.deps.captureCache(liveLayer, liveDocument, (rect) =>
            txn.growHistory(rgbaBytes(rect))
          );
          if (beforePixels === 'over-budget') {
            return { status: 'over-budget' };
          }
          if (!beforePixels || beforePixels === 'not-ready') {
            return { status: 'not-ready' };
          }
          const before = structuredClone(liveLayer);
          const cropped = cropPixels(this.deps.backend, exported.surface, exported.rect, cropRect);
          const after = croppedLayerContract(before, cropRect);
          const afterPixels: LayerPixels = { pixels: cropped, rect: cropRect };
          /** Replaces the layer's contract and cache together; a failed replacement restores the current contract. */
          const replaceStep = (contract: CanvasLayerContract, snapshot: LayerPixels): EditStep =>
            replaceLayerStep(ctx, contract, ctx.preparePixels(layerId, snapshot.rect, snapshot.pixels), {
              beforeInstall: () => this.discardPersisted(layerId),
              restore: getDocumentLayer(ctx.getReducerDocument(), layerId) ?? undefined,
            });
          const result = txn.publish('Crop layer to frame', replaceStep(after, afterPixels), {
            bytes: rgbaBytes(beforePixels.rect) + rgbaBytes(cropRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
            heldAssetRefs: collectHistoryMediaRefs(before, after),
            redo: () =>
              withReplayReservation(ctx, rgbaBytes(cropRect), () => ctx.applyStep(replaceStep(after, afterPixels))),
            undo: () =>
              withReplayReservation(ctx, rgbaBytes(beforePixels.rect), () =>
                ctx.applyStep(replaceStep(before, beforePixels))
              ),
          });
          const status = layerOperationStatus(result);
          return status === 'committed' ? { status: 'cropped' } : status === 'failed' ? failedCrop : { status };
        } finally {
          txn.end();
        }
      } finally {
        exported.release();
      }
    } catch (error) {
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  }

  /**
   * Crops every leaf of every stack to the frame as one history entry. Locked leaves (their own lock or an ancestor's)
   * are skipped; leaves already inside the frame, empty ones and unsupported sources are left alone; a leaf entirely
   * outside the frame is emptied in place, so ids, order, groups and flags never change. Each leaf's export is
   * released as soon as its cropped pixels are copied, so the peak is the cropped set plus one export.
   */
  async cropAll(): Promise<CropAllLayersResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return { status: 'busy' };
    }
    const document = ctx.getDocument();
    if (!document) {
      return { skippedLocked: 0, status: 'nothing' };
    }
    const frame = roundOut(document.bbox);
    let skippedLocked = 0;
    const candidates: CanvasLayerContract[] = [];
    for (const leaf of compileDocumentLeaves(document)) {
      const source = renderableSourceOf(leaf.layer);
      if (!source || !this.deps.isSupportedSource(source)) {
        continue;
      }
      if (!leaf.effectiveLocked) {
        candidates.push(leaf.layer);
        continue;
      }
      // A locked leaf counts only when cropping would have changed it: content reaching past the frame.
      const content = getSourceContentRect(leaf.layer, document);
      if (!isEmpty(content) && cropRectFor(transformBounds(leaf.worldTransform, content), frame)) {
        skippedLocked += 1;
      }
    }
    const guards: LayerExportGuard[] = [];
    const cropped: CroppedLeaf[] = [];
    try {
      for (const layer of candidates) {
        const exported = await this.exportForCrop(layer, frame);
        if (exported.status === 'skip') {
          continue;
        }
        if (exported.status !== 'ok') {
          return exported;
        }
        try {
          guards.push(exported.guard);
          const rect = cropRectFor(exported.rect, frame);
          if (!rect) {
            continue;
          }
          const reservation = ctx.reserveRaster(rgbaBytes(rect));
          if (!reservation) {
            return { status: 'over-budget' };
          }
          let pixels: RasterSurface;
          try {
            pixels = cropPixels(this.deps.backend, exported.surface, exported.rect, rect);
          } catch (error) {
            reservation.release();
            throw error;
          }
          cropped.push({ layer, pixels: { pixels, rect }, reservation });
        } finally {
          exported.release();
        }
      }
      if (!ctx.isPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      const liveDocument = ctx.getDocument();
      // Any edit while the pixels were read (a layer, the frame, the stacks) makes the plan stale.
      if (
        !liveDocument ||
        liveDocument.stacks !== document.stacks ||
        !isSameRect(liveDocument.bbox, document.bbox) ||
        guards.some((guard) => !this.deps.isGuardCurrent(guard))
      ) {
        return { status: 'stale' };
      }
      if (cropped.length === 0) {
        return { skippedLocked, status: 'nothing' };
      }
      return this.publishAll(liveDocument, cropped, skippedLocked);
    } catch {
      return { status: 'failed' };
    } finally {
      for (const leaf of cropped) {
        leaf.reservation.release();
      }
    }
  }

  /**
   * A leaf's baked pixels for crop all, or `skip` when it has nothing to crop. A leaf with its own adjustment stack is
   * first exported raw: adjustments never move pixels, so the raw rect is the baked one, and a leaf already inside the
   * frame never runs its stack (a spatial job that could refuse the whole crop). Its guard still joins the staleness
   * check, as every read leaf's does.
   */
  private async exportForCrop(
    layer: CanvasLayerContract,
    frame: Rect
  ): Promise<Extract<ExportResult, { status: 'ok' }> | { status: 'skip' } | CropAllLayersResult> {
    const settle = (
      exported: ExportResult
    ): Extract<ExportResult, { status: 'ok' }> | { status: 'skip' } | CropAllLayersResult => {
      if (exported.status === 'ok' || exported.status === 'adjustment-refused') {
        return exported;
      }
      if (exported.status === 'missing') {
        return { status: 'stale' };
      }
      if (exported.status === 'not-ready' || exported.status === 'over-budget') {
        return { status: exported.status };
      }
      // Empty, disabled-only or unsupported content has nothing to crop.
      return { status: 'skip' };
    };
    if (layer.type !== 'raster' || isIdentityAdjustments(layer.adjustments)) {
      return settle(await this.deps.exportBaked(layer.id));
    }
    const raw = settle(await this.deps.exportBaked(layer.id, { applyAdjustments: false }));
    if (raw.status !== 'ok' || !cropRectFor(raw.rect, frame)) {
      return raw;
    }
    raw.release();
    return settle(await this.deps.exportBaked(layer.id));
  }

  private publishAll(
    document: CanvasDocumentContractV3,
    cropped: readonly CroppedLeaf[],
    skippedLocked: number
  ): CropAllLayersResult {
    const { ctx } = this.deps;
    const afterBytes = cropped.reduce((sum, leaf) => sum + rgbaBytes(leaf.pixels.rect), 0);
    const txn = ctx.begin({ historyBytes: afterBytes + HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      // The prepared cache replacements; the cropped pixels hold their own reservations.
      if (!txn.reserveRaster(afterBytes)) {
        return { status: 'over-budget' };
      }
      const crops: LayerCrop[] = [];
      for (const leaf of cropped) {
        // Each before snapshot grows the one admission; a set that cannot be kept refuses with nothing changed.
        const beforePixels = this.deps.captureCache(leaf.layer, document, (cacheRect) =>
          txn.growHistory(rgbaBytes(cacheRect))
        );
        if (beforePixels === 'over-budget') {
          return { status: 'over-budget' };
        }
        if (!beforePixels || beforePixels === 'not-ready') {
          return { status: 'not-ready' };
        }
        const before = structuredClone(leaf.layer);
        crops.push({
          after: croppedLayerContract(before, leaf.pixels.rect),
          afterPixels: leaf.pixels,
          before,
          beforePixels,
        });
      }
      const beforeBytes = crops.reduce((sum, crop) => sum + rgbaBytes(crop.beforePixels.rect), 0);
      const replay = (side: 'before' | 'after') => () =>
        withReplayReservation(ctx, side === 'before' ? beforeBytes : afterBytes, () =>
          ctx.applyStep(this.replaceAllStep(crops, side))
        );
      const result = txn.publish('Crop all layers to frame', this.replaceAllStep(crops, 'after'), {
        bytes: beforeBytes + afterBytes + HISTORY_ENTRY_OVERHEAD_BYTES,
        heldAssetRefs: collectHistoryMediaRefs(...crops.flatMap((crop) => [crop.before, crop.after])),
        redo: replay('after'),
        undo: replay('before'),
      });
      const status = layerOperationStatus(result);
      return status === 'committed' ? { cropped: crops.length, skippedLocked, status: 'cropped' } : { status };
    } finally {
      txn.end();
    }
  }

  /** Every cropped leaf's contract and cache, replaced together as one step; it rolls back to the reducer's contracts. */
  private replaceAllStep(crops: readonly LayerCrop[], side: 'before' | 'after'): EditStep {
    const { ctx } = this.deps;
    const reducerDocument = ctx.getReducerDocument();
    return replaceLayersStep(
      ctx,
      crops.map((crop) => {
        const pixels = side === 'before' ? crop.beforePixels : crop.afterPixels;
        return { contract: crop[side], prepared: ctx.preparePixels(crop.before.id, pixels.rect, pixels.pixels) };
      }),
      {
        beforeInstall: (layerId) => this.discardPersisted(layerId),
        restore: crops.flatMap((crop) => getDocumentLayer(reducerDocument, crop.before.id) ?? []),
      }
    );
  }

  /** Drops a replaced layer's pending persistence; the installed pixels are marked dirty and persist on their own. */
  private discardPersisted(layerId: string): void {
    try {
      this.deps.discardPersisted(layerId);
    } catch {
      // The installed pixels persist on their own.
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}

const isSameRect = (a: Rect, b: Rect): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/**
 * Where a leaf's exported pixels land once cropped to `frame`: their overlap, an empty rect at the frame origin when
 * they lie entirely outside it, or null when they already fit inside and there is nothing to crop.
 */
const cropRectFor = (exportedRect: Rect, frame: Rect): Rect | null => {
  const inside =
    exportedRect.x >= frame.x &&
    exportedRect.y >= frame.y &&
    exportedRect.x + exportedRect.width <= frame.x + frame.width &&
    exportedRect.y + exportedRect.height <= frame.y + frame.height;
  if (inside) {
    return null;
  }
  const overlap = intersect(exportedRect, frame);
  return overlap && !isEmpty(overlap) ? roundOut(overlap) : { height: 0, width: 0, x: frame.x, y: frame.y };
};
