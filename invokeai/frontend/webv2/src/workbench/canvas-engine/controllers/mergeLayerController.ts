import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { History } from '@workbench/canvas-engine/history/history';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { mergeDownMatrix } from '@workbench/canvas-engine/document/mergeDown';
import { getMergeVisibleRasterLayers } from '@workbench/canvas-engine/document/mergeVisible';
import { isMergeableRasterLayer } from '@workbench/canvas-engine/document/sources';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

export type MergeVisibleResult = 'merged' | 'not-ready' | 'busy' | 'nothing';

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

export interface MergeLayerControllerOptions {
  readonly backend: RasterBackend;
  readonly history: History;
  readonly layers: LayerCacheStore;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly canEdit: () => boolean;
  readonly capturePermit: () => object | null;
  readonly isPermitCurrent: (permit: object) => boolean;
  readonly isGestureActive: () => boolean;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => boolean;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly createLayerId: () => string;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => unknown;
  readonly installPrepared: (prepared: unknown) => void;
  readonly dispatchPrepared: (
    action: WorkbenchAction,
    expectedReducer: () => boolean,
    expectedMirror: () => boolean
  ) => void;
  readonly endBurst: () => void;
  readonly dispatch: (action: WorkbenchAction) => void;
  readonly notifyPainted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
}

/** Owns destructive merge-down and non-destructive merge-visible pixel operations. */
export class MergeLayerController {
  private disposed = false;

  constructor(private readonly deps: MergeLayerControllerOptions) {}

  mergeDown(upperLayerId: string): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return false;
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
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
    if (
      !upperCache ||
      !belowCache ||
      !this.deps.isCacheReady(upper, document) ||
      !this.deps.isCacheReady(below, document)
    ) {
      return false;
    }
    const matrix = mergeDownMatrix(below.transform, upper.transform);
    if (!matrix) {
      return false;
    }
    if (isEmpty(belowCache.rect) && isEmpty(upperCache.rect)) {
      this.deps.dispatch({
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
        type: 'mergeCanvasLayersDown',
        upperLayerId,
      });
      this.deps.layers.delete(below.id);
      this.deps.notifyPainted(below.id);
      this.deps.markDirty(below.id);
      return true;
    }
    const mergedRect = roundOut(union(belowCache.rect, transformBounds(matrix, upperCache.rect)));
    const merged = this.deps.backend.createSurface(mergedRect.width, mergedRect.height);
    const context = merged.ctx;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, mergedRect.width, mergedRect.height);
    if (!isEmpty(belowCache.rect)) {
      context.drawImage(belowCache.surface.canvas, belowCache.rect.x - mergedRect.x, belowCache.rect.y - mergedRect.y);
    }
    if (!isEmpty(upperCache.rect)) {
      context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - mergedRect.x, matrix.f - mergedRect.y);
      context.globalAlpha = upper.opacity;
      context.globalCompositeOperation = blendToComposite(upper.blendMode);
      context.drawImage(upperCache.surface.canvas, upperCache.rect.x, upperCache.rect.y);
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.globalAlpha = 1;
      context.globalCompositeOperation = 'source-over';
    }
    this.deps.dispatch({
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
    const permit = this.deps.capturePermit();
    if (this.disposed || !permit || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return 'busy';
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
    if (!document) {
      return 'nothing';
    }
    const contributors = getMergeVisibleRasterLayers(document.layers, this.deps.hasExportableContent);
    if (contributors.length < 2) {
      return 'nothing';
    }
    const exports = await Promise.all(contributors.map((layer) => this.deps.exportBaked(layer.id)));
    if (!this.deps.isPermitCurrent(permit)) {
      return 'busy';
    }
    if (exports.some((result) => result.status !== 'ok')) {
      return 'not-ready';
    }
    if (this.deps.isGestureActive()) {
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
        !this.deps.isGuardCurrent(exported.guard)
      ) {
        return 'not-ready';
      }
    }

    const liveDocument = this.deps.getDocument();
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

    const resultId = this.deps.createLayerId();
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
      const prepared = this.deps.preparePixels(resultId, rect, pixels);
      this.deps.dispatchPrepared(
        {
          add: { index: 0, layers: [resultLayer] },
          enabledUpdates: [],
          selectedLayerId: resultId,
          type: 'applyCanvasLayerStackMutation',
        },
        () => hasResult(this.deps.getReducerDocument()),
        () => hasResult(this.deps.getDocument())
      );
      this.deps.installPrepared(prepared);
    };
    if (!this.deps.isPermitCurrent(permit)) {
      return 'busy';
    }
    apply();
    this.deps.history.push({
      bytes: rect.width * rect.height * 4 + 256,
      label: 'Merge visible',
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        this.deps.dispatchPrepared(
          { enabledUpdates: [], removeIds: [resultId], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
          () =>
            this.deps.getReducerDocument()?.selectedLayerId === selectedLayerId &&
            this.deps.getReducerDocument()?.layers.some((layer) => layer.id === resultId) === false,
          () =>
            this.deps.getDocument()?.selectedLayerId === selectedLayerId &&
            this.deps.getDocument()?.layers.some((layer) => layer.id === resultId) === false
        ),
    });
    return 'merged';
  }

  dispose(): void {
    this.disposed = true;
  }
}
