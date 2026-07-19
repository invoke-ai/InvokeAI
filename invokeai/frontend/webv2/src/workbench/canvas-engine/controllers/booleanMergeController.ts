import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { isMergeableRasterLayer } from '@workbench/canvas-engine/document/sources';
import { isEmpty, roundOut, union } from '@workbench/canvas-engine/math/rect';

export type BooleanRasterOperation = 'intersect' | 'cutout' | 'cutaway' | 'exclude';
export type BooleanRasterResult = 'merged' | 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty';

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export interface BooleanMergeControllerOptions {
  readonly backend: RasterBackend;
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly capturePermit: () => object | null;
  readonly isPermitCurrent: (permit: object) => boolean;
  readonly isGestureActive: () => boolean;
  readonly endBurst: () => void;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly createLayerId: () => string;
  readonly dispatchPrepared: (
    action: CanvasProjectMutation,
    expectedReducer: () => boolean,
    expectedMirror: () => boolean
  ) => void;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => unknown;
  readonly installPrepared: (prepared: unknown) => void;
}

const modes: Record<BooleanRasterOperation, GlobalCompositeOperation> = {
  cutaway: 'source-out',
  cutout: 'destination-in',
  exclude: 'xor',
  intersect: 'source-in',
};

/** Owns guarded two-layer boolean compositing and atomic stack history. */
export class BooleanMergeController {
  private disposed = false;

  constructor(private readonly deps: BooleanMergeControllerOptions) {}

  async merge(upperLayerId: string, operation: BooleanRasterOperation): Promise<BooleanRasterResult> {
    const permit = this.deps.capturePermit();
    if (this.disposed || !permit || this.deps.isGestureActive()) {
      return 'busy';
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
    if (!document) {
      return 'missing';
    }
    const upperIndex = document.layers.findIndex((layer) => layer.id === upperLayerId);
    const upper = document.layers[upperIndex];
    const below = document.layers[upperIndex + 1];
    if (upperIndex < 0 || !upper || !below) {
      return 'missing';
    }
    if (!isMergeableRasterLayer(upper) || !isMergeableRasterLayer(below)) {
      return 'unsupported';
    }
    if (!this.deps.isCacheReady(upper, document) || !this.deps.isCacheReady(below, document)) {
      return 'not-ready';
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
      const settled = await Promise.allSettled([acquire(upper.id), acquire(below.id)]);
      const rejected = settled.find((result) => result.status === 'rejected');
      if (rejected?.status === 'rejected') {
        throw rejected.reason instanceof Error ? rejected.reason : new Error(String(rejected.reason));
      }
      const [upperPixels, belowPixels] = settled.map(
        (result) => (result as PromiseFulfilledResult<ExportResult>).value
      );
      if (!this.deps.isPermitCurrent(permit)) {
        return 'busy';
      }
      if (upperPixels.status !== 'ok' || belowPixels.status !== 'ok') {
        if (upperPixels.status === 'not-ready' || belowPixels.status === 'not-ready') {
          return 'not-ready';
        }
        if (
          upperPixels.status === 'disabled' ||
          upperPixels.status === 'unsupported' ||
          belowPixels.status === 'disabled' ||
          belowPixels.status === 'unsupported'
        ) {
          return 'unsupported';
        }
        return 'empty';
      }
      if (
        !this.deps.isPermitCurrent(permit) ||
        this.deps.isGestureActive() ||
        upperPixels.guard.layer !== upper ||
        belowPixels.guard.layer !== below ||
        !this.deps.isGuardCurrent(upperPixels.guard) ||
        !this.deps.isGuardCurrent(belowPixels.guard)
      ) {
        return this.deps.isPermitCurrent(permit) ? 'not-ready' : 'busy';
      }
      const liveDocument = this.deps.getDocument();
      const liveIndex = liveDocument?.layers.findIndex((layer) => layer.id === upperLayerId) ?? -1;
      if (!liveDocument || liveDocument.layers[liveIndex] !== upper || liveDocument.layers[liveIndex + 1] !== below) {
        return 'not-ready';
      }
      const resultRect = roundOut(union(upperPixels.rect, belowPixels.rect));
      if (isEmpty(resultRect)) {
        return 'empty';
      }
      const pixels = this.deps.backend.createSurface(resultRect.width, resultRect.height);
      pixels.ctx.setTransform(1, 0, 0, 1, 0, 0);
      pixels.ctx.clearRect(0, 0, resultRect.width, resultRect.height);
      pixels.ctx.globalAlpha = below.opacity;
      pixels.ctx.globalCompositeOperation = 'source-over';
      pixels.ctx.drawImage(
        belowPixels.surface.canvas,
        belowPixels.rect.x - resultRect.x,
        belowPixels.rect.y - resultRect.y
      );
      pixels.ctx.globalAlpha = upper.opacity;
      pixels.ctx.globalCompositeOperation = modes[operation];
      pixels.ctx.drawImage(
        upperPixels.surface.canvas,
        upperPixels.rect.x - resultRect.x,
        upperPixels.rect.y - resultRect.y
      );
      const resultId = this.deps.createLayerId();
      const resultLayer: CanvasLayerContract = {
        blendMode: 'normal',
        id: resultId,
        isEnabled: true,
        isLocked: false,
        name: `${upper.name} ${operation}`,
        opacity: 1,
        source: { bitmap: null, offset: { x: resultRect.x, y: resultRect.y }, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'raster',
      };
      const original = [
        { id: upper.id, isEnabled: upper.isEnabled },
        { id: below.id, isEnabled: below.isEnabled },
      ];
      const disabled = original.map(({ id }) => ({ id, isEnabled: false }));
      const selectedLayerId = liveDocument.selectedLayerId;
      const hasState = (doc: CanvasDocumentContractV2 | null, updates: typeof original): boolean =>
        updates.every((update) => doc?.layers.find((layer) => layer.id === update.id)?.isEnabled === update.isEnabled);
      const apply = (): void => {
        const prepared = this.deps.preparePixels(resultId, resultRect, pixels);
        this.deps.dispatchPrepared(
          {
            add: { index: liveIndex, layers: [resultLayer] },
            enabledUpdates: disabled,
            selectedLayerId: resultId,
            type: 'applyCanvasLayerStackMutation',
          },
          () =>
            this.deps.getReducerDocument()?.selectedLayerId === resultId &&
            hasState(this.deps.getReducerDocument(), disabled),
          () => this.deps.getDocument()?.selectedLayerId === resultId && hasState(this.deps.getDocument(), disabled)
        );
        this.deps.installPrepared(prepared);
      };
      if (!this.deps.isPermitCurrent(permit)) {
        return 'busy';
      }
      apply();
      this.deps.history.push({
        bytes: resultRect.width * resultRect.height * 4 + 256,
        label: `Boolean ${operation}`,
        redo: apply,
        replayFailureAtomic: true,
        undo: () =>
          this.deps.dispatchPrepared(
            { enabledUpdates: original, removeIds: [resultId], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
            () =>
              this.deps.getReducerDocument()?.selectedLayerId === selectedLayerId &&
              hasState(this.deps.getReducerDocument(), original),
            () =>
              this.deps.getDocument()?.selectedLayerId === selectedLayerId &&
              hasState(this.deps.getDocument(), original)
          ),
      });
      return 'merged';
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
