import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { History } from '@workbench/canvas-engine/history/history';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

export interface CopyLayerControllerOptions {
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly capturePermit: () => object | null;
  readonly isPermitCurrent: (permit: object) => boolean;
  readonly isGestureActive: () => boolean;
  readonly endBurst: () => void;
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
}

/** Owns guarded baked copies into new raster paint layers. */
export class CopyLayerController {
  private disposed = false;
  constructor(private readonly deps: CopyLayerControllerOptions) {}

  async copyToRaster(layerId: string): Promise<string | null> {
    const permit = this.deps.capturePermit();
    if (this.disposed || !permit || this.deps.isGestureActive()) {
      return null;
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
    const sourceLayer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !sourceLayer) {
      return null;
    }
    const baked = await this.deps.exportBaked(layerId);
    if (!this.deps.isPermitCurrent(permit) || baked.status !== 'ok') {
      return null;
    }
    if (this.deps.isGestureActive() || !this.deps.isGuardCurrent(baked.guard) || baked.guard.layer !== sourceLayer) {
      return null;
    }
    const liveDocument = this.deps.getDocument();
    const sourceIndex = liveDocument?.layers.findIndex((layer) => layer.id === layerId) ?? -1;
    if (!liveDocument || liveDocument.layers[sourceIndex] !== sourceLayer || sourceIndex < 0) {
      return null;
    }
    const newId = this.deps.createLayerId();
    const layer: CanvasLayerContract = {
      blendMode: 'normal',
      id: newId,
      isEnabled: true,
      isLocked: false,
      name: `${sourceLayer.name} copy`,
      opacity: 1,
      source: { bitmap: null, offset: { x: baked.rect.x, y: baked.rect.y }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const selectedLayerId = liveDocument.selectedLayerId;
    const apply = (): void => {
      const prepared = this.deps.preparePixels(newId, baked.rect, baked.surface);
      this.deps.dispatchPrepared(
        {
          add: { index: sourceIndex, layers: [layer] },
          enabledUpdates: [],
          selectedLayerId: newId,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          this.deps.getReducerDocument()?.selectedLayerId === newId &&
          this.deps.getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          this.deps.getDocument()?.selectedLayerId === newId &&
          this.deps.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      this.deps.installPrepared(prepared);
    };
    if (!this.deps.isPermitCurrent(permit)) {
      return null;
    }
    apply();
    this.deps.history.push({
      bytes: baked.rect.width * baked.rect.height * 4 + 256,
      label: 'Copy layer to raster',
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        this.deps.dispatchPrepared(
          { enabledUpdates: [], removeIds: [newId], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
          () =>
            this.deps.getReducerDocument()?.selectedLayerId === selectedLayerId &&
            this.deps.getReducerDocument()?.layers.some((candidate) => candidate.id === newId) === false,
          () =>
            this.deps.getDocument()?.selectedLayerId === selectedLayerId &&
            this.deps.getDocument()?.layers.some((candidate) => candidate.id === newId) === false
        ),
    });
    return newId;
  }

  dispose(): void {
    this.disposed = true;
  }
}
