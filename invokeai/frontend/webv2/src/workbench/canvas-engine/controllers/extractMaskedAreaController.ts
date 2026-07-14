import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { History } from '@workbench/canvas-engine/history/history';
import type { DerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';

export type ExtractMaskedAreaResult =
  | { status: 'extracted'; layerId: string }
  | { status: 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty' };

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

export interface ExtractMaskedAreaControllerOptions {
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly derived: DerivedSurfaceCache;
  readonly diagnostics: CanvasDiagnostics;
  readonly history: History;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly capturePermit: () => object | null;
  readonly isPermitCurrent: (permit: object) => boolean;
  readonly isGestureActive: () => boolean;
  readonly endBurst: () => void;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => boolean;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string, includeDisabled?: boolean) => Promise<ExportResult>;
  readonly rasterize: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly getAdjustedSurface: (layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null;
  readonly getMaskPattern: (style: string, color: string) => RasterSurface | null;
  readonly createLayerId: () => string;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => unknown;
  readonly installPrepared: (prepared: unknown) => void;
  readonly dispatchPrepared: (
    action: WorkbenchAction,
    expectedReducer: () => boolean,
    expectedMirror: () => boolean
  ) => void;
}

/** Owns guarded extraction of raster content through an inpaint mask. */
export class ExtractMaskedAreaController {
  private disposed = false;
  constructor(private readonly deps: ExtractMaskedAreaControllerOptions) {}

  async extract(maskLayerId: string): Promise<ExtractMaskedAreaResult> {
    const permit = this.deps.capturePermit();
    if (this.disposed || !permit || this.deps.isGestureActive()) {
      return { status: 'busy' };
    }
    this.deps.endBurst();
    const document = this.deps.getDocument();
    if (!document) {
      return { status: 'missing' };
    }
    const maskIndex = document.layers.findIndex((layer) => layer.id === maskLayerId);
    const mask = document.layers[maskIndex];
    if (maskIndex < 0 || !mask) {
      return { status: 'missing' };
    }
    if (mask.type !== 'inpaint_mask' || mask.isLocked) {
      return { status: 'unsupported' };
    }
    const liveMask = this.deps.layers.get(maskLayerId);
    if (isEmpty(getSourceContentRect(mask, document)) && (!liveMask || isEmpty(liveMask.rect))) {
      return { status: 'empty' };
    }
    const contributors = document.layers.filter(
      (layer) => layer.isEnabled && layer.type === 'raster' && this.deps.hasExportableContent(layer.id)
    );
    if (contributors.length === 0) {
      return { status: 'empty' };
    }
    if (
      !this.deps.isCacheReady(mask, document) ||
      contributors.some((layer) => !this.deps.isCacheReady(layer, document))
    ) {
      return { status: 'not-ready' };
    }
    const [maskPixels, contributorPixels] = await Promise.all([
      this.deps.exportBaked(maskLayerId, true),
      Promise.all(contributors.map((layer) => this.deps.rasterize(layer.id))),
    ]);
    if (!this.deps.isPermitCurrent(permit)) {
      return { status: 'busy' };
    }
    if (maskPixels.status !== 'ok') {
      return { status: maskPixels.status === 'not-ready' ? 'not-ready' : 'empty' };
    }
    if (contributorPixels.some((pixels) => pixels.status !== 'ok')) {
      return { status: contributorPixels.some((pixels) => pixels.status === 'not-ready') ? 'not-ready' : 'empty' };
    }
    if (this.deps.isGestureActive()) {
      return { status: 'busy' };
    }
    if (maskPixels.guard.layer !== mask || !this.deps.isGuardCurrent(maskPixels.guard)) {
      return { status: 'not-ready' };
    }
    const liveDocument = this.deps.getDocument();
    const liveMaskIndex = liveDocument?.layers.findIndex((layer) => layer.id === maskLayerId) ?? -1;
    const currentMask = liveDocument?.layers[liveMaskIndex];
    if (!liveDocument || !currentMask) {
      return { status: 'missing' };
    }
    if (currentMask !== mask) {
      return { status: currentMask.type === 'inpaint_mask' && currentMask.isLocked ? 'unsupported' : 'not-ready' };
    }
    const liveContributors = liveDocument.layers.filter(
      (layer) => layer.isEnabled && layer.type === 'raster' && this.deps.hasExportableContent(layer.id)
    );
    if (
      liveMaskIndex !== maskIndex ||
      liveContributors.some((layer, index) => layer !== contributors[index]) ||
      liveContributors.length !== contributors.length
    ) {
      return { status: 'not-ready' };
    }
    for (let index = 0; index < contributorPixels.length; index += 1) {
      const pixels = contributorPixels[index];
      const contributor = contributors[index];
      if (
        !pixels ||
        pixels.status !== 'ok' ||
        !contributor ||
        pixels.guard.layer !== contributor ||
        !this.deps.isGuardCurrent(pixels.guard)
      ) {
        return { status: 'not-ready' };
      }
    }
    const rect = maskPixels.rect;
    if (isEmpty(rect)) {
      return { status: 'empty' };
    }
    const pixels = this.deps.backend.createSurface(rect.width, rect.height);
    compositeDocument(
      pixels,
      { ...document, layers: contributors },
      this.deps.layers,
      { a: 1, b: 0, c: 0, d: 1, e: -rect.x, f: -rect.y },
      {
        adjustedSurface: this.deps.getAdjustedSurface,
        backend: this.deps.backend,
        derivedSurfaces: this.deps.derived,
        diagnostics: this.deps.diagnostics,
        maskPatternTile: this.deps.getMaskPattern,
      }
    );
    pixels.ctx.setTransform(1, 0, 0, 1, 0, 0);
    pixels.ctx.globalAlpha = 1;
    pixels.ctx.globalCompositeOperation = 'destination-in';
    pixels.ctx.drawImage(maskPixels.surface.canvas, 0, 0);
    const resultId = this.deps.createLayerId();
    const layer: CanvasLayerContract = {
      blendMode: 'normal',
      id: resultId,
      isEnabled: true,
      isLocked: false,
      name: `${mask.name} extraction`,
      opacity: 1,
      source: { bitmap: null, offset: { x: rect.x, y: rect.y }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const selectedLayerId = liveDocument.selectedLayerId;
    const apply = (): void => {
      const prepared = this.deps.preparePixels(resultId, rect, pixels);
      this.deps.dispatchPrepared(
        {
          add: { index: maskIndex, layer },
          enabledUpdates: [],
          selectedLayerId: resultId,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          this.deps.getReducerDocument()?.selectedLayerId === resultId &&
          this.deps.getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          this.deps.getDocument()?.selectedLayerId === resultId &&
          this.deps.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      this.deps.installPrepared(prepared);
    };
    if (!this.deps.isPermitCurrent(permit)) {
      return { status: 'busy' };
    }
    apply();
    this.deps.history.push({
      bytes: rect.width * rect.height * 4 + 256,
      label: 'Extract masked area',
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        this.deps.dispatchPrepared(
          { enabledUpdates: [], removeIds: [resultId], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
          () =>
            this.deps.getReducerDocument()?.selectedLayerId === selectedLayerId &&
            this.deps.getReducerDocument()?.layers.some((candidate) => candidate.id === resultId) === false,
          () =>
            this.deps.getDocument()?.selectedLayerId === selectedLayerId &&
            this.deps.getDocument()?.layers.some((candidate) => candidate.id === resultId) === false
        ),
    });
    return { layerId: resultId, status: 'extracted' };
  }

  dispose(): void {
    this.disposed = true;
  }
}
