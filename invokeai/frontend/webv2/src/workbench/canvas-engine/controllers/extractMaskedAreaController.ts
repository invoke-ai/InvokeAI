import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { isEmpty } from '@workbench/canvas-engine/math/rect';
import {
  type AdjustmentRefusal,
  AdjustmentRefusedError,
  adjustmentRefusal,
  type SpatialRunner,
} from '@workbench/canvas-engine/render/exactAdjustments';
import {
  collectCompositedGroups,
  planGroupCompositeScopes,
} from '@workbench/canvas-engine/render/groupCompositeScopes';
import {
  compositeScratchPixels,
  planRasterComposite,
  RasterCompositeOverBudgetError,
  renderRasterComposite,
} from '@workbench/canvas-engine/render/rasterComposite';

import type { CanvasMutationContext } from './mutationContext';

import {
  addedLayerHistoryBytes,
  layerEditRefusal,
  layerOperationStatus,
  paintLayerAt,
  publishAddedRasterLayer,
  readinessOfAll,
  rgbaBytes,
  type LayerCacheReadiness,
  type LayerStepContext,
  type LayerEditRefusal,
} from './editSteps';

export type ExtractMaskedAreaResult =
  | { status: 'extracted'; layerId: string }
  | {
      status:
        | LayerEditRefusal
        | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'>
        | 'empty'
        | 'unreadable'
        | 'failed';
    }
  | AdjustmentRefusal;

type OwnedExport = Extract<ExportResult, { status: 'ok' }>;

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'unreadable' | 'over-budget' }
  | AdjustmentRefusal;

export interface ExtractMaskedAreaControllerOptions {
  readonly ctx: LayerStepContext &
    Pick<
      CanvasMutationContext,
      | 'begin'
      | 'capturePermit'
      | 'captureInsertionAnchor'
      | 'createLayerId'
      | 'getDocument'
      | 'getReducerDocument'
      | 'isPermitCurrent'
    >;
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly cacheReadiness: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => LayerCacheReadiness;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string, includeDisabled?: boolean) => Promise<ExportResult>;
  readonly rasterize: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  /** The display's exact adjusted copy of a layer's live cache `surface`, or null; never builds one. */
  readonly peekAdjustedCacheSurface?: (
    layerId: string,
    surface: RasterSurface,
    adjustments: CanvasAdjustmentsContract
  ) => RasterSurface | null;
  /** Evaluates spatial adjustment runs; without it a spatial stack refuses `error`. */
  readonly runSpatial?: SpatialRunner;
}

/** The group compositing `layers` pass through (ancestor opacity, blend and adjustments), for comparison. */
const groupScopesKey = (document: CanvasDocumentContractV3, layers: readonly CanvasLayerContract[]): string => {
  const ids = new Set(layers.map((layer) => layer.id));
  const leaves = compileDocumentLeaves(document).filter((leaf) => ids.has(leaf.id));
  return JSON.stringify(planGroupCompositeScopes(leaves, collectCompositedGroups(document)));
};

type RasterLeaf = SemanticLeaf & { layer: CanvasRasterLayerContractV2 };

const contributingLeaves = (document: CanvasDocumentContractV3, hasContent: (layerId: string) => boolean) =>
  compileDocumentLeaves(document).filter(
    (leaf): leaf is RasterLeaf =>
      leaf.stack === 'raster' && leaf.layer.type === 'raster' && leaf.contributionEnabled && hasContent(leaf.id)
  );

const contributingRasters = (
  document: CanvasDocumentContractV3,
  hasContent: (layerId: string) => boolean
): CanvasLayerContract[] => contributingLeaves(document, hasContent).map((leaf) => leaf.layer);

/** Owns guarded extraction of raster content through an inpaint mask. */
export class ExtractMaskedAreaController {
  private disposed = false;
  constructor(private readonly deps: ExtractMaskedAreaControllerOptions) {}

  async extract(maskLayerId: string): Promise<ExtractMaskedAreaResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return { status: 'busy' };
    }
    const document = ctx.getDocument();
    if (!document) {
      return { status: 'missing' };
    }
    const maskIndex = getDocumentLeaves(document).findIndex((layer) => layer.id === maskLayerId);
    const mask = getDocumentLeaves(document)[maskIndex];
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
    const contributors = contributingRasters(document, this.deps.hasExportableContent);
    if (contributors.length === 0) {
      return { status: 'empty' };
    }
    const readiness = readinessOfAll([mask, ...contributors].map((layer) => this.deps.cacheReadiness(layer, document)));
    if (readiness !== 'ready') {
      return { status: readiness };
    }
    const owned: Extract<ExportResult, { status: 'ok' }>[] = [];
    const acquire = async (resultPromise: Promise<ExportResult>): Promise<ExportResult> => {
      const result = await resultPromise;
      if (result.status === 'ok') {
        owned.push(result);
      }
      return result;
    };
    try {
      const settled = await Promise.allSettled([
        acquire(this.deps.exportBaked(maskLayerId, true)),
        ...contributors.map((layer) => acquire(this.deps.rasterize(layer.id))),
      ]);
      const rejected = settled.find((result) => result.status === 'rejected');
      if (rejected?.status === 'rejected') {
        throw rejected.reason instanceof Error ? rejected.reason : new Error(String(rejected.reason));
      }
      const [maskPixels, ...contributorPixels] = settled.map(
        (result) => (result as PromiseFulfilledResult<ExportResult>).value
      );
      if (!ctx.isPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      const statuses = [maskPixels, ...contributorPixels].map((pixels) => pixels.status);
      for (const status of ['over-budget', 'unreadable', 'not-ready'] as const) {
        if (statuses.includes(status)) {
          return { status };
        }
      }
      if (maskPixels.status !== 'ok' || statuses.some((status) => status !== 'ok')) {
        return { status: 'empty' };
      }
      if (maskPixels.guard.layer !== mask || !this.deps.isGuardCurrent(maskPixels.guard)) {
        return { status: 'not-ready' };
      }
      // A preview-tolerant recheck; see `CanvasMutationContext.getReducerDocument`.
      const liveDocument = ctx.getReducerDocument();
      const liveMaskIndex =
        getDocumentLeaves(liveDocument ?? null).findIndex((layer) => layer.id === maskLayerId) ?? -1;
      const currentMask = getDocumentLeaves(liveDocument ?? null)[liveMaskIndex];
      if (!liveDocument || !currentMask) {
        return { status: 'missing' };
      }
      if (currentMask !== mask) {
        return { status: currentMask.type === 'inpaint_mask' && currentMask.isLocked ? 'unsupported' : 'not-ready' };
      }
      const liveContributors = contributingRasters(liveDocument, this.deps.hasExportableContent);
      if (
        liveMaskIndex !== maskIndex ||
        liveContributors.some((layer, index) => layer !== contributors[index]) ||
        liveContributors.length !== contributors.length ||
        // The composite below reads the pre-await groups around the contributors.
        groupScopesKey(liveDocument, liveContributors) !== groupScopesKey(document, contributors)
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
      const txn = ctx.begin({ historyBytes: addedLayerHistoryBytes(rect) });
      if (!('publish' in txn)) {
        return { status: layerEditRefusal(txn.status) };
      }
      try {
        // The contributors composite as generation and export do: exact adjustments (spatial ones on whole sources,
        // never a display interim), from the leased pixels.
        const leased = new Map(
          contributorPixels.map((leasedPixels, index) => [contributors[index]!.id, leasedPixels as OwnedExport])
        );
        const entry = planRasterComposite(
          document,
          contributingLeaves(document, this.deps.hasExportableContent),
          rect,
          new Map([...leased].map(([id, owned]) => [id, owned.rect]))
        );
        // The output plus each scratch buffer and its ImageData, until the transaction ends.
        if (!txn.reserveRaster(rgbaBytes(rect) + compositeScratchPixels(entry) * 2 * 4)) {
          return { status: 'over-budget' };
        }
        let pixels: RasterSurface;
        try {
          pixels = await renderRasterComposite(entry, {
            adjustedSurface: this.deps.peekAdjustedCacheSurface,
            backend: this.deps.backend,
            getLayerSurface: (layerId) => {
              const owned = leased.get(layerId);
              return owned
                ? Promise.resolve({ rect: owned.rect, release: () => undefined, surface: owned.surface })
                : Promise.reject(new Error(`Extraction did not lease layer ${layerId}.`));
            },
            reserve: (bytes) =>
              txn.reserveRaster(bytes)
                ? { lease: { release: () => undefined }, status: 'ok' }
                : { availableBytes: 0, requestedBytes: bytes, status: 'over-budget' },
            runSpatial: this.deps.runSpatial,
          });
        } catch (error) {
          if (error instanceof AdjustmentRefusedError) {
            return adjustmentRefusal(document, error.ownerId, error.reason);
          }
          if (error instanceof RasterCompositeOverBudgetError) {
            return { status: 'over-budget' };
          }
          if (error instanceof DOMException && error.name === 'AbortError') {
            return { status: 'not-ready' };
          }
          throw error;
        }
        // Spatial jobs yield: the extraction stands only for pixels that still describe their layers.
        if (
          this.disposed ||
          !this.deps.isGuardCurrent(maskPixels.guard) ||
          [...leased.values()].some((owned) => !this.deps.isGuardCurrent(owned.guard))
        ) {
          return { status: 'not-ready' };
        }
        pixels.ctx.setTransform(1, 0, 0, 1, 0, 0);
        pixels.ctx.globalAlpha = 1;
        pixels.ctx.globalCompositeOperation = 'destination-in';
        pixels.ctx.drawImage(maskPixels.surface.canvas, 0, 0);
        const layer = paintLayerAt(ctx.createLayerId(), `${mask.name} extraction`, rect);
        const status = layerOperationStatus(
          publishAddedRasterLayer(ctx, txn, {
            anchor: ctx.captureInsertionAnchor('raster', null),
            label: 'Extract masked area',
            layer,
            pixels,
            rect,
            selectedLayerId: liveDocument.selectedLayerId,
          })
        );
        return status === 'committed' ? { layerId: layer.id, status: 'extracted' } : { status };
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
