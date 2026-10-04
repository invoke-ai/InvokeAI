import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { CompositeOptions } from '@workbench/canvas-engine/render/compositor';
import type { DerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createEmptyStacks, isGroupNode } from '@workbench/canvas-engine/document/documentTree';
import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';

import type { CanvasMutationContext } from './mutationContext';

import {
  addedLayerHistoryBytes,
  layerEditRefusal,
  layerOperationStatus,
  paintLayerAt,
  publishAddedRasterLayer,
  rgbaBytes,
  type LayerStepContext,
  type LayerEditRefusal,
} from './editSteps';

export type ExtractMaskedAreaResult =
  | { status: 'extracted'; layerId: string }
  | { status: LayerEditRefusal | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'> | 'empty' | 'failed' };

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export interface ExtractMaskedAreaControllerOptions {
  readonly ctx: LayerStepContext &
    Pick<
      CanvasMutationContext,
      'begin' | 'capturePermit' | 'captureInsertionAnchor' | 'createLayerId' | 'getDocument' | 'isPermitCurrent'
    >;
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly derived: DerivedSurfaceCache;
  readonly diagnostics: CanvasDiagnostics;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly hasExportableContent: (layerId: string) => boolean;
  readonly exportBaked: (layerId: string, includeDisabled?: boolean) => Promise<ExportResult>;
  readonly rasterize: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly getAdjustedSurface: (layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null;
  readonly getGroupSurface?: CompositeOptions['groupSurface'];
  readonly getMaskPattern: (style: string, color: string) => RasterSurface | null;
}

/** The raster forest reduced to `keep` leaves, groups intact so their stacks still apply. */
const filterRasterForest = (nodes: readonly CanvasNodeContract[], keep: ReadonlySet<string>): CanvasNodeContract[] =>
  nodes.flatMap((node): CanvasNodeContract[] => {
    if (isGroupNode(node)) {
      const children = filterRasterForest(node.children, keep);
      return children.length > 0 ? [{ ...node, children }] : [];
    }
    return keep.has(node.id) ? [node] : [];
  });

const contributingRasters = (
  document: CanvasDocumentContractV3,
  hasContent: (layerId: string) => boolean
): CanvasLayerContract[] =>
  compileDocumentLeaves(document)
    .filter((leaf) => leaf.stack === 'raster' && leaf.contributionEnabled && hasContent(leaf.id))
    .map((leaf) => leaf.layer);

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
    if (
      !this.deps.isCacheReady(mask, document) ||
      contributors.some((layer) => !this.deps.isCacheReady(layer, document))
    ) {
      return { status: 'not-ready' };
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
      if (statuses.includes('over-budget') || statuses.includes('not-ready')) {
        return { status: statuses.includes('over-budget') ? 'over-budget' : 'not-ready' };
      }
      if (maskPixels.status !== 'ok' || statuses.some((status) => status !== 'ok')) {
        return { status: 'empty' };
      }
      if (maskPixels.guard.layer !== mask || !this.deps.isGuardCurrent(maskPixels.guard)) {
        return { status: 'not-ready' };
      }
      const liveDocument = ctx.getDocument();
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
      const txn = ctx.begin({ historyBytes: addedLayerHistoryBytes(rect) });
      if (!('publish' in txn)) {
        return { status: layerEditRefusal(txn.status) };
      }
      try {
        if (!txn.reserveRaster(rgbaBytes(rect))) {
          return { status: 'over-budget' };
        }
        const pixels = this.deps.backend.createSurface(rect.width, rect.height);
        compositeDocument(
          pixels,
          {
            ...document,
            stacks: {
              ...createEmptyStacks(),
              raster: filterRasterForest(document.stacks.raster, new Set(contributors.map((layer) => layer.id))),
            },
          },
          this.deps.layers,
          { a: 1, b: 0, c: 0, d: 1, e: -rect.x, f: -rect.y },
          {
            adjustedSurface: this.deps.getAdjustedSurface,
            groupSurface: this.deps.getGroupSurface,
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
