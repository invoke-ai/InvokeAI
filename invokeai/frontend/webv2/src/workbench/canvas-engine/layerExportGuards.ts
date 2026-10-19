import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import {
  getSourceContentRect,
  isEmptyPolygonShape,
  renderableSourceOf,
} from '@workbench/canvas-engine/document/sources';
import { isEmpty } from '@workbench/canvas-engine/math/rect';

/** A pointless polygon shape has no raster, so it can never back an export. */
export const isSupportedExportSource = (source: CanvasLayerSourceContract): boolean =>
  source.type !== 'shape' || !isEmptyPolygonShape(source);

/** A raster layer a PSD or raster export can contain; an empty layer is still counted and handled at export time. */
export const isExportableRasterLayer = (layer: CanvasLayerContract): boolean =>
  layer.type === 'raster' && isSupportedExportSource(layer.source);

export interface CreateLayerExportGuardsDeps {
  readonly projectId: string;
  readonly layerCache: LayerCacheStore;
  readonly isDisposed: () => boolean;
  readonly hasCanvasState: () => boolean;
  readonly getDocumentGeneration: () => number;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly getRasterizationJob: (layerId: string) => RasterizationJob | undefined;
}

export interface LayerExportGuards {
  /** True while a rasterization for this layer's *current* source and cache version is running. */
  isCurrentRasterizationJob(layer: CanvasLayerContract): boolean;
  /** Stamps a guard from a layer and cache entry the caller already holds. */
  captureLayerExportGuard(layer: CanvasLayerContract, entry: LayerCacheEntry): LayerExportGuard;
  /** True when everything the guard was stamped against is still exactly as it was. */
  isLayerExportGuardCurrent(guard: LayerExportGuard): boolean;
  /** Stamps a guard for a layer, or `null` if its pixels are not trustworthy right now. */
  captureCurrentLayerExportGuard(layerId: string): LayerExportGuard | null;
  /** True when the layer has pixels an export would actually find. */
  hasExportableLayerContent(layerId: string): boolean;
}

/**
 * Async pixel guards capture layer object identity, cache version, document generation and project id, then
 * recheck before publication. Any replacement, disposal or teardown invalidates the guard, even if pixels look
 * unchanged.
 */
export const createLayerExportGuards = (deps: CreateLayerExportGuardsDeps): LayerExportGuards => {
  const { layerCache } = deps;

  const isCurrentRasterizationJob = (layer: CanvasLayerContract): boolean => {
    const job = deps.getRasterizationJob(layer.id);
    const source = renderableSourceOf(layer);
    return (
      !!job &&
      !!source &&
      job.version === layerCache.version(layer.id) &&
      job.documentGeneration === deps.getDocumentGeneration() &&
      areJsonValuesStructurallyEqual(job.source, source)
    );
  };

  const captureLayerExportGuard = (layer: CanvasLayerContract, entry: LayerCacheEntry): LayerExportGuard => ({
    cacheVersion: entry.version,
    documentGeneration: deps.getDocumentGeneration(),
    layer,
    layerId: layer.id,
    projectId: deps.projectId,
  });

  const isLayerExportGuardCurrent = (guard: LayerExportGuard): boolean => {
    if (
      deps.isDisposed() ||
      guard.projectId !== deps.projectId ||
      !deps.hasCanvasState() ||
      guard.documentGeneration !== deps.getDocumentGeneration()
    ) {
      return false;
    }
    const document = deps.getDocument();
    const liveLayer = getDocumentLayer(document, guard.layerId);
    const entry = layerCache.get(guard.layerId);
    return !!entry && liveLayer === guard.layer && entry.version === guard.cacheVersion;
  };

  return {
    captureCurrentLayerExportGuard: (layerId) => {
      const document = deps.getDocument();
      const layer = getDocumentLayer(document, layerId);
      const source = layer ? renderableSourceOf(layer) : null;
      const entry = layerCache.get(layerId);
      if (
        !layer ||
        !source ||
        !isSupportedExportSource(source) ||
        !entry ||
        entry.stale ||
        isCurrentRasterizationJob(layer) ||
        isEmpty(entry.rect)
      ) {
        return null;
      }
      const guard = captureLayerExportGuard(layer, entry);
      return isLayerExportGuardCurrent(guard) ? guard : null;
    },

    captureLayerExportGuard,

    hasExportableLayerContent: (layerId) => {
      const doc = deps.getDocument();
      const layer = getDocumentLayer(doc, layerId);
      if (!doc || !layer) {
        return false;
      }
      const source = renderableSourceOf(layer);
      if (!source || !isSupportedExportSource(source)) {
        return false;
      }
      if (!isEmpty(getSourceContentRect(layer, doc))) {
        return true;
      }
      // Only paint-backed layers (including masks through renderableSourceOf) can
      // have real pixels beyond their persisted source rect. The live cache must
      // describe the current source revision and must not be mid-rasterization.
      if (source.type !== 'paint') {
        return false;
      }
      const entry = layerCache.get(layerId);
      return !!entry && !entry.stale && !isCurrentRasterizationJob(layer) && !isEmpty(entry.rect);
    },

    isCurrentRasterizationJob,
    isLayerExportGuardCurrent,
  };
};
