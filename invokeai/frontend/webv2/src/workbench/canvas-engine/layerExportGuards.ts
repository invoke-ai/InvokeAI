import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { isEmpty } from '@workbench/canvas-engine/math/rect';

/** Structural equality for JSON-safe canvas contracts (including synthetic mask paint sources). */
export const isDeeplyEqual = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) {
    return true;
  }
  if (typeof left !== 'object' || left === null || typeof right !== 'object' || right === null) {
    return false;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => isDeeplyEqual(value, right[index]))
    );
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord);
  const rightKeys = Object.keys(rightRecord);
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key) =>
        Object.prototype.hasOwnProperty.call(rightRecord, key) && isDeeplyEqual(leftRecord[key], rightRecord[key])
    )
  );
};

/** Polygon shapes have no raster path, so they can never back an export. */
export const isSupportedExportSource = (source: CanvasLayerSourceContract): boolean => {
  if (source.type === 'shape') {
    return source.kind !== 'polygon';
  }
  return true;
};

export interface CreateLayerExportGuardsDeps {
  readonly projectId: string;
  readonly layerCache: LayerCacheStore;
  readonly isDisposed: () => boolean;
  readonly hasCanvasState: () => boolean;
  readonly getDocumentGeneration: () => number;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
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
 * The engine's answer to "are this layer's cached pixels trustworthy right now?"
 *
 * Every async commit in the engine — filter, generation, SAM, merge, export —
 * captures a guard before it starts and re-checks it before it publishes. The
 * guard is deliberately identity-based: it holds the layer *object* the caller
 * saw, so any mutation that replaces the contract invalidates it even when the
 * id and the pixels are unchanged. Cache version, document generation and
 * project id cover the rest, and disposal or a torn-down canvas state
 * invalidates everything at once.
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
      isDeeplyEqual(job.source, source)
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
    const liveLayer = document?.layers.find((candidate) => candidate.id === guard.layerId);
    const entry = layerCache.get(guard.layerId);
    return !!entry && liveLayer === guard.layer && entry.version === guard.cacheVersion;
  };

  return {
    captureCurrentLayerExportGuard: (layerId) => {
      const document = deps.getDocument();
      const layer = document?.layers.find((candidate) => candidate.id === layerId);
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
      const layer = doc?.layers.find((candidate) => candidate.id === layerId);
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
