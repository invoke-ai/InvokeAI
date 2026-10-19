import type {
  FilterPreviewInput,
  LayerExportGuard,
  StagedPreviewInput,
  StagedPreviewPlacement,
} from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { PreviewStateController } from '@workbench/canvas-engine/controllers/previewStateController';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { ImageResolver } from '@workbench/canvas-engine/render/rasterizers';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { LayerFilterOutputDimensionError } from '@workbench/canvas-engine/filterError';

import { createStagedPreviewBlobCache } from './stagedPreviewBlobCache';

/** Outcome of a guarded filter publish, mirroring what the filter session reports upward. */
export type FilterPreviewOutcome = 'shown' | 'missing' | 'stale';

interface DecodedPreview {
  surface: RasterSurface;
  width: number;
  height: number;
  placement?: StagedPreviewPlacement;
}

export interface CreatePreviewPublisherDeps {
  readonly previews: PreviewStateController;
  readonly resolveImage: ImageResolver;
  readonly decodeBlob: (
    blob: Blob,
    dimensions?: { width: number; height: number; scale?: boolean }
  ) => Promise<{ surface: RasterSurface; decodedWidth: number; decodedHeight: number }>;
  readonly invalidateAll: () => void;
  readonly invalidateLayer: (layerId: string) => void;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
}

export interface PreviewPublisher {
  /** Drops any staged preview and bumps the token so an in-flight decode is discarded. */
  clearStagedPreview(): void;
  /** Releases staged fetch/decode coalescing state while keeping the publisher reusable. */
  clearStagedPreviewCache(): void;
  /** Releases cached staged-result bytes and aborts outstanding prefetches. */
  dispose(): void;
  /** Warms one full-resolution staged result without decoding it. */
  preloadStagedPreview(imageName: string): void;
  /** Decodes and publishes a staged-generation preview; `null` clears. Never throws. */
  setStagedPreview(input: StagedPreviewInput | null): void;
  /** Drops one layer's filter preview and invalidates it. */
  clearFilterPreview(layerId: string): void;
  /** Drops every layer's filter preview — for a wholesale document replace. */
  clearAllFilterPreviews(): void;
  /** Decodes and publishes a filter preview, re-validating the guard either side of the decode. */
  setGuardedFilterPreview(
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<FilterPreviewOutcome>;
}

/** DOM-free data-URL decoding lets injected-backend progress previews run in node tests. */
const dataUrlToBlob = (dataUrl: string): Blob => {
  const commaIndex = dataUrl.indexOf(',');
  const header = commaIndex >= 0 ? dataUrl.slice(0, commaIndex) : '';
  const data = commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
  const mime = /^data:([^;,]+)/i.exec(header)?.[1] ?? 'application/octet-stream';
  if (/;base64/i.test(header)) {
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: mime });
  }
  return new Blob([decodeURIComponent(data)], { type: mime });
};

/**
 * Staged and filter previews token-guard async decodes against newer sets/clears. Filters also recheck layer
 * existence and export guards after decode. Typed-filter dimension mismatches are contract errors, not silent
 * stale results.
 */
export const createPreviewPublisher = (deps: CreatePreviewPublisherDeps): PreviewPublisher => {
  const { previews } = deps;
  const stagedPreviewBlobs = createStagedPreviewBlobCache(deps.resolveImage);
  const stagedImageDecodes = new Map<string, ReturnType<CreatePreviewPublisherDeps['decodeBlob']>>();

  const clearStagedPreview = (): void => {
    if (previews.clearStaged()) {
      deps.invalidateAll();
    }
  };

  const clearStagedPreviewCache = (): void => {
    stagedPreviewBlobs.clear();
    stagedImageDecodes.clear();
  };

  const dispose = (): void => {
    stagedPreviewBlobs.dispose();
    stagedImageDecodes.clear();
  };

  /** Decodes a preview input to a surface (imageName via resolver, dataUrl via the backend seam). */
  const decodePreview = async (
    input: StagedPreviewInput,
    useStagedBlobCache = false,
    isCurrent?: () => boolean
  ): Promise<DecodedPreview> => {
    if ('imageName' in input) {
      const blob = useStagedBlobCache
        ? await stagedPreviewBlobs.get(input.imageName)
        : await deps.resolveImage(input.imageName);
      // Blob resolution is an async boundary. A rapid selection change may
      // supersede this request while it waits; do not spend a full bitmap
      // decode on a candidate that can no longer be published.
      if (isCurrent && !isCurrent()) {
        throw new Error('Staged preview request was superseded before decode.');
      }
      let pendingDecode = useStagedBlobCache ? stagedImageDecodes.get(input.imageName) : undefined;
      if (!pendingDecode) {
        pendingDecode = deps.decodeBlob(blob);
        if (useStagedBlobCache) {
          stagedImageDecodes.set(input.imageName, pendingDecode);
          const forget = (): void => {
            if (stagedImageDecodes.get(input.imageName) === pendingDecode) {
              stagedImageDecodes.delete(input.imageName);
            }
          };
          // The awaiting caller owns rejection handling; this side chain only
          // bounds the coalescing map to the lifetime of the decode.
          void pendingDecode.then(forget, forget);
        }
      }
      const decoded = await pendingDecode;
      return {
        height: decoded.decodedHeight,
        placement: input.placement ? { ...input.placement } : undefined,
        surface: decoded.surface,
        width: decoded.decodedWidth,
      };
    }
    const { dataUrl, height, width } = input;
    const decoded = await deps.decodeBlob(dataUrlToBlob(dataUrl), { height, scale: true, width });
    return { height, surface: decoded.surface, width };
  };

  /**
   * Drop preview state and increment, never delete/reset, its token so late decodes cannot revive previews after
   * layer-id reuse.
   */
  const clearFilterPreview = (layerId: string): void => {
    if (previews.clearFilter(layerId)) {
      deps.invalidateLayer(layerId);
    }
  };

  const publishFilterPreview = async (
    layerId: string,
    input: FilterPreviewInput,
    validate: () => FilterPreviewOutcome,
    guard: LayerExportGuard
  ): Promise<FilterPreviewOutcome> => {
    const nextToken = previews.beginGuardedFilter(layerId);
    const dropGuardedRequest = (): void => {
      previews.finishGuardedFilter(layerId, nextToken);
    };
    const beforeDecode = validate();
    if (beforeDecode !== 'shown') {
      dropGuardedRequest();
      return beforeDecode;
    }
    try {
      const decoded = await decodePreview({ imageName: input.imageName });
      if (input.filterType && (decoded.width !== input.rect.width || decoded.height !== input.rect.height)) {
        throw new LayerFilterOutputDimensionError(
          input.filterType,
          { height: decoded.height, width: decoded.width },
          input.rect
        );
      }
      const beforePublish = validate();
      if (beforePublish !== 'shown') {
        dropGuardedRequest();
        return beforePublish;
      }
      // A newer set/clear for THIS layer superseded the decode in flight.
      if (!previews.isFilterTokenCurrent(layerId, nextToken)) {
        dropGuardedRequest();
        return 'stale';
      }
      previews.publishFilter(layerId, nextToken, { guard, rect: { ...input.rect }, surface: decoded.surface });
      deps.invalidateLayer(layerId);
      return 'shown';
    } catch (error) {
      // Transient decode failure leaves any prior preview untouched.
      dropGuardedRequest();
      if (error instanceof LayerFilterOutputDimensionError) {
        throw error;
      }
      return 'stale';
    }
  };

  return {
    clearAllFilterPreviews: () => {
      for (const id of previews.filterLayerIds()) {
        clearFilterPreview(id);
      }
    },

    clearFilterPreview,
    clearStagedPreview,
    clearStagedPreviewCache,
    dispose,
    preloadStagedPreview: stagedPreviewBlobs.preload,

    setGuardedFilterPreview: (layerId, input, guard) => {
      const validate = (): FilterPreviewOutcome => {
        const liveLayer = getDocumentLayer(deps.getDocument(), layerId);
        if (!liveLayer) {
          return 'missing';
        }
        if (layerId !== guard.layerId || !deps.isGuardCurrent(guard)) {
          return 'stale';
        }
        return 'shown';
      };
      return publishFilterPreview(layerId, input, validate, guard);
    },

    setStagedPreview: (input) => {
      if (input === null) {
        clearStagedPreview();
        return;
      }
      const token = previews.nextStagedToken();
      decodePreview(input, true, () => previews.isStagedTokenCurrent(token))
        .then((decoded) => {
          // A newer set/clear superseded this decode while it was in flight.
          if (previews.publishStaged(token, decoded)) {
            deps.invalidateAll();
          }
        })
        .catch(() => {
          // A transient decode failure leaves any prior preview untouched rather
          // than blanking the canvas; the next selection re-drives a decode.
        });
    },
  };
};
