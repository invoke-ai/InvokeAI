import type { SelectObjectStartContext } from '@workbench/canvas-engine/applicationHost';
import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';

/** The Select Object preview tint, composited over the mask's alpha coverage. */
const SELECT_OBJECT_PREVIEW_TINT = '#38bdf8';

export interface CreateSelectObjectBridgeDeps {
  readonly layerCache: LayerCacheStore;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly captureGuard: (layerId: string) => LayerExportGuard | null;
  readonly decodeImage: (
    image: CanvasImageRef,
    options: { scaleToImage?: boolean; signal?: AbortSignal; validateDecoded?: (w: number, h: number) => void }
  ) => Promise<DecodeImageResult>;
}

export interface SelectObjectBridge {
  /** Decodes a SAM result into a tinted preview surface, or throws on abort/dimension mismatch. */
  decodeSelectObjectPreview(result: { image: CanvasImageRef; rect: Rect }, signal: AbortSignal): Promise<RasterSurface>;
  /** Gates a Select Object run on the layer being present, supported, editable and rasterized. */
  prepareSelectObjectStart(layerId: string): SelectObjectStartContext;
}

/**
 * The engine's half of the Select Object (SAM) flow.
 *
 * Both halves are policy rather than wiring. The decode validates that what
 * came back actually matches the mask it was asked for — a SAM output whose
 * dimensions disagree with the preview rect would silently paint the wrong
 * region, so it fails loudly with a tagged error the application layer can
 * report. The start gate mirrors the refusal reasons the UI shows, and computes
 * the layer's document-space bounds so the caller does not have to re-derive
 * them from the transform.
 */
export const createSelectObjectBridge = (deps: CreateSelectObjectBridgeDeps): SelectObjectBridge => ({
  decodeSelectObjectPreview: async (result, signal) => {
    const validateDecoded = (width: number, height: number): void => {
      const valid =
        Number.isInteger(width) &&
        width > 0 &&
        Number.isInteger(height) &&
        height > 0 &&
        width === result.image.width &&
        height === result.image.height &&
        width === result.rect.width &&
        height === result.rect.height;
      if (!valid) {
        throw Object.assign(
          new Error(
            `Decoded Select Object preview dimensions ${String(width)}x${String(height)} do not match SAM output ${result.image.width}x${result.image.height} and preview rect ${result.rect.width}x${result.rect.height}.`
          ),
          { samErrorCode: 'output-dimension' as const }
        );
      }
    };
    const decoded = await deps.decodeImage(result.image, { scaleToImage: false, signal, validateDecoded });
    if (decoded.status !== 'ok') {
      throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
    }
    const surface = decoded.surface;
    surface.ctx.globalCompositeOperation = 'source-in';
    surface.ctx.fillStyle = SELECT_OBJECT_PREVIEW_TINT;
    surface.ctx.fillRect(0, 0, surface.width, surface.height);
    surface.ctx.globalCompositeOperation = 'source-over';
    if (signal.aborted) {
      throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
    }
    return surface;
  },

  prepareSelectObjectStart: (layerId) => {
    const document = deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (layer.type !== 'raster' && layer.type !== 'control') {
      return { status: 'unsupported' };
    }
    if (!layer.isEnabled) {
      return { status: 'disabled' };
    }
    if (layer.isLocked) {
      return { status: 'locked' };
    }
    const guard = deps.captureGuard(layer.id);
    const entry = deps.layerCache.get(layer.id);
    if (!guard || !entry) {
      return { status: 'not-ready' };
    }
    const sourceRect = roundOut(
      transformBounds(
        fromTRS(
          { x: layer.transform.x, y: layer.transform.y },
          layer.transform.rotation,
          layer.transform.scaleX,
          layer.transform.scaleY
        ),
        entry.rect
      )
    );
    if (isEmpty(sourceRect)) {
      return { status: 'not-ready' };
    }
    return { guard, layerId, layerName: layer.name, layerType: layer.type, sourceRect, status: 'ready' };
  },
});
