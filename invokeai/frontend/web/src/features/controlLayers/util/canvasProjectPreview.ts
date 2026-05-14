import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { Rect } from 'features/controlLayers/store/types';

const PREVIEW_MAX_SIZE = 512;
const PREVIEW_MIME = 'image/webp';
const PREVIEW_QUALITY = 0.8;

/**
 * Renders the current canvas bbox as a downscaled WebP preview blob (≤512 px longside),
 * for use as the server-side gallery thumbnail of a saved canvas project.
 *
 * Returns `null` if the canvas has nothing to render (empty bbox, no visible raster layers).
 */
export const renderCanvasProjectPreview = async (canvasManager: CanvasManager): Promise<Blob | null> => {
  const bbox = canvasManager.stateApi.getBbox();
  const rect: Rect = bbox.rect;
  if (rect.width === 0 || rect.height === 0) {
    return null;
  }

  const rasterAdapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');
  if (rasterAdapters.length === 0) {
    return null;
  }

  const sourceCanvas = canvasManager.compositor.getCompositeCanvas(rasterAdapters, rect);

  // Scale down to PREVIEW_MAX_SIZE longside (no upscale).
  const longside = Math.max(sourceCanvas.width, sourceCanvas.height);
  const scale = longside > PREVIEW_MAX_SIZE ? PREVIEW_MAX_SIZE / longside : 1;
  const targetWidth = Math.max(1, Math.round(sourceCanvas.width * scale));
  const targetHeight = Math.max(1, Math.round(sourceCanvas.height * scale));

  const target = document.createElement('canvas');
  target.width = targetWidth;
  target.height = targetHeight;
  const ctx = target.getContext('2d');
  if (!ctx) {
    return null;
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);

  return await new Promise<Blob | null>((resolve) => {
    target.toBlob(
      (blob) => {
        resolve(blob);
      },
      PREVIEW_MIME,
      PREVIEW_QUALITY
    );
  });
};
