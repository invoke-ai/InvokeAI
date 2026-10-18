/**
 * Bridge clipboard images and engine Blob/ImageData on the widget side; inject dependencies for tests without real
 * canvas access.
 */

export { copyBlobToClipboard } from '@workbench/widgets/layers/layerExportActions';

export interface DecodeImageBlobDeps {
  createImageBitmap?: (blob: Blob) => Promise<ImageBitmap>;
  createCanvas?: (width: number, height: number) => HTMLCanvasElement | OffscreenCanvas;
}

const defaultCreateCanvas = (width: number, height: number): HTMLCanvasElement | OffscreenCanvas => {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  }
  const canvas = globalThis.document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
};

/** Decodes an image blob to `ImageData`, or `null` when it cannot be decoded. */
export const decodeImageBlob = async (blob: Blob, deps: DecodeImageBlobDeps = {}): Promise<ImageData | null> => {
  const decode = deps.createImageBitmap ?? globalThis.createImageBitmap;
  if (!decode) {
    return null;
  }
  let bitmap: ImageBitmap;
  try {
    bitmap = await decode(blob);
  } catch {
    return null;
  }
  try {
    if (bitmap.width <= 0 || bitmap.height <= 0) {
      return null;
    }
    const canvas = (deps.createCanvas ?? defaultCreateCanvas)(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (!ctx) {
      return null;
    }
    ctx.drawImage(bitmap, 0, 0);
    return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  } finally {
    bitmap.close?.();
  }
};
