/**
 * Bridge clipboard images and engine Blob/ImageData on the widget side; inject dependencies for tests without real
 * clipboard/canvas access.
 */

import { type ClipboardAccess, ClipboardUnavailableError, readClipboardItems } from '@platform/browser/clipboard';

/** MIME types accepted from the clipboard, best first. */
const IMAGE_TYPES = ['image/png', 'image/webp', 'image/jpeg'] as const;

/**
 * The first image on the clipboard, or `null` when it holds none or cannot be read (permission refused, document not
 * focused): a paste that finds nothing is an ordinary outcome. Rejects with `ClipboardUnavailableError` when the
 * browser has no clipboard API (plain HTTP), so paste can say why, and with the item's own error when an image it
 * lists cannot be retrieved.
 */
export const readClipboardImage = async (deps: ClipboardAccess = {}): Promise<Blob | null> => {
  let items: readonly ClipboardItem[];
  try {
    items = await readClipboardItems(deps);
  } catch (error) {
    if (error instanceof ClipboardUnavailableError) {
      throw error;
    }
    return null;
  }
  for (const item of items) {
    const type = IMAGE_TYPES.find((candidate) => item.types.includes(candidate));
    if (type) {
      return item.getType(type);
    }
  }
  return null;
};

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
