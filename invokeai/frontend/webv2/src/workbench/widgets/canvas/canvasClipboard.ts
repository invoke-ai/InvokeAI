/**
 * System-clipboard access for canvas pixel selections.
 *
 * The engine deliberately stops at `Blob` / `ImageData` — `canvas-engine` may
 * not reach `workbench/widgets`, and clipboard permissions are a DOM concern
 * either way. This module is the widget-side bridge: it reads an image off the
 * clipboard and decodes it to `ImageData` the engine can paste, and re-exports
 * the existing write helper so a canvas caller has one place to import from.
 *
 * Every dependency is injectable so the whole path is testable without a real
 * clipboard or canvas.
 */

export { copyBlobToClipboard } from '@workbench/widgets/layers/layerExportActions';

type ClipboardReadLike = { read(): Promise<readonly ClipboardItem[]> };

export interface ReadClipboardImageDeps {
  clipboard?: ClipboardReadLike;
}

/** MIME types accepted from the clipboard, best first. */
const IMAGE_TYPES = ['image/png', 'image/webp', 'image/jpeg'] as const;

/**
 * The first image on the clipboard, or `null` when it holds none. Returns
 * `null` rather than throwing when the clipboard is unavailable or permission
 * was refused — a paste that finds nothing is an ordinary outcome, not an error.
 */
export const readClipboardImage = async (deps: ReadClipboardImageDeps = {}): Promise<Blob | null> => {
  const clipboard = deps.clipboard ?? (globalThis.navigator?.clipboard as ClipboardReadLike | undefined);
  if (!clipboard?.read) {
    return null;
  }
  let items: readonly ClipboardItem[];
  try {
    items = await clipboard.read();
  } catch {
    // Permission refused, or the document is not focused. Nothing to paste.
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
