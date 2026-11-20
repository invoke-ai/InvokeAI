import type { CanvasSelectionCapability } from '@workbench/canvas-engine/api';

import { type ClipboardAccess, ClipboardUnavailableError, readClipboardItems } from '@platform/browser/clipboard';

/** MIME types accepted from the clipboard, best first. */
const IMAGE_TYPES = ['image/png', 'image/webp', 'image/jpeg'] as const;

/** The first clipboard image, or null when none is readable; rejects `ClipboardUnavailableError` without the API. */
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

/** Copies the selection; a cut checks the removal can run first, so a refused cut writes nothing. */
export const copyCanvasSelection = async (
  selection: Pick<CanvasSelectionCapability, 'admitSelectionErase' | 'eraseSelection' | 'exportSelectionBlob'>,
  cut: boolean,
  write: (blob: Blob) => Promise<void>
): Promise<void> => {
  if (cut && !selection.admitSelectionErase()) {
    return;
  }
  const blob = await selection.exportSelectionBlob();
  if (!blob) {
    return;
  }
  await write(blob);
  if (cut) {
    selection.eraseSelection();
  }
};
