import { describe, expect, it, vi } from 'vitest';

import { decodeImageBlob } from './canvasClipboard';

const blob = (type: string): Blob => ({ type }) as Blob;

describe('decodeImageBlob', () => {
  const imageData = { data: new Uint8ClampedArray(4), height: 1, width: 1 } as ImageData;

  const fakeCanvas = (pixels: ImageData | null = imageData) => {
    const drawImage = vi.fn();
    const canvas = {
      getContext: () => (pixels ? { drawImage, getImageData: () => pixels } : null),
    } as unknown as HTMLCanvasElement;
    return { canvas, drawImage };
  };

  const bitmap = (width: number, height: number) => {
    const close = vi.fn();
    return { bitmap: { close, height, width } as unknown as ImageBitmap, close };
  };

  it('decodes a blob to ImageData', async () => {
    const { bitmap: bmp } = bitmap(4, 3);
    const { canvas, drawImage } = fakeCanvas();

    await expect(
      decodeImageBlob(blob('image/png'), {
        createCanvas: () => canvas,
        createImageBitmap: () => Promise.resolve(bmp),
      })
    ).resolves.toBe(imageData);
    expect(drawImage).toHaveBeenCalledWith(bmp, 0, 0);
  });

  it('releases the decoded bitmap even when the draw path bails out', async () => {
    const { bitmap: bmp, close } = bitmap(4, 3);
    const { canvas } = fakeCanvas(null);

    await expect(
      decodeImageBlob(blob('image/png'), {
        createCanvas: () => canvas,
        createImageBitmap: () => Promise.resolve(bmp),
      })
    ).resolves.toBeNull();
    expect(close).toHaveBeenCalled();
  });

  it('returns null for an undecodable blob', async () => {
    await expect(
      decodeImageBlob(blob('image/png'), { createImageBitmap: () => Promise.reject(new Error('bad')) })
    ).resolves.toBeNull();
  });

  it('returns null for a zero-sized image', async () => {
    const { bitmap: bmp } = bitmap(0, 0);
    await expect(
      decodeImageBlob(blob('image/png'), { createImageBitmap: () => Promise.resolve(bmp) })
    ).resolves.toBeNull();
  });

  it('returns null when decoding is unavailable', async () => {
    await expect(decodeImageBlob(blob('image/png'), { createImageBitmap: undefined })).resolves.toBeNull();
  });
});
